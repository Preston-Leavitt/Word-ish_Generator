#!/usr/bin/env python3
"""
finetune_runner.py (updated for openai>=1.0.0)

Usage:
  export OPENAI_API_KEY="sk-..."
  python finetune_runner.py --train train.jsonl --valid valid.jsonl --model gpt-3.5-turbo \
      --n_epochs 4 --learning_rate_multiplier 0.1

Requirements:
  pip install openai pandas tqdm

What it does:
  - Uploads train + validation JSONL to OpenAI (purpose=fine-tune)
  - Starts a supervised fine-tune job with chosen base model
  - Polls job status until completion
  - Prints the final fine-tuned model name
  - Optionally runs a quick evaluation loop on test.jsonl and prints simple accuracy (classification)
"""

import os, time, json, argparse
from tqdm import tqdm  # if not used elsewhere you may remove
import pandas as pd    # (still optional; kept for parity)
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI

# --- New style OpenAI client import ---
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Install new SDK: pip install --upgrade openai")

load_dotenv()
client: Optional[OpenAI] = None

# --- Supported / fallback fine-tune base models (ordered by preference) ---
SUPPORTED_FT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "babbage-002",
    "davinci-002"
]
# --- New: detect chat (messages-based) fine-tune models ---
def _is_chat_model(model: str) -> bool:
    return model.startswith("gpt-3.5-turbo")

# ---- helpers -----------------------------------------------------------------
def _ensure_client():
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
    return client

def upload_file(path):
    """Upload file with purpose='fine-tune' using new Files API."""
    _client = _ensure_client()
    print(f"Uploading {path} ...")
    with open(path, "rb") as f:
        resp = _client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded {path} -> file id {resp.id}")
    return resp.id

# --- New: dataset conversion helpers --------------------------------------- #
def _line_has_messages(rec: dict) -> bool:
    return isinstance(rec, dict) and "messages" in rec and isinstance(rec["messages"], list)

def _convert_pc_record_to_messages(rec: dict) -> dict:
    """
    Convert {"prompt": "...", "completion": " ..."} to chat format.
    Leading space in completion retained (trim if undesired).
    """
    prompt = rec.get("prompt", "").strip()
    completion = rec.get("completion", "")
    # Keep original leading space (classification label) but strip right whitespace
    completion_out = completion.rstrip("\n")
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful model that predicts or generates LinkedIn engagement outputs."
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": completion_out.lstrip()  # remove leading space for assistant message clarity
            }
        ]
    }

def _convert_file_if_needed(in_path: str, model: str, tag: str) -> str:
    """
    If model requires messages[] and file is in prompt/completion format, convert.
    Returns path to (possibly new) file.
    """
    if not _is_chat_model(model):
        return in_path  # legacy instruct models keep prompt/completion
    out_path = f"{in_path.rsplit('.jsonl',1)[0]}._chat_{tag}.jsonl"
    needs_conversion = False
    # Peek first few lines
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                needs_conversion = True
                break
            if not _line_has_messages(rec):
                needs_conversion = True
            break
    if not needs_conversion:
        print(f"[INFO] {tag}: already in messages[] format; no conversion needed.")
        return in_path

    print(f"[INFO] {tag}: converting prompt/completion -> messages[] for chat fine-tune ...")
    converted = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if _line_has_messages(rec):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps(_convert_pc_record_to_messages(rec), ensure_ascii=False) + "\n")
            converted += 1
    print(f"[INFO] {tag}: wrote {converted} converted examples -> {out_path}")
    return out_path

def start_fine_tune(training_file_id, validation_file_id, model, n_epochs, lr_multiplier, suffix=None):
    """Create fine-tune job (fine_tuning.jobs.create) with hyperparameters."""
    _client = _ensure_client()
    hyper = {}
    if n_epochs is not None:
        hyper["n_epochs"] = n_epochs
    if lr_multiplier is not None:
        hyper["learning_rate_multiplier"] = lr_multiplier
    params = {
        "training_file": training_file_id,
        "model": model,
        "hyperparameters": hyper
    }
    if validation_file_id:
        params["validation_file"] = validation_file_id
    if suffix:
        params["suffix"] = suffix
    print("Creating fine-tune job with params:", params)
    resp = _client.fine_tuning.jobs.create(**params)
    print("Fine-tune created, id:", resp.id)
    return resp.id

def poll_fine_tune(job_id, poll_seconds=10):
    """Poll job until terminal state."""
    _client = _ensure_client()
    print(f"Polling fine-tune status (job id: {job_id}) ...")
    last = None
    while True:
        job = _client.fine_tuning.jobs.retrieve(job_id)
        if job.status != last:
            print("Status ->", job.status)
            last = job.status
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(poll_seconds)

def _get_events(job_id, limit=100):
    _client = _ensure_client()
    try:
        ev = _client.fine_tuning.jobs.list_events(id=job_id, limit=limit)
        return ev.data or []
    except Exception:
        return []

def evaluate_model_classification(model_name, test_jsonl_path, max_examples=200):
    """
    Evaluate a chat fine-tuned model for simple classification:
    - Sends prompt as single user message
    - Compares first trimmed response token to gold completion label
    """
    _client = _ensure_client()
    print("Running simple evaluation (classification) using model:", model_name)
    total = 0
    correct = 0
    per_label = {}
    try:
        with open(test_jsonl_path, "r", encoding="utf8") as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                rec = json.loads(line)
                prompt = rec["prompt"]
                gold = rec["completion"].strip().lower()
                try:
                    resp = _client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=8
                    )
                    pred_raw = resp.choices[0].message.content or ""
                except Exception as e:
                    print("Evaluation call failed:", e)
                    break
                pred = pred_raw.strip().split()[0].lower()
                total += 1
                per_label.setdefault(gold, {"tp": 0, "count": 0})
                per_label[gold]["count"] += 1
                if pred == gold:
                    correct += 1
                    per_label[gold]["tp"] += 1
    except FileNotFoundError:
        print("Test file not found; skipping evaluation.")
        return
    if total == 0:
        print("No evaluation examples processed.")
        return
    acc = correct / total
    print(f"Eval examples: {total}, Accuracy: {acc:.4f}")
    for lbl, stats in per_label.items():
        tp = stats["tp"]; cnt = stats["count"]
        print(f"  label {lbl}: {tp}/{cnt} correct ({tp/cnt:.3f})")

def resolve_finetune_model(requested: str) -> str:
    """
    If requested model not in supported list, fall back to first available.
    (We attempt a simple heuristic: keep the user's choice if it contains a supported prefix.)
    """
    if requested in SUPPORTED_FT_MODELS:
        return requested
    # simple prefix match (e.g. user passed gpt-3.5-turbo-1106)
    for base in SUPPORTED_FT_MODELS:
        if requested.startswith(base):
            return base
    print(f"[WARN] Model '{requested}' not available for fine-tuning. Falling back to '{SUPPORTED_FT_MODELS[0]}'")
    return SUPPORTED_FT_MODELS[0]

# ---- CLI ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Upload JSONL files and run OpenAI fine-tune (new SDK).")
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--valid", required=True, help="Path to valid.jsonl")
    parser.add_argument("--test", required=False, help="Optional path to test.jsonl for quick eval")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Base model to fine-tune (supported: gpt-3.5-turbo, babbage-002, davinci-002)")
    parser.add_argument("--n_epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--learning_rate_multiplier", type=float, default=None, help="Learning rate multiplier")
    parser.add_argument("--suffix", default=None, help="Optional suffix for fine-tuned model name")
    parser.add_argument("--no_upload", action="store_true", help="Treat train/valid args as existing file IDs")
    parser.add_argument("--poll_interval", type=int, default=15, help="Seconds between status polls")
    parser.add_argument("--eval_max_examples", type=int, default=200, help="Max test examples to evaluate")
    args = parser.parse_args()

    _ensure_client()

    # --- Model resolution / fallback ---
    chosen_model = resolve_finetune_model(args.model)
    if chosen_model != args.model:
        print(f"[INFO] Using fallback base model: {chosen_model}")
        args.model = chosen_model
    else:
        print(f"[INFO] Using requested base model: {args.model}")

    # --- NEW: convert datasets if chat model & legacy format ---
    train_path_for_upload = _convert_file_if_needed(args.train, args.model, "train")
    valid_path_for_upload = _convert_file_if_needed(args.valid, args.model, "valid")

    if args.no_upload:
        train_file_id = train_path_for_upload
        valid_file_id = valid_path_for_upload
    else:
        train_file_id = upload_file(train_path_for_upload)
        valid_file_id = upload_file(valid_path_for_upload)

    job_id = start_fine_tune(
        training_file_id=train_file_id,
        validation_file_id=valid_file_id,
        model=args.model,
        n_epochs=args.n_epochs,
        lr_multiplier=args.learning_rate_multiplier,
        suffix=args.suffix
    )

    final = poll_fine_tune(job_id, poll_seconds=args.poll_interval)
    status = final.status
    if status != "succeeded":
        print("Fine-tune ended with status:", status)
        if getattr(final, "error", None):
            print("Error:", final.error)
        # Show last few events for debugging
        events = _get_events(job_id, limit=20)
        if events:
            print("Recent events:")
            for e in events[-5:]:
                print(f" - {e.level}: {e.message}")
        return

    fine_tuned_model = final.fine_tuned_model
    if not fine_tuned_model:
        # attempt extraction from events
        events = _get_events(job_id)
        for e in reversed(events):
            if "fine-tuned model" in (e.message or "").lower():
                fine_tuned_model = e.message.split()[-1]
                break

    if not fine_tuned_model:
        print("Could not determine fine-tuned model name.")
        print(final)
        return

    print("Fine-tune complete. Model name:", fine_tuned_model)

    if args.test:
        try:
            evaluate_model_classification(fine_tuned_model, args.test, max_examples=args.eval_max_examples)
        except Exception as e:
            print("Evaluation failed:", e)

if __name__ == "__main__":
    main()
