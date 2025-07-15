import argparse
import sys
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)

def load_and_prepare(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df[['word', 'definition']])
    def preprocess(examples):
        return {
            "input_text": [f"define: {w}" for w in examples["word"]],
            "target_text": examples["definition"],
        }
    return ds.map(preprocess, batched=True)

def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tokenize(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=32,
            truncation=True,
            padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=64,
                truncation=True,
                padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(_tokenize, batched=True, remove_columns=ds.column_names)

def train_and_save(args):
    # prepare data
    raw_ds = load_and_prepare(args.csv)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokenized = tokenize_dataset(raw_ds, tokenizer)

    # set up model + training arguments
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
    )

    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.shuffle(seed=42).select(range(0, int(0.8*len(tokenized)))),
        eval_dataset=tokenized.shuffle(seed=42).select(range(int(0.8*len(tokenized)), len(tokenized))),
        tokenizer=tokenizer,
    )
    trainer.train()

    # save model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model & tokenizer saved under {args.output_dir}")

def interactive_generate(model, tokenizer, device):
    print("\nEnter words to generate fake definitions. Type 'exit' to quit.")
    while True:
        word = input(">> ").strip()
        if word.lower() in ("exit", "quit"):
            sys.exit()
        FEW = """
        {word} refers to
        """

        prompt = FEW.format(word=word)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            input_ids,
            max_length=60,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.8,
            num_return_sequences=1,
        )
        print("  " + tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n")

def load_and_generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    print(f"Loaded model from {args.model_dir} on {device}")
    interactive_generate(model, tokenizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake definition generator")
    parser.add_argument("--mode", choices=["train", "generate"], required=True,
                        help="train: fine-tune+save | generate: load+interactive REPL")
    parser.add_argument("--csv", type=str, default="dict_cleaned.csv",
                        help="CSV path for training")
    parser.add_argument("--output_dir", type=str, default="fake_def_model",
                        help="Where to save model+tokenizer")
    parser.add_argument("--model_dir", type=str, default="fake_def_model",
                        help="Where to load model+tokenizer for generation")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs")
    args = parser.parse_args()

    if args.mode == "train":
        train_and_save(args)
    else:
        load_and_generate(args)