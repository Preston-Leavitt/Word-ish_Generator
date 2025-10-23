import os
import sys
import base64
import csv
import time
import io
import argparse
from typing import List, Tuple

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Folder containing images
DEFAULT_IMAGES_DIR = os.path.join("static", "images")


def slugify(text: str, max_len: int = 40) -> str:
    """Create a filesystem-friendly, lowercase slug."""
    import re
    text = text.strip().lower()
    # Replace non-alphanumeric with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        text = "image"
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text


def list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    files = []
    for name in os.listdir(images_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(name)
    return sorted(files)


def _load_and_downscale(image_path: str, max_dim: int) -> bytes:
    """Load an image and optionally downscale to max_dim on the longest side; return JPEG bytes."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max_dim and max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def describe_image(client: OpenAI, image_path: str, model: str, detail: str, max_dim: int) -> Tuple[str, dict]:
    """Use OpenAI Vision to get a short, family-friendly name for an image. Returns (text, usage_dict)."""
    img_bytes = _load_and_downscale(image_path, max_dim)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    system = (
        "You are an expert photo captioner. Return a short, filename-safe name "
        "for the image in 3-5 words, neutral and family-friendly. "
        "Do not include punctuation except spaces. Do not include quotes. "
        "Examples: sunset over beach; golden retriever puppy; mountain trail in fog."
    )
    user = "Name this image succinctly. Return only the name."

    msg_content = [
        {"type": "text", "text": user},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": msg_content},
        ],
        temperature=0.4,
        max_tokens=32,
    )
    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }
    return text, usage_dict


def ensure_unique_name(dst_dir: str, base_slug: str, ext: str) -> str:
    name = f"{base_slug}{ext}"
    if not os.path.exists(os.path.join(dst_dir, name)):
        return name
    i = 1
    while True:
        candidate = f"{base_slug}-{i}{ext}"
        if not os.path.exists(os.path.join(dst_dir, candidate)):
            return candidate
        i += 1


def main():
    load_dotenv()  # load OPENAI_API_KEY from .env if present
    parser = argparse.ArgumentParser(description="Name images in static/images using OpenAI Vision")
    parser.add_argument("--images_dir", default=DEFAULT_IMAGES_DIR, help="Directory of images to process")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (overrides env/.env if provided)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--apply", action="store_true", help="Actually rename files. If not set, do a dry run.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls in seconds")
    parser.add_argument("--out_csv", default="image_renames.csv", help="Write mapping old->new to this CSV")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all)")
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="low", help="Vision detail level")
    parser.add_argument("--max_dim", type=int, default=512, help="Resize longest side to this many px (0 = no resize)")
    parser.add_argument("--estimate_only", type=int, default=0, help="Run on the first N images, show token usage summary and exit (no renames)")
    parser.add_argument("--in_price_per_1k", type=float, default=0.0, help="USD price per 1K prompt tokens for your chosen model (optional for cost calc)")
    parser.add_argument("--out_price_per_1k", type=float, default=0.0, help="USD price per 1K completion tokens for your chosen model (optional for cost calc)")
    args = parser.parse_args()

    images_dir = args.images_dir
    if not os.path.isdir(images_dir):
        print(f"Images dir not found: {images_dir}")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.\nSet it in your environment, put it in a .env file, or pass --api_key.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    files = list_images(images_dir)
    if not files:
        print("No images found.")
        return
    # Apply limits
    if args.estimate_only > 0:
        files = files[: args.estimate_only]
    elif args.limit > 0:
        files = files[: args.limit]

    mappings: List[Tuple[str, str, str]] = []  # (old_name, suggested_name, new_file)
    total_prompt = 0
    total_completion = 0

    for i, name in enumerate(files, 1):
        src = os.path.join(images_dir, name)
        ext = os.path.splitext(name)[1].lower()
        try:
            suggestion, usage = describe_image(client, src, args.model, args.detail, args.max_dim)
            base = slugify(suggestion)
            new_name = ensure_unique_name(images_dir, base, ext)
            mappings.append((name, suggestion, new_name))
            print(f"[{i}/{len(files)}] {name} -> '{suggestion}' -> {new_name}")
            if usage and usage.get("prompt_tokens") is not None:
                total_prompt += usage["prompt_tokens"]
            if usage and usage.get("completion_tokens") is not None:
                total_completion += usage["completion_tokens"]
            if args.apply and new_name != name:
                os.rename(src, os.path.join(images_dir, new_name))
        except Exception as e:
            print(f"[{i}/{len(files)}] {name} -> ERROR: {e}")
        time.sleep(args.delay)

    # Write CSV mapping
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_filename", "suggested_name", "new_filename"])
        writer.writerows(mappings)

    # Token/cost summary
    total_images = len(files)
    print("\nSummary:")
    print(f"Processed images: {total_images}")
    print(f"Total prompt tokens: {total_prompt}")
    print(f"Total completion tokens: {total_completion}")
    if args.in_price_per_1k > 0 or args.out_price_per_1k > 0:
        in_cost = (total_prompt / 1000.0) * args.in_price_per_1k
        out_cost = (total_completion / 1000.0) * args.out_price_per_1k
        print(f"Approx input token cost:  ${in_cost:.4f}")
        print(f"Approx output token cost: ${out_cost:.4f}")
        print(f"Approx total cost:        ${in_cost + out_cost:.4f}")

    print("\nDone.")
    print(f"Wrote mapping CSV to {args.out_csv}")
    if args.estimate_only > 0:
        print("Estimate only: no renames were applied.")
        return
    if not args.apply:
        print("Dry run only. Re-run with --apply to rename files.")


if __name__ == "__main__":
    main()
