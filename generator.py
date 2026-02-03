import random
import csv
import os
from flask import Flask, render_template, url_for, request
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
CSV_ONE = "place.csv"
CSV_TWO = "relative.csv"
CACHE_FILE = "definitions_cache.csv"
ALT_CACHE_FILE = "defenitions_cache.csv"  # common misspelling; load if present
# Note: Instantiate the OpenAI client lazily to avoid requiring API key at import time.
def _load_json_safe(text: str, context_label: str = ""):
    """Try to parse a JSON object from text. If it fails, attempt a few non-destructive fallbacks.
    Returns dict on success, or None on failure (and logs a brief snippet)."""
    try:
        import json as _json
        return _json.loads(text)
    except Exception:
        pass
    try:
        # Try to extract the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            import json as _json
            snippet = text[start : end + 1]
            return _json.loads(snippet)
    except Exception:
        pass
    try:
        # Print a short preview to help debugging
        preview = text.strip().replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        if context_label:
            print(f"{context_label} JSON parse failed. Raw content preview: {preview}")
        else:
            print(f"JSON parse failed. Raw content preview: {preview}")
    except Exception:
        pass
    return None

def _extract_pairs(text: str, keys: list[str]) -> dict:
    """Best-effort extraction of simple JSON key-value pairs from a text blob.
    Looks for patterns like "key": "value" and returns a dict of any found keys.
    """
    out = {}
    try:
        import re
        for k in keys:
            m = re.search(rf'"{re.escape(k)}"\s*:\s*"([^"]+)"', text, flags=re.DOTALL)
            if m:
                out[k] = m.group(1)
    except Exception:
        pass
    return out
def _price_from_env():
    try:
        in_p = float(os.getenv("OPENAI_INPUT_PRICE_PER_1K", "0") or 0)
    except Exception:
        in_p = 0.0
    try:
        out_p = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_1K", "0") or 0)
    except Exception:
        out_p = 0.0
    return in_p, out_p

def _print_usage_cost(label: str, usage) -> None:
    try:
        in_price, out_price = _price_from_env()
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        msg = f"{label} tokens: prompt={pt} completion={ct} total={tt}"
        if (pt is not None or ct is not None) and (in_price > 0 or out_price > 0):
            cost = 0.0
            if pt is not None:
                cost += (pt / 1000.0) * in_price
            if ct is not None:
                cost += (ct / 1000.0) * out_price
            msg += f" | est cost=${cost:.4f}"
        print(msg)
    except Exception:
        pass

# === Dictionary loading to detect real words ===
_WORDS_SET = None
def _load_words_set() -> set:
    global _WORDS_SET
    if _WORDS_SET is not None:
        return _WORDS_SET
    try:
        base_dir = Path(__file__).resolve().parent
        dict_path = base_dir / "words_alpha.txt"
        words = set()
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
        _WORDS_SET = words
        print(f"Loaded dictionary with {len(words)} words from {dict_path}")
    except Exception as e:
        print(f"Dictionary load failed: {e}")
        _WORDS_SET = set()
    return _WORDS_SET

def _is_known_word(word: str) -> bool:
    try:
        return word.lower() in _load_words_set()
    except Exception:
        return False

# === Cache of definitions to avoid re-calling AI ===
def _cache_file_path() -> Path:
    return Path(__file__).resolve().parent / CACHE_FILE

def _alt_cache_file_path() -> Path:
    return Path(__file__).resolve().parent / ALT_CACHE_FILE

def _load_defs_cache() -> dict:
    cache = {}
    paths = []
    p_main = _cache_file_path()
    p_alt = _alt_cache_file_path()
    if p_main.exists():
        paths.append(p_main)
    if p_alt.exists() and p_alt != p_main:
        paths.append(p_alt)
    if not paths:
        return cache
    total_rows = 0
    try:
        import csv as _csv
        for p in paths:
            # First try headered CSV
            with open(p, "r", encoding="utf-8", newline="") as f:
                reader = _csv.DictReader(f)
                used_header = False
                if reader.fieldnames and "word" in [h.strip().lower() for h in reader.fieldnames if h]:
                    used_header = True
                    for row in reader:
                        total_rows += 1
                        w = (row.get("word") or "").strip().lower()
                        d = (row.get("definition") or "").strip()
                        is_new = (row.get("is_new") or "1").strip().lower() in ("1", "true", "yes")
                        img = (row.get("image") or "").strip()
                        song = (row.get("song") or "").strip()
                        source = (row.get("source") or "").strip()
                        created_at = (row.get("created_at") or "").strip()
                        if w:
                            cache[w] = {
                                "definition": d,
                                "is_new": is_new,
                                "image": (img or None),
                                "song": (song or None),
                                "source": (source or None),
                                "created_at": (created_at or None),
                            }
                if not used_header:
                    # Fallback: headerless CSV in column order
                    # Columns: 0=word, 1=definition, 2=is_new, 3=image, 4=song, 5=source (opt), 6=created_at (opt)
                    f.seek(0)
                    rr = _csv.reader(f)
                    for row in rr:
                        if not row:
                            continue
                        total_rows += 1
                        try:
                            w = (row[0] if len(row) > 0 else "").strip().lower()
                            d = (row[1] if len(row) > 1 else "").strip()
                            is_new_raw = (row[2] if len(row) > 2 else "1").strip().lower()
                            img = (row[3] if len(row) > 3 else "").strip()
                            song = (row[4] if len(row) > 4 else "").strip()
                            source = (row[5] if len(row) > 5 else "").strip()
                            created_at = (row[6] if len(row) > 6 else "").strip()
                            is_new = is_new_raw in ("1", "true", "yes")
                        except Exception:
                            # Skip malformed rows
                            continue
                        if w:
                            cache[w] = {
                                "definition": d,
                                "is_new": is_new,
                                "image": (img or None),
                                "song": (song or None),
                                "source": (source or None),
                                "created_at": (created_at or None),
                            }
        try:
            srcs = ", ".join(str(p) for p in paths)
            print(f"Cache loaded {len(cache)} unique words from {total_rows} rows across: {srcs}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"Cache load failed: {e}")
        except Exception:
            pass
    return cache

def _append_defs_cache(word: str, definition: str, is_new: bool, image: str | None = None, song: str | None = None, *, source: str | None = None, created_at: str | None = None) -> None:
    p = _cache_file_path()
    exists = p.exists()
    try:
        import csv as _csv
        with open(p, "a", encoding="utf-8", newline="") as f:
            fieldnames = ["word", "definition", "is_new", "image", "song", "source", "created_at"]
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "word": (word or "").strip(),
                "definition": (definition or "").strip(),
                "is_new": "1" if is_new else "0",
                "image": (image or ""),
                "song": (song or ""),
                "source": (source or ""),
                "created_at": (created_at or datetime.utcnow().isoformat(timespec='seconds') + 'Z'),
            })
    except Exception as e:
        try:
            print(f"Cache append failed: {e}")
        except Exception:
            pass

# === Optional AI-based known-word check ===
def _is_known_word_ai(word: str) -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _is_known_word(word)
    system = (
        "You are a precise English lexicographer. Determine if the given token is an existing English word "
        "in standard dictionaries (not proper nouns or product names). Return strict JSON only."
    )
    user = (
        "Word: " + (word or "") + "\n" +
        "Reply strictly as {\"known\": true|false}."
    )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=8,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        data = _load_json_safe(content, "Word-existence") or {}
        _print_usage_cost("Word-existence", getattr(resp, "usage", None))
        return bool(data.get("known") is True)
    except Exception as e:
        try:
            print(f"Word-existence ERROR: {e}")
        except Exception:
            pass
        return _is_known_word(word)
def by_place(placeNum,place):
    n = 0
    for i in place[placeNum]:
        n += int(i)
    num = random.randint(1, n)
    p = 1
    n = 0
    for i in place[placeNum]:
        n += int(i)
        if (n >= num): 
            break
        else:
            p += 1
    return place[0][p-1]

def letter_to_num(char):
    char.lower()
    if char == 'a':
        return 1
    elif char == 'b':
        return 2
    elif char == 'c':
        return 3
    elif char == 'd':
        return 4
    elif char == 'e':
        return 5
    elif char == 'f':
        return 6
    elif char == 'g':
        return 7
    elif char == 'h':
        return 8
    elif char == 'i':
        return 9
    elif char == 'j':
        return 10
    elif char == 'k':
        return 11
    elif char == 'l':
        return 12
    elif char == 'm':
        return 13
    elif char == 'n':
        return 14
    elif char == 'o':
        return 15
    elif char == 'p':
        return 16
    elif char == 'q':
        return 17
    elif char == 'r':
        return 18
    elif char == 's':
        return 19
    elif char == 't':
        return 20
    elif char == 'u':
        return 21
    elif char == 'v':
        return 22
    elif char == 'w':
        return 23
    elif char == 'x':
        return 24
    elif char == 'y':
        return 25
    elif char == 'z':
        return 26
    elif char == 'A':
        return 1
    elif char == 'B':
        return 2
    elif char == 'C':
        return 3
    elif char == 'D':
        return 4
    elif char == 'E':
        return 5
    elif char == 'F':
        return 6
    elif char == 'G':
        return 7
    elif char == 'H':
        return 8
    elif char == 'I':
        return 9
    elif char == 'J':
        return 10
    elif char == 'K':
        return 11
    elif char == 'L':
        return 12
    elif char == 'M':
        return 13
    elif char == 'N':
        return 14
    elif char == 'O':
        return 15
    elif char == 'P':
        return 16
    elif char == 'Q':
        return 17
    elif char == 'R':
        return 18
    elif char == 'S':
        return 19
    elif char == 'T':
        return 20
    elif char == 'U':
        return 21
    elif char == 'V':
        return 22
    elif char == 'W':
        return 23
    elif char == 'X':
        return 24
    elif char == 'Y':
        return 25
    elif char == 'Z':
        return 26
    else:
        return 0
    
def by_relative(placeNum,relative):
    n = 0
    for i in relative[placeNum]:
        n += int(i)
    num = random.randint(1, n)
    p = 1
    n = 0
    for i in relative[placeNum]:
        n += int(i)
        if (n >= num): 
            break
        else:
            p += 1
    return relative[0][p-1]

def generate_definition_ai(word: str) -> str:
    """Use OpenAI to create a unique, playful dictionary-style definition for a fake word."""
    # Fail fast if API key is missing
    if not os.getenv("OPENAI_API_KEY"):
        return "[Missing OPENAI_API_KEY: set it in your environment to enable AI definitions.]"

    system = (
        "If the target word is an established entry in major dictionaries, return its standard dictionary definition instead of inventing one."
        "You are a lexicographer and wordsmith tasked with inventing plausible definitions for new words. "
        "Avoid vague phrasings such as \"state of\", \"feeling of\", or \"thing that\". "
        "Make it realistic. "
        "Give a good mix of verbs, nouns, and adjectives, adverbs, and other parts of speech."
    )
    user = (
        "If the target word is an established entry in major dictionaries, return its standard dictionary definition instead of inventing one."
        f"Invent a dictionary-style plausible definition for the fake word \"{word}\". "
        "Produce 1–2 sentences and keep the whole entry under ~40 words. "
        "Append the part of speech in parentheses immediately after the word (for example (noun) or (verb)). "
        "Optionally add a tiny example after \"e.g.,\" only if it sharpens the image. "
        "Do not mention or admit the word is invented; present the entry as a natural dictionary line."
        
    )
    banned_terms = ["sock", "socks", "shoe", "shoes", "hosiery", "laundry", "drawer", "drawers","dance","squirrel","whimsical"]
    def _uses_banned(text: str) -> bool:
        t = (text or "").lower()
        return any(term in t for term in banned_terms)
    try:
        client = OpenAI()
        def _make_call(note: str | None = None):
            msgs = [{"role": "system", "content": system}]
            u = user if not note else (user + "\nNote: " + note)
            msgs.append({"role": "user", "content": u})
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs,
                temperature=0.6,
                top_p=0.9,
                max_tokens=140,
            )
        resp = _make_call()
        text = (resp.choices[0].message.content or "").strip()
        _print_usage_cost("Definition", getattr(resp, "usage", None))
        if _uses_banned(text):
            try:
                print("Anti-sock check triggered; regenerating once without clothing terms.")
            except Exception:
                pass
            resp2 = _make_call("Previous draft used banned clothing terms; regenerate without any clothing-related words.")
            text2 = (resp2.choices[0].message.content or "").strip()
            _print_usage_cost("Definition-regenerate", getattr(resp2, "usage", None))
            if not _uses_banned(text2):
                return text2
        return text
    except Exception as e:
        return f"[AI definition error: {e}]"

def _shortlist(items, k):
    """Return up to k items from the list. If more than k, sample deterministically using seed for stability."""
    if len(items) <= k:
        return items
    random.seed(42)
    return random.sample(items, k)

def _use_full_media_lists() -> bool:
    val = os.getenv("MEDIA_FULL_LIST", "1").strip().lower()
    return val in ("1", "true", "yes", "on")


def _sample_items(items: list[str], ratio: float | None = None) -> list[str]:
    """Return a random sample of items according to ratio (default 0.5).
    Always keeps at least 1 item when the list is non-empty. If ratio >= 1, returns full list.
    """
    if not items:
        return items
    if ratio is None:
        try:
            ratio = float(os.getenv("MEDIA_SAMPLE_RATIO", "1") or 0.5)
        except Exception:
            ratio = 1
    # Clamp ratio to [0, 1]
    ratio = max(0.0, min(1.0, ratio))
    k = int(round(len(items) * ratio))
    k = max(1, min(len(items), k))
    if k >= len(items):
        return list(items)
    # Use an unseeded sample so each request is fresh
    return random.sample(items, k)


def choose_media_ai(word: str, definition: str, images: list[str], songs: list[str]) -> tuple[str, str]:
    """Ask the AI to pick one image and one song filename from provided shortlists.

    Falls back to random choices if API key is missing or on error.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or 1 == 1:
        # Fallback to random if no key
        return (random.choice(images) if images else None, random.choice(songs) if songs else None)

    # Sample down to save tokens (default: half of each list). Still keeps at least one.
    # You can tune the fraction via MEDIA_SAMPLE_RATIO (0.0–1.0).
    img_short = _sample_items(images)
    song_short = _sample_items(songs)
    list_mode = "sampled"
    # Guard against extremely large prompts (> budget)
    joined = ", ".join(img_short) + ", " + ", ".join(song_short)
    char_budget = int(os.getenv("MEDIA_LIST_CHAR_BUDGET", "45000"))
    if len(joined) > char_budget:
        # Switch to tag-based local matching so we can consider ALL files without listing them
        list_mode = "tag-selection"
        try:
            print(f"Media selection mode: {list_mode} | using tags over all files (images={len(images)}, songs={len(songs)})")
        except Exception:
            pass

        def _tokens(s: str):
            import re
            return set([t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if t])

        def _score(name: str, tags):
            name_tokens = _tokens(os.path.splitext(name)[0])
            tag_tokens = set()
            for t in tags:
                tag_tokens |= _tokens(t)
            # basic overlap score
            return sum(1 for t in tag_tokens if any(t in nt or nt in t for nt in name_tokens))

        # Ask AI for compact tags
        tag_system = (
            "You are curating media for a whimsical word generator. Return compact keyword tags that describe "
            "what image and song would best fit the word and its definition. Return strict JSON only.Strictly choose what makes the most sense, don't make any stretches."
        )
        tag_user = (
            "Fake word: " + word + "\n" +
            "Definition: " + (definition or "") + "\n" +
            "Respond ONLY as JSON like {\"image_tags\":[\"tag1\",\"tag2\"],\"song_tags\":[\"tag1\"],\"rationale\":\"short reason\"}. Keep tags 1-3 words each."
        )
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": tag_system},
                    {"role": "user", "content": tag_user},
                ],
                temperature=0.6,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            data = _load_json_safe(content, "Media-tags")
            if data is None:
                raise ValueError("Invalid JSON in media-tags response")
            image_tags = data.get("image_tags") or []
            song_tags = data.get("song_tags") or []
            rationale = data.get("rationale")
            _print_usage_cost("Media-tags", getattr(resp, "usage", None))
            if rationale:
                try:
                    print(f"Media rationale (tags): {rationale}")
                except Exception:
                    pass

            # Score all files locally
            chosen_image = max(images, key=lambda n: _score(n, image_tags)) if images else None
            chosen_song = max(songs, key=lambda n: _score(n, song_tags)) if songs else None
            return chosen_image, chosen_song
        except Exception as e:
            try:
                print(f"Media-tags ERROR: {e}")
            except Exception:
                pass
            # Fall through to sampled mode if tags fail
            img_short = _sample_items(images)
            song_short = _sample_items(songs)
            list_mode = "fallback-sampled"

    # Log which mode and counts we are using
    try:
        ratio_env = os.getenv("MEDIA_SAMPLE_RATIO", "0.5")
        print(f"Media selection mode: {list_mode} | images={len(img_short)}/{len(images)} | songs={len(song_short)}/{len(songs)} | ratio={ratio_env}")
    except Exception:
        pass

    system = (
        "You are curating media for a whimsical word generator. Choose ONE image filename and ONE song filename "
        "from the provided lists that best match the word's mood and definition. Be coherent and precise. Don't make any stretches."
        "Return strict JSON only."
    )
    user = (
        "Fake word: " + word + "\n" +
        "Definition: " + (definition or "") + "\n" +
        "Images (filenames only): " + ", ".join(img_short) + "\n" +
        "Songs (filenames only): " + ", ".join(song_short) + "\n" +
        "Respond ONLY as JSON with keys image, song, and rationale (a short explanation of the choice), e.g.: "
        "{\"image\": \"<image filename>\", \"song\": \"<song filename>\", \"rationale\": \"why they fit\"}."
    )

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.6,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        data = _load_json_safe(content, "Media-choice")
        chosen_image = None
        chosen_song = None
        rationale = None
        if data is not None:
            chosen_image = data.get("image")
            chosen_song = data.get("song")
            rationale = data.get("rationale")
        else:
            # Try regex salvage from partially formed JSON
            pairs = _extract_pairs(content, ["image", "song", "rationale"])
            chosen_image = pairs.get("image")
            chosen_song = pairs.get("song")
            rationale = pairs.get("rationale")
            try:
                print("Media-choice JSON recovery: using regex-extracted fields")
            except Exception:
                pass
        # Log model usage and rationale
        _print_usage_cost("Media-choice", getattr(resp, "usage", None))
        if rationale:
            try:
                print(f"Media rationale: {rationale}")
            except Exception:
                pass
        # Validate choices against available sets
        images_set = set(images)
        songs_set = set(songs)
        if chosen_image not in images_set:
            chosen_image = img_short[0] if img_short else (images[0] if images else None)
        if chosen_song not in songs_set:
            chosen_song = song_short[0] if song_short else (songs[0] if songs else None)
        return chosen_image, chosen_song
    except Exception as e:
        try:
            print(f"Media-choice ERROR: {e}")
        except Exception:
            pass
        # On any failure, fallback to random
        return (random.choice(images) if images else None, random.choice(songs) if songs else None)

# ===== Flask Web App =====
load_dotenv()  # load variables from .env if present
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route("/", methods=["GET"])
def index():
    return render_template('app.html', song_url=None, pic_url=None, word=None, definition=None)
@app.route("/predict", methods=["POST"])
def guess():
    song_folder = os.path.join(app.static_folder, 'songs')
    songs = [f for f in os.listdir(song_folder) if f.lower().endswith(('.ogg', '.mp3', '.wav'))]
    pic_folder = os.path.join(app.static_folder, 'images')
    pics = [f for f in os.listdir(pic_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    # New: allow user-provided word
    custom = (request.form.get("yourWord", "") or "").strip()
    sLetter = request.form.get("sLetter", "")
    length = request.form.get("length", "")

    if custom:
        reconstructed = custom
    else:
        if (sLetter):
            print(sLetter)
        else:
            print("Nuh Uh")
        if length:
            print(length)
        else:
            print("Nope")
        with open(CSV_ONE, "r") as file:
            reader = csv.reader(file)
            place = [[str(cell) for cell in row] for row in reader]
        with open(CSV_TWO, "r") as file:
            reader = csv.reader(file)
            relative = [[str(cell) for cell in row] for row in reader]
            if length:
                max_len = int(length)
            else:
                max_len = random.randint(2, 10)
            char_array = []

            for x in range(max_len):
                char1 = 'a'
                char2 = 'b'
                help_flag = False
                while char1 != char2:
                    if (len(place) > x + 1):
                        char1 = by_place(x + 1, place)
                    else:
                        help_flag = True
                    if (x != 0):
                        prev_char = char_array[x - 1]
                        num = letter_to_num(prev_char)
                        print(prev_char)
                        print(num)
                        char2 = by_relative(num, relative)
                    else:
                        char2 = char1
                    if x == 0 and sLetter:
                        char1 = sLetter
                        char2 = sLetter
                    if help_flag is True:
                        char1 = char2
                char_array.append(char1)
                print(char_array)
        reconstructed = ''.join(char_array)

    # Cache lookup first
    cache = _load_defs_cache()
    cached = cache.get(reconstructed.lower())
    status_label = None
    if cached:
        definition = cached.get("definition") or ""
        # Display logic: cached entries are no longer shown as freshly minted
        was_new = bool(cached.get("is_new", False))
        created_at_cached = cached.get("created_at") or ""
        if was_new:
            # Show created date if we have it; else generic note
            date_part = created_at_cached.split('T')[0] if 'T' in created_at_cached else (created_at_cached or None)
            status_label = f" Created on {date_part}" if date_part else " Created earlier"
        else:
            status_label = " Pre-existing word"
        is_new_word = False
        chosen_image = cached.get("image")
        chosen_song = cached.get("song")
        try:
            print(f"Cache HIT for '{reconstructed}': using cached definition and media")
        except Exception:
            pass
    else:
        definition = generate_definition_ai(reconstructed)
        use_ai_for_known = (os.getenv("WORD_EXISTENCE_VIA_AI", "1").strip().lower() in ("1","true","yes","on"))
        if use_ai_for_known:
            is_new_word = not _is_known_word_ai(reconstructed)
        else:
            is_new_word = not _is_known_word(reconstructed)
        # Choose media once and cache them for repeatability
        chosen_image, chosen_song = choose_media_ai(reconstructed, definition, pics, songs)
        source_label = "user" if custom else "generator"
        # Persist a created_at we can also show on the page
        created_now = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        _append_defs_cache(reconstructed, definition, is_new_word, chosen_image, chosen_song, source=source_label, created_at=created_now)
        status_label = " Freshly minted" if is_new_word else " Pre-existing word"
        try:
            print(f"Cache MISS for '{reconstructed}': generated and cached definition + media")
        except Exception:
            pass

    # If cache hit provided media, reuse; otherwise choose now (covers legacy cache rows)
    if not cached:
        # already chosen above
        pass
    else:
        if not chosen_image or not chosen_song:
            ci, cs = choose_media_ai(reconstructed, definition, pics, songs)
            chosen_image = chosen_image or ci
            chosen_song = chosen_song or cs
            # Update cache row append-only: write a new row with media populated and is_new set to 0
            _append_defs_cache(reconstructed, definition, False, chosen_image, chosen_song, source=("user" if custom else "generator"))

    pic_url = url_for('static', filename=f'images/{chosen_image}') if chosen_image else None
    song_url = url_for('static', filename=f'songs/{chosen_song}') if chosen_song else None

    return render_template("app.html", word=reconstructed, song_url=song_url, pic_url=pic_url, definition=definition, is_new_word=is_new_word, status_label=status_label)






if __name__ == '__main__':
    # Run on localhost by default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=False)
