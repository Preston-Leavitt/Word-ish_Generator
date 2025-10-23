import random
import csv
import os
from flask import Flask, render_template, url_for, request
from openai import OpenAI
from dotenv import load_dotenv
CSV_ONE = "place.csv"
CSV_TWO = "relative.csv"
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
        "You are a playful lexicographer inventing novel but plausible definitions for made-up words. "
        "Write in a fresh, vivid style suitable for all audiences. Avoid generic clichés like 'state of', 'feeling of', or 'thing that'. "
        "Prefer surprising imagery, compact wit, and specificity."
        "Make the definition funny too, if possible."
    )
    user = (
        f"Invent a distinctive dictionary-style definition for the fake word '{word}'. "
        "Constraints: 1–2 sentences, under ~40 words total; include part of speech in parentheses after the word (e.g., (noun) or (verb)); "
        "optionally add a tiny example after 'e.g.,' if it amplifies the vibe. Do not mention that the word is made-up."
    )
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.9,
            max_tokens=120,
        )
        # Print token usage and estimated cost if pricing env vars are set
        _print_usage_cost("Definition", getattr(resp, "usage", None))
        return (resp.choices[0].message.content or "").strip()
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


def choose_media_ai(word: str, definition: str, images: list[str], songs: list[str]) -> tuple[str, str]:
    """Ask the AI to pick one image and one song filename from provided shortlists.

    Falls back to random choices if API key is missing or on error.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to random if no key
        return (random.choice(images) if images else None, random.choice(songs) if songs else None)

    # Use full lists by default, with a size guard; otherwise try tag-based selection or fall back to shortlists
    allow_full = _use_full_media_lists()
    img_short = images if allow_full else _shortlist(images, 30)
    song_short = songs if allow_full else _shortlist(songs, 15)
    list_mode = "full" if allow_full else "shortlist"
    # Guard against extremely large prompts (> ~12k chars)
    joined = ", ".join(img_short) + ", " + ", ".join(song_short)
    char_budget = int(os.getenv("MEDIA_LIST_CHAR_BUDGET", "45000"))
    if len(joined) > char_budget and allow_full:
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
            "what image and song would best fit the word and its definition. Return strict JSON only."
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
            # Fall through to shortlist mode if tags fail
            img_short = _shortlist(images, 30)
            song_short = _shortlist(songs, 15)
            list_mode = "fallback-shortlist"

    # Log which mode and counts we are using
    try:
        print(f"Media selection mode: {list_mode} | images={len(img_short)}/{len(images)} | songs={len(song_short)}/{len(songs)}")
    except Exception:
        pass

    system = (
        "You are curating media for a whimsical word generator. Choose ONE image filename and ONE song filename "
        "from the provided lists that best match the word's mood and definition. Be bold and fun, but coherent. "
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
    sLetter = request.form.get("sLetter", "")
    length = request.form.get("length", "")
    if (sLetter):
        
        print(sLetter)
    else:
        print("Nuh Uh")
    if length:
        print (length)
    else:
        print("Nope")
    with open(CSV_ONE, "r") as file:
        reader = csv.reader(file)
        place = [[str(cell) for cell in row] for row in reader]
    with open(CSV_TWO, "r") as file:
        reader = csv.reader(file)
        relative = [[str(cell) for cell in row] for row in reader]
        if length:
            max = int(length)
        else:
            max = random.randint(2,10)
        char_array = []
       
        for x in range(max):
            char1 = 'a'
            char2 = 'b'
            help = False
            while char1 != char2:
                if(len(place) > x+1):
                    char1 = by_place(x+1,place)
                else:
                    help = True
                if(x!=0):
                    char = char_array[x-1]
                    num = letter_to_num(char)
                    print(char)
                    print(num)
                    char2 = by_relative(num,relative)
                else:
                    char2 = char1
                if x==0 and sLetter:
                    char1 = sLetter
                    char2 = sLetter
                if help == True:
                    char1 = char2
            char_array.append(char1)
            print(char_array)
    reconstructed = ''.join(char_array)
    definition = generate_definition_ai(reconstructed)

    # Ask AI to choose media from available files (fallback to random if AI is unavailable)
    chosen_image, chosen_song = choose_media_ai(reconstructed, definition, pics, songs)

    pic_url = url_for('static', filename=f'images/{chosen_image}') if chosen_image else None
    song_url = url_for('static', filename=f'songs/{chosen_song}') if chosen_song else None

    return render_template("app.html", word=reconstructed, song_url=song_url, pic_url=pic_url, definition=definition)






if __name__ == '__main__':
    # Run on localhost by default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=False)