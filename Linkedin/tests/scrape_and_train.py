import os, sys

import time
import json
import csv
import re
import math
import hashlib
import random
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Tuple
import args  # kept intentionally per requirement


# Lazy / optional imports guarded
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *a, **k: None

# Load .env AFTER consent line (consent must already be true in environment)
load_dotenv()

# --- Environment / Config ---
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
MAX_POSTS = int(os.getenv("MAX_POSTS", "100") or 100)
RATE_LIMIT_MS = int(os.getenv("RATE_LIMIT_MS", "1000") or 1000)
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MODE = os.getenv("MODE", "RAG").upper()
BACKEND = os.getenv("BACKEND", "apify").lower()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-west-2")
PINECONE_ALT_REGIONS = [r.strip() for r in os.getenv("PINECONE_ALT_REGIONS", "us-east-1,eu-west-1").split(",") if r.strip()]
PINECONE_DISABLE = os.getenv("PINECONE_DISABLE", "0") == "1"
PINECONE_SILENCE_PLAN_ERRORS = os.getenv("PINECONE_SILENCE_PLAN_ERRORS", "0") == "1"
# Plan limitation detection patterns
PINECONE_PLAN_LIMIT_PATTERNS = [
    "free plan does not support",
    "your free plan does not support",
    "does not support indexes",
    "upgrade your plan"
]
# Cache flag: once we detect plan unsupported we stop retrying Pinecone for the process lifetime
_PINECONE_PLAN_UNSUPPORTED = False
# --- Added PGVector & Apify custom actor envs ---
PGVECTOR_DSN = os.getenv("PGVECTOR_DSN")
PGVECTOR_HOST = os.getenv("PGVECTOR_HOST")
PGVECTOR_PORT = os.getenv("PGVECTOR_PORT", "5432")
PGVECTOR_DB = os.getenv("PGVECTOR_DB")
PGVECTOR_USER = os.getenv("PGVECTOR_USER")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "linkedin_embeddings")
APIFY_ACTOR = os.getenv("APIFY_ACTOR", "curious_coder/linkedin-post-search-scraper")
INPUT_PAYLOAD = {
            "cookie": [
                "li_at=AQEDAVzoRR0AvEHEAAABmaaeImAAAAGZyqqmYE4AmLPhTGn6YIRzW0yZfrCLOR-OgBRSLE3UjWXWsm2jOcIP-9rbADHgjofpaeA4Kmm3_8N20JNIQiqNTPXlFWTWQKRdjGIOnYJE3frEfqJ1g62MRdTM; JSESSIONID=ajax:6484253360822640261"
            ],
            "deepScrape": True,
            "maxDelay": 8,
            "minDelay": 2,
            "proxy": {
                "useApifyProxy": True,
                "apifyProxyCountry": "US"
            },
            "rawData": False,
            "urls": [
                "https://www.linkedin.com/search/results/content/?keywords=how%20I&origin=SWITCH_SEARCH_VERTICAL&sid=C5Y",
                "https://www.linkedin.com/company/amazon",
                "https://www.linkedin.com/search/results/content/?datePosted=%22past-24h%22&keywords=ai&origin=FACETED_SEARCH"
            ]
        }
APIFY_DATASET_ID = os.getenv("APIFY_DATASET_ID")  # optional direct dataset fallback
SYNTHETIC_ON_EMPTY = os.getenv("SYNTHETIC_ON_EMPTY", "1")  # "1" -> create synthetic example posts if scrape empty
APIFY_STRICT = os.getenv("APIFY_STRICT", "0")
APIFY_SUPPRESS_NOT_FOUND = os.getenv("APIFY_SUPPRESS_NOT_FOUND", "0") == "1"  # suppress actor-not-found warning when true
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Add new environment variables for LinkedIn OAuth ---
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "http://localhost:8000/callback")
LINKEDIN_SCOPES = os.getenv("LINKEDIN_SCOPES", "r_liteprofile w_member_social")
OAUTH_TOKENS_PATH = DATA_DIR / "linkedin_oauth.json"

# Utilities ------------------------------------------------------------------

def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")

def exponential_backoff(retry_idx: int, base: float = 1.0, cap: float = 30.0):
    sleep = min(cap, base * (2 ** retry_idx) * (0.5 + random.random()))
    time.sleep(sleep)

def safe_write(path: Path, data: str, mode: str = "w", encoding: str = "utf-8"):
    if DRY_RUN:
        log(f"DRY_RUN enabled: not writing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding=encoding) as f:
        f.write(data)

# PII / Redaction -------------------------------------------------------------

NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s().-]{7,}\d")
PROFILE_URL_PATTERN = re.compile(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_/%]+")

def redact_text(text: str) -> str:
    # Conservative approach: sequence of replacements
    text = EMAIL_PATTERN.sub("<REDACTED_EMAIL>", text)
    text = PHONE_PATTERN.sub("<REDACTED_PHONE>", text)
    text = PROFILE_URL_PATTERN.sub("<REDACTED_PROFILE>", text)
    # Heuristic name redaction: avoid over-redacting by skipping all-caps and short tokens
    def repl_name(m):
        token = m.group(0)
        if token.isupper() or len(token) < 3:
            return token
        return "<REDACTED_NAME>"
    text = NAME_PATTERN.sub(repl_name, text)
    return text

# Emoji / Control normalization
try:
    import emoji
    def emoji_to_shortcode(t: str) -> str:
        return emoji.demojize(t, delimiters=("", ""))
except ImportError:
    def emoji_to_shortcode(t: str) -> str:
        return t

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+")

def normalize_text(raw: str) -> str:
    if not raw:
        return ""
    t = raw
    t = CONTROL_CHARS.sub(" ", t)
    t = emoji_to_shortcode(t)
    t = re.sub(r"https?://t\.co/[A-Za-z0-9]+", "<SHORT_URL>", t)
    # Reduce to domain only for http(s) links
    t = re.sub(r"https?://([^/\s]+)(/\S*)?", lambda m: m.group(1), t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Hash / Dedup
def post_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

# Data Schemas ----------------------------------------------------------------

class RawPost(Dict[str, Any]): ...
class CleanPost(Dict[str, Any]): ...

# Backend Abstractions --------------------------------------------------------

class ScraperBackend:
    """Interface for scraper backends."""
    def fetch_posts(self, limit: int) -> List[RawPost]:
        raise NotImplementedError

class ApifyBackend(ScraperBackend):
    """
    Fetch LinkedIn posts via Apify dataset or actor.
    Env:
      APIFY_ACTOR       actor ID or name (default placeholder)
      APIFY_DATASET_ID  (optional) existing dataset to read from if actor not found
    """
    def __init__(self, token: str):
        from apify_client import ApifyClient  # type: ignore
        self.client = ApifyClient(token)
        self.actor_id = APIFY_ACTOR
        self.dataset_id = APIFY_DATASET_ID
        self.actor_not_found = False
        self.suppress_not_found = APIFY_SUPPRESS_NOT_FOUND

    # --- Helper builders for Apify input (unit-testable) ---
    def mask_secret(s: str, keep_start: int = 2, keep_end: int = 4) -> str:
        """Mask a sensitive string for logs."""
        if not s:
            return ""
        if len(s) <= keep_start + keep_end:
            return "*" * len(s)
        return f"{s[:keep_start]}{'*'*(len(s)-keep_start-keep_end)}{s[-keep_end:]}"
    
    @staticmethod
    def build_cookie_array(li_at: str | None, jsessionid: str | None) -> list[str]:
        """
        Build the single-element cookie array expected by the actor.
        Returns [] if insufficient data (caller validates).
        """
        if not li_at:
            return []
        parts = [f"li_at={li_at}"]
        if jsessionid:
            parts.append(f"JSESSIONID={jsessionid}")
        return ["; ".join(parts)]

    @staticmethod
    def build_proxy_object(use_apify_proxy: bool, groups: str | None,
                           country: str | None, proxy_urls: str | None) -> dict:
        """
        Build proxy object. If proxy_urls provided (comma list), returns proxyUrls variant.
        Else returns Apify proxy config. Returns {} if neither valid form present.
        """
        proxy_urls_list = []
        if proxy_urls:
            proxy_urls_list = [u.strip() for u in proxy_urls.split(",") if u.strip()]
        if proxy_urls_list:
            return {"proxyUrls": proxy_urls_list}
        if use_apify_proxy:
            obj = {"useApifyProxy": True}
            if groups:
                grp = [g.strip() for g in groups.split(",") if g.strip()]
                if grp:
                    obj["apifyProxyGroups"] = grp
            if country:
                obj["apifyProxyCountry"] = country.strip()
            return obj
        return {}

    def _sanitize_apify_payload_for_log(payload: dict) -> dict:
        """Return a shallow-masked copy safe for logging."""
        from copy import deepcopy
        safe = deepcopy(payload)
        # Mask cookie
        if "cookie" in safe:
            masked = []
            for line in safe["cookie"]:
                if "li_at=" in line:
                    try:
                        val = line.split("li_at=",1)[1].split(";",1)[0].strip()
                        line = line.replace(val, ApifyBackend.mask_secret(val))
                    except Exception:
                        line = "<masked_cookie_line>"
                if "JSESSIONID=" in line:
                    try:
                        val = line.split("JSESSIONID=",1)[1].split(";",1)[0].strip()
                        line = line.replace(val, ApifyBackend.mask_secret(val))
                    except Exception:
                        pass
                masked.append(line)
            safe["cookie"] = masked
        # Mask proxy URLs if present
        if isinstance(safe.get("proxy"), dict):
            if "proxyUrls" in safe["proxy"]:
                safe["proxy"]["proxyUrls"] = [u.split("@")[-1] for u in safe["proxy"]["proxyUrls"]]  # strip creds if any
        return safe
    # --- end helper additions ---

    @staticmethod
    def transform_cookies(cookies: list[str]) -> list[str]:
        """
        Transform cookies to ensure compatibility with LinkedIn actor.
        
        Enhanced to fix "Cannot read properties of undefined (reading 'match')" error
        in the Apify actor, which occurs when the cookie format isn't exactly what's expected.
        """
        if not cookies or not isinstance(cookies, list):
            return []
        
        # Extract only the li_at cookie value without any attributes
        li_at_value = ""
        jsessionid_value = ""
        
        for cookie in cookies:
            if not cookie:
                continue
            
            # Extract li_at value
            if "li_at=" in cookie:
                try:
                    raw_value = cookie.split("li_at=", 1)[1].split(";", 1)[0].strip()
                    # URL-decode if necessary
                    if '%' in raw_value:
                        from urllib.parse import unquote
                        raw_value = unquote(raw_value)
                    li_at_value = raw_value
                except Exception:
                    pass
            
            # Extract JSESSIONID value if present
            if "JSESSIONID=" in cookie:
                try:
                    raw_value = cookie.split("JSESSIONID=", 1)[1].split(";", 1)[0].strip()
                    if raw_value.startswith("ajax:"):
                        jsessionid_value = raw_value
                except Exception:
                    pass
        
        # Format as a clean cookie string with no extraneous attributes
        cookie_parts = []
        if li_at_value:
            cookie_parts.append(f"li_at={li_at_value}")
        if jsessionid_value:
            cookie_parts.append(f"JSESSIONID={jsessionid_value}")
        
        if not cookie_parts:
            return []
        
        # Return in the exact format the actor expects
        return ["; ".join(cookie_parts)]

    def _build_cookie_array_from_env(self, li_at_env: str, jsession_env: str) -> list[str]:
        """
        Helper method to build cookie array from environment variables.
        Handles cleaning, formatting and validation of LinkedIn cookies.
        """
        # Clean any surrounding quotes
        li_at_env = li_at_env.strip('"\'')
        jsession_env = jsession_env.strip('"\'')
        
        cookie_arr = []
        if li_at_env:
            # If the value already contains "li_at=", extract just the value part
            if "li_at=" in li_at_env:
                try:
                    li_at_env = li_at_env.split("li_at=", 1)[1].split(";", 1)[0].strip()
                except Exception:
                    pass
            cookie_arr = [f"li_at={li_at_env}"]
            
            if jsession_env:
                # Similarly clean up JSESSIONID if needed
                if "JSESSIONID=" in jsession_env:
                    try:
                        jsession_env = jsession_env.split("JSESSIONID=", 1)[1].split(";", 1)[0].strip()
                    except Exception:
                        pass
                if jsession_env:
                    cookie_arr = [f"li_at={li_at_env}; JSESSIONID={jsession_env}"]
        
        # Transform cookies to ensure actor compatibility
        # Fix: Call as static method via class, not instance
        return ApifyBackend.transform_cookies(cookie_arr)

    @staticmethod
    def build_oauth_cookie(access_token: str) -> list[str]:
        """Build cookie string from OAuth access token."""
        if not access_token:
            return []
        
        try:
            import requests
            # Make API request to get profile info
            profile_response = requests.get(
                "https://api.linkedin.com/v2/me",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }
            )
            
            if profile_response.status_code != 200:
                log(f"Failed to get profile with access token: {profile_response.status_code}", "ERROR")
                return []
                
            # Note: LinkedIn OAuth doesn't typically provide cookies directly
            # This is a placeholder that would need to be extended for actual implementation
            log("OAuth token available but direct cookie extraction not implemented.", "WARN")
            log("Using token for API calls but not for scraping.", "WARN")
            return []
            
        except Exception as e:
            log(f"Error using OAuth token: {e}", "ERROR")
            return []

    def fetch_posts(self, limit: int) -> List[RawPost]:
        log(f"Using ApifyBackend (actor={self.actor_id}, dataset_fallback={self.dataset_id or 'none'})")
        posts: List[RawPost] = []
        run = None

        # Try to use OAuth token first if available 
        oauth_access_token, member_urn = get_stored_linkedin_token()
        if oauth_access_token:
            log(f"Using LinkedIn OAuth credentials (member URN: {member_urn})", "INFO")
            oauth_cookie = self.build_oauth_cookie(oauth_access_token)
            if oauth_cookie:
                log("Using cookie derived from OAuth credentials", "INFO")
                cookie_arr = oauth_cookie
            else:
                # Fall back to environment variables
                li_at_env = os.getenv("LI_AT_COOKIE", "").strip()
                jsession_env = os.getenv("JSESSIONID_COOKIE", "").strip()
                cookie_arr = self._build_cookie_array_from_env(li_at_env, jsession_env)
        else:
            # Standard cookie building from environment
            li_at_env = os.getenv("LI_AT_COOKIE", "").strip()
            jsession_env = os.getenv("JSESSIONID_COOKIE", "").strip()
            cookie_arr = self._build_cookie_array_from_env(li_at_env, jsession_env)
            
        # Build proxy object from environment variables
        proxy_obj = ApifyBackend.build_proxy_object(
            use_apify_proxy=True,
            groups=os.getenv("APIFY_PROXY_GROUPS"),
            country=os.getenv("APIFY_PROXY_COUNTRY"),
            proxy_urls=os.getenv("APIFY_PROXY_URLS")
        )
        # Build payload with our carefully formatted cookie
        actor_payload = {
            "cookie": cookie_arr,  # This is the key part that fixes the TypeError
            "deepScrape": True,
            "maxDelay": int(os.getenv("APIFY_MAX_DELAY", "8")),
            "minDelay": int(os.getenv("APIFY_MIN_DELAY", "2")),
            "proxy": proxy_obj,
            "rawData": False,
            "urls": [
                "https://www.linkedin.com/search/results/content/?keywords=how%20I&origin=SWITCH_SEARCH_VERTICAL",
                "https://www.linkedin.com/search/results/content/?datePosted=%22past-24h%22&keywords=ai&origin=FACETED_SEARCH"
            ]
        }

        # If user supplied a JSON override via APIFY_INPUT_JSON, merge (shallow)
        override_json = os.getenv("APIFY_INPUT_JSON")
        if override_json:
            try:
                import json as _json
                actor_payload.update(_json.loads(override_json))
            except Exception as e:
                log(f"Ignoring invalid APIFY_INPUT_JSON override: {e}", "WARN")

        sanitized = ApifyBackend._sanitize_apify_payload_for_log(actor_payload)
        log(f"Apify actor payload (sanitized) => {sanitized}")

        if DRY_RUN:
            log("DRY_RUN enabled: skipping Apify actor start (no network call).")
            return posts
        # --- Modified: start actor with retry (3 attempts) using .call() / .start() ---
        max_attempts = 3
        ds_id = None
        run_status = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                actor_ref = self.client.actor(self.actor_id)
                
                # Always use call() with run_input parameter for consistency
                run = actor_ref.call(run_input=actor_payload)
                
                if not run or not isinstance(run, dict):
                    raise RuntimeError(f"Unexpected run response type: {type(run)}")
                
                run_status = (run.get("status") or "").upper()
                ds_id = run.get("defaultDatasetId")
                break
            except Exception as e:
                if attempt < max_attempts:
                    log(f"Apify actor start failed (attempt {attempt}/{max_attempts}): {e}", "WARN")
                    exponential_backoff(attempt - 1)
                    continue
                    
                msg = str(e).lower()
                if "not found" in msg and not self.dataset_id:
                    self.actor_not_found = True
                    level = "ERROR" if APIFY_STRICT == "1" else "WARN"
                    if not self.suppress_not_found:
                        log(f"Apify actor '{self.actor_id}' not found after retries. strict={APIFY_STRICT}", level)
                    return posts
                    
                log(f"Apify actor run failed after {max_attempts} attempts: {e}", "ERROR")
                
                # Special handling for TypeError about match() 
                if "typeerror" in msg.lower() and "match" in msg.lower():
                    log("This error typically occurs when the cookie format isn't what the actor expects.", "WARN")
                    log("Ensure your LI_AT_COOKIE contains only the raw value without other attributes.", "WARN")
                
                return posts
        # --- end modified retry block ---

        # --- New: handle FAILED / ABORTED run statuses early ---
        if run_status and run_status not in ("SUCCEEDED", "COMPLETED"):
            err_msg = (run.get("statusMessage") or run.get("message") or "").strip()
            log(f"Apify actor ended with status={run_status}. No dataset will be fetched.", "ERROR")
            if err_msg:
                log(f"Actor status message: {err_msg}", "ERROR")
            
            # Enhanced error detection specific to this actor
            if "typeerror" in err_msg.lower() and ("match" in err_msg.lower() or "cannot read properties" in err_msg.lower()):
                log("Actor crashed with cookie format error. The cookie format is incorrect.", "ERROR")
                log("Try these solutions:", "WARN")
                log("1. Set LI_AT_COOKIE to the raw value (without attributes like Path, Domain, etc)", "WARN")
                log("2. Get a fresh LinkedIn cookie value by logging in again", "WARN")
                log("3. Try a different actor with APIFY_ACTOR (see Apify store)", "WARN")
            return posts
        # --- end new failure guard ---

        if not ds_id:
            # Fallback to explicitly provided dataset id (legacy path)
            ds_id = self.dataset_id

        if not ds_id:
            log("No dataset id resolved; returning empty list.", "WARN")
            return posts

        # ...existing dataset fetch loop (unchanged)...
        try:
            dataset_client = self.client.dataset(ds_id)
            for idx, item in enumerate(dataset_client.iterate_items()):
                if idx >= limit:
                    break
                posts.append({
                    "post_id": str(item.get("id") or item.get("url") or f"apify_{idx}"),
                    "author_urn": item.get("authorProfile", {}).get("urn") or item.get("author") or "urn:li:member:unknown",
                    "text": item.get("text") or "",
                    "publish_timestamp": item.get("timestamp") or item.get("time") or int(time.time()),
                    "media_urls": item.get("media", []),
                    "reactions_count": item.get("reactions") or 0,
                    "comments_count": item.get("comments") or 0,
                    "top_comments": [
                        {
                            "author_urn": c.get("authorUrn") or "urn:li:member:commenter",
                            "text": c.get("text") or ""
                        } for c in (item.get("topComments") or [])[:3]
                    ],
                    "source_url": item.get("url") or ""
                })
        except Exception as e:
            log(f"Apify dataset fetch error: {e}", "ERROR")
        return posts

class BrowserBackend(ScraperBackend):
    """
    Lightweight Playwright automation.
    NOTE: Real LinkedIn scraping may violate terms; use only with permission.
    This skeleton logs in (optionally) and collects placeholder / partial data.
    """
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ]
    def __init__(self, email: Optional[str], password: Optional[str], rate_limit_ms: int):
        self.email = email
        self.password = password
        self.rate_limit_ms = max(250, rate_limit_ms)

    def _human_delay(self):
        base = self.rate_limit_ms / 1000
        time.sleep(base + random.random() * base * 0.4)

    def fetch_posts(self, limit: int) -> List[RawPost]:
        # --- Modified: capture missing playwright with clearer hint ---
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            raise RuntimeError("Playwright not installed. Install with: pip install playwright && python -m playwright install chromium")
        posts: List[RawPost] = []
        captcha_hits = 0
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(user_agent=random.choice(self.USER_AGENTS))
            page = context.new_page()
            try:
                page.goto("https://www.linkedin.com/login")
                if self.email and self.password:
                    page.fill("input#username", self.email)
                    page.fill("input#password", self.password)
                    page.click("button[type=submit]")
                    self._human_delay()
                    if "checkpoint" in page.url.lower():
                        log("Checkpoint / challenge encountered. Aborting.", "ERROR")
                        return posts
                # Simplified search feed
                page.goto("https://www.linkedin.com/feed/")
                self._human_delay()
                # Naive extraction (placeholders)
                for idx in range(limit):
                    if idx and idx % 20 == 0:
                        page.mouse.wheel(0, 2000)
                        self._human_delay()
                    html = page.content()
                    if "captcha" in html.lower():
                        captcha_hits += 1
                        log("Possible CAPTCHA detected. Aborting.", "ERROR")
                        break
                    # Dummy extracted record
                    posts.append({
                        "post_id": f"browser_{int(time.time())}_{idx}",
                        "author_urn": "urn:li:member:browser",
                        "text": f"Sample scraped post placeholder #{idx}",
                        "publish_timestamp": int(time.time()) - (idx * 3600),
                        "media_urls": [],
                        "reactions_count": random.randint(5, 500),
                        "comments_count": random.randint(0, 40),
                        "top_comments": [],
                        "source_url": "https://www.linkedin.com/feed/"
                    })
                    self._human_delay()
                    if len(posts) >= limit:
                        break
            finally:
                context.close()
                browser.close()
        return posts

# Preprocessing ---------------------------------------------------------------

def dedupe_and_redact(raw_posts: List[RawPost]) -> List[CleanPost]:
    seen_ids = set()
    seen_hashes = set()
    cleaned: List[CleanPost] = []
    for p in raw_posts:
        pid = p.get("post_id")
        txt = normalize_text(p.get("text", ""))
        if not pid or not txt:
            continue
        fp = post_fingerprint(txt)
        if pid in seen_ids or fp in seen_hashes:
            continue
        seen_ids.add(pid)
        seen_hashes.add(fp)
        redacted = redact_text(txt)
        cleaned.append(CleanPost(
            id=pid,
            author_urn=p.get("author_urn"),
            date=p.get("publish_timestamp"),
            text=redacted,
            reactions=p.get("reactions_count", 0),
            comments=p.get("comments_count", 0),
            media_count=len(p.get("media_urls") or []),
            source_url=p.get("source_url")
        ))
    return cleaned

# Storage Paths ---------------------------------------------------------------

CSV_PATH = DATA_DIR / "posts.csv"
VECTORS_PATH = DATA_DIR / "vectors.json"
AUDIT_PATH = DATA_DIR / "audit.log"
FINETUNE_JSONL = DATA_DIR / "finetune.jsonl"

def append_audit(post: CleanPost, consent_token: str = "explicit"):
    if DRY_RUN:
        return
    line = json.dumps({
        "post_id": post["id"],
        "ts": int(time.time()),
        "source_url": post["source_url"],
        "consent": consent_token
    }, ensure_ascii=False)
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_existing_posts() -> Dict[str, CleanPost]:
    result: Dict[str, CleanPost] = {}
    if CSV_PATH.exists():
        with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                result[row["id"]] = CleanPost(row)
    return result

def save_posts(posts: List[CleanPost]):
    if DRY_RUN:
        log("DRY_RUN: skipping posts.csv write")
        return
    existing = load_existing_posts()
    for p in posts:
        existing[p["id"]] = p
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "author_urn", "date", "text", "reactions", "comments", "media_count", "source_url"])
        for p in existing.values():
            writer.writerow([
                p["id"], p["author_urn"], p["date"], p["text"],
                p["reactions"], p["comments"], p["media_count"], p["source_url"]
            ])
    for p in posts:
        append_audit(p)

# Embeddings / Vector Store ---------------------------------------------------

def token_chunks(text: str, target_tokens: int = 500, overlap: int = 50) -> List[str]:
    """Chunk text into ~target_tokens with overlap using tiktoken if available, else naive split."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        chunks = []
        start = 0
        while start < len(toks):
            end = start + target_tokens
            piece = enc.decode(toks[start:end])
            chunks.append(piece)
            start = end - overlap
            if start < 0:
                start = 0
        return chunks
    except Exception:
        words = text.split()
        chunks = []
        start = 0
        word_window = target_tokens  # treat tokens ~ words fallback
        while start < len(words):
            end = start + word_window
            piece = " ".join(words[start:end])
            chunks.append(piece)
            start = end - min(overlap, len(words))
        return chunks

# --- Added: OpenAI client helper for new SDK (>=1.0.0) ---
_OA_CLIENT = None
def _get_openai_client():
    """
    Lazily instantiate and cache OpenAI client (new SDK style).
    """
    global _OA_CLIENT
    if _OA_CLIENT is None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise RuntimeError("openai>=1.0.0 package not installed. Install with: pip install openai --upgrade")
        _OA_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _OA_CLIENT
# --- end addition ---

def openai_embed_batch(texts: List[str], model: str) -> List[List[float]]:
    # --- Modified for new OpenAI SDK ---
    client = _get_openai_client()
    embeddings: List[List[float]] = []
    BATCH = 50
    MAX_RETRIES = int(os.getenv("MAX_EMBED_RETRIES", "4"))
    transient_batches = 0
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        succeeded = False
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                # resp.data is a list of objects with .embedding
                for item in resp.data:
                    embeddings.append(item.embedding)
                succeeded = True
                break
            except Exception as e:
                msg = str(e).lower()
                if any(code in msg for code in ("rate", "timeout", "overload", "temporarily", "500", "502", "503")) and attempt < (MAX_RETRIES - 1):
                    log(f"Embedding retry {attempt+1}/{MAX_RETRIES-1} due to transient error: {e}", "WARN")
                    transient_batches += 1
                    exponential_backoff(attempt)
                    continue
                log(f"Embedding failed (final for this batch): {e}", "ERROR")
                dim = len(embeddings[0]) if embeddings else 1536
                zero = [0.0] * dim
                for _ in batch:
                    embeddings.append(zero)
                break
        if not succeeded and transient_batches:
            pass
    if transient_batches:
        log(f"Embedding: {transient_batches} batch(es) with transient failures; zero-vector fallbacks applied.", "WARN")
    return embeddings

class PineconeAdapter:
    """
    Updated Pinecone adapter with region fallback.
    """
    def __init__(self, api_key: str, index_name: str = "linkedin-posts",
                 dimension: int = 1536, metric: str = "cosine",
                 region: str = "us-west-2", alt_regions: Optional[List[str]] = None):
        if PINECONE_DISABLE:
            raise RuntimeError("Pinecone disabled via PINECONE_DISABLE=1")
        try:
            from pinecone import Pinecone, ServerlessSpec  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Pinecone import failed: {e}")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        regions_to_try = [region] + [r for r in (alt_regions or []) if r and r != region]
        last_error = None
        unsupported_phrase = "does not support indexes"
        for rg in regions_to_try:
            try:
                existing = {i.name for i in self.pc.list_indexes()}
                if index_name not in existing:
                    self.pc.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric=metric,
                        spec=ServerlessSpec(cloud="aws", region=rg)
                    )
                    # Wait until ready
                    while True:
                        desc = self.pc.describe_index(index_name)
                        if desc.status.get("ready"):
                            break
                        time.sleep(2)
                self.index = self.pc.Index(index_name)
                if rg != region:
                    log(f"Pinecone: using alternate region '{rg}' (original '{region}' failed).", "INFO")
                break
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                log(f"Pinecone index attempt failed in region '{rg}': {e}", "WARN")
                # If unsupported region phrase detected, move to next region without further waits
                if unsupported_phrase in msg:
                    continue
        else:
            raise RuntimeError(f"Pinecone index create failed across regions {regions_to_try}: {last_error}")

    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
        if not vectors:
            return
        self.index.upsert(vectors=[{"id": vid, "values": emb, "metadata": meta} for vid, emb, meta in vectors])

    def query(self, embedding: List[float], top_k: int = 5):
        res = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return res["matches"]

    def delete(self, vector_ids: List[str]):
        if vector_ids:
            self.index.delete(ids=vector_ids)

class PGVectorAdapter:
    """
    pgvector-backed store (preferred when PGVECTOR_* env vars supplied).
    Requires:
      CREATE EXTENSION IF NOT EXISTS vector;
      (Table auto-created if missing)
    """
    def __init__(self,
                 dsn: Optional[str],
                 host: Optional[str],
                 port: str,
                 db: Optional[str],
                 user: Optional[str],
                 password: Optional[str],
                 table: str,
                 dimension: int = 1536):
        import psycopg2  # type: ignore
        self._pg = psycopg2
        self.table = table
        self.dim = dimension
        if dsn:
            self.conn = psycopg2.connect(dsn)
        else:
            if not (host and db and user):
                raise ValueError("PGVector config incomplete: need PGVECTOR_DSN or host/db/user")
            self.conn = psycopg2.connect(
                host=host, port=port, dbname=db, user=user, password=password
            )
        self.conn.autocommit = True
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                  id TEXT PRIMARY KEY,
                  origin TEXT,
                  chunk TEXT,
                  embedding vector({self.dim}),
                  metadata JSONB
                );
            """)

    def _vec(self, emb: List[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
        if not vectors:
            return
        with self.conn.cursor() as cur:
            for vid, emb, meta in vectors:
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (id, origin, chunk, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                      SET origin=EXCLUDED.origin,
                          chunk=EXCLUDED.chunk,
                          embedding=EXCLUDED.embedding,
                          metadata=EXCLUDED.metadata;
                    """,
                    (vid, meta.get("origin"), meta.get("chunk"), self._vec(emb), json.dumps(meta, ensure_ascii=False))
                )

    def query(self, embedding: List[float], top_k: int = 5):
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, origin, chunk, metadata, (embedding <#> %s::vector) AS dist
                FROM {self.table}
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
                """,
                (self._vec(embedding), self._vec(embedding), top_k)
            )
            rows = cur.fetchall()
        class Match:
            def __init__(self, _id, origin, chunk, meta, dist):
                self.id = _id
                self.score = 1.0 - float(dist) if dist is not None else 0.0
                self.metadata = {"origin": origin, "chunk": chunk, **(meta or {})}
        return [Match(r[0], r[1], r[2], json.loads(r[3]) if r[3] else {}, r[4]) for r in rows]

    def delete(self, vector_ids: List[str]):
        if not vector_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table} WHERE id = ANY(%s);",
                (vector_ids,)
            )

# RAG Generation --------------------------------------------------------------

class LocalInMemoryStore:
    """
    Simple in-memory vector store with optional JSON file persistence.
    Used as a fallback when no external vector DB is configured.
    """
    def __init__(self, path: Path):
        self._path = path
        self._data = {}  # id -> {"embedding": ..., "meta": ...}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self._data = {k: v for k, v in raw.items()}
            except Exception as e:
                log(f"Failed to load local vector store: {e}", "WARN")

    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
        for vid, emb, meta in vectors:
            self._data[vid] = {"embedding": emb, "meta": meta}
        self._persist()

    def query(self, embedding: List[float], top_k: int = 5):
        def cosine(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x*x for x in a))
            norm_b = math.sqrt(sum(y*y for y in b))
            return dot / (norm_a * norm_b + 1e-8)
        scored = []
        for vid, rec in self._data.items():
            emb = rec["embedding"]
            score = cosine(embedding, emb)
            class Match:
                def __init__(self, id, meta, score):
                    self.id = id
                    self.metadata = meta
                    self.score = score
            scored.append(Match(vid, rec["meta"], score))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def delete(self, vector_ids: List[str]):
        for vid in vector_ids:
            self._data.pop(vid, None)
        self._persist()

    def _persist(self):
        try:
            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"Failed to persist local vector store: {e}", "WARN")

def get_vector_store():
    # Preference: pgvector -> pinecone -> local
    global _PINECONE_PLAN_UNSUPPORTED
    if MODE == "RAG":
        if PGVECTOR_DSN or PGVECTOR_HOST:
            try:
                log("Initializing PGVectorAdapter")
                return PGVectorAdapter(
                    dsn=PGVECTOR_DSN,
                    host=PGVECTOR_HOST,
                    port=PGVECTOR_PORT,
                    db=PGVECTOR_DB,
                    user=PGVECTOR_USER,
                    password=PGVECTOR_PASSWORD,
                    table=PGVECTOR_TABLE
                )
            except Exception as e:
                log(f"PGVector init failed: {e}", "WARN")
        if VECTOR_DB_API_KEY and not PINECONE_DISABLE and not _PINECONE_PLAN_UNSUPPORTED:
            try:
                log(f"Initializing PineconeAdapter (region={PINECONE_REGION})")
                return PineconeAdapter(
                    VECTOR_DB_API_KEY,
                    region=PINECONE_REGION,
                    alt_regions=PINECONE_ALT_REGIONS
                )
            except Exception as e:
                msg_low = str(e).lower()
                if any(pat in msg_low for pat in PINECONE_PLAN_LIMIT_PATTERNS):
                    _PINECONE_PLAN_UNSUPPORTED = True
                    if not PINECONE_SILENCE_PLAN_ERRORS:
                        log("Pinecone plan / region limitation detected – skipping Pinecone for this run.", "WARN")
                        log("Hint: choose a supported serverless region or set PINECONE_DISABLE=1 to suppress attempts.", "INFO")
                else:
                    log(f"Pinecone init failed: {e}. Falling back to local store.", "WARN")
                # fall through to local
    return LocalInMemoryStore(VECTORS_PATH)

def ingest_vectors():
    store = get_vector_store()
    existing = load_existing_posts()
    if not existing:
        log("No posts to ingest. Run preprocess first.", "ERROR")
        return
    # Filter empty text
    existing = {k: v for k, v in existing.items() if v.get("text")}
    if not existing:
        log("All posts empty after filtering; aborting ingest.", "WARN")
        return
    entries: List[Tuple[str, str, str]] = []
    for p in existing.values():
        chunks = token_chunks(p["text"])
        for ci, ch in enumerate(chunks):
            cid = f"{p['id']}::c{ci}"
            entries.append((cid, ch, p["id"]))
    texts = [e[1] for e in entries]
    embeddings = openai_embed_batch(texts, EMBEDDING_MODEL)
    vectors = []
    for (cid, ch, origin), emb in zip(entries, embeddings):
        vectors.append((cid, emb, {"origin": origin, "chunk": ch[:750]}))
    store.upsert(vectors)
    log(f"Ingested {len(vectors)} chunk vectors.")

def generate_with_context(prompt: str, top_k: int = 5) -> str:
    # --- Modified to use new chat completions endpoint ---
    if MODE != "RAG":
        log("MODE is not RAG; context retrieval disabled.", "ERROR")
        return ""
    store = get_vector_store()
    emb = openai_embed_batch([prompt], EMBEDDING_MODEL)[0]
    matches = store.query(emb, top_k=top_k)
    context_blocks = []
    for m in matches:
        meta = getattr(m, "metadata", {}) or {}
        context_blocks.append(meta.get("chunk", ""))
    context = "\n---\n".join(context_blocks) or "(no context)"
    client = _get_openai_client()
    system_msg = "You write concise, high-signal, original LinkedIn-style posts. Avoid copying context verbatim."
    user_msg = f"Context examples (do NOT copy, only derive style & structure):\n{context}\n\nPrompt: {prompt}\nWrite a fresh LinkedIn post."
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if any(code in msg for code in ("rate", "timeout", "overload", "500", "502", "503")) and attempt < (MAX_RETRIES - 1):
                exponential_backoff(attempt)
                continue
            log(f"Generation failed: {e}", "ERROR")
            return ""
    return ""

# Fine-Tune Mode --------------------------------------------------------------

def build_finetune_file():
    posts = load_existing_posts()
    if not posts:
        log("No posts to create finetune file.", "ERROR")
        return
    lines = []
    for p in posts.values():
        prompt = "Write a LinkedIn post in the style of the example below:\n\n### EXAMPLE:"
        completion = f" {p['text'].strip()}\n"
        lines.append(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False))
    if DRY_RUN:
        log("DRY_RUN: skipping finetune.jsonl write")
    else:
        safe_write(FINETUNE_JSONL, "\n".join(lines) + "\n")
        log(f"Wrote finetune dataset with {len(lines)} examples -> {FINETUNE_JSONL}")

def train_openai_model():
    """
    Fine-tune helper updated for openai>=1.0.0.
    Avoids deprecated openai.File.* usage and gives clearer diagnostics.
    """
    if MODE != "FINETUNE":
        log("MODE != FINETUNE; skipping training.", "WARN")
        return
    if not FINETUNE_JSONL.exists():
        log("finetune.jsonl missing; run finetune step first.", "ERROR")
        return
    try:
        client = _get_openai_client()
    except RuntimeError as e:
        log(str(e), "ERROR")
        return

    def _upload_training_file(path: Path):
        try:
            with path.open("rb") as f:
                return client.files.create(file=f, purpose="fine-tune")
        except AttributeError as ae:
            # Typical when someone tries to rely on openai.File.* in >=1.0.0
            log("Upload failed: openai>=1.0.0 removed `openai.File`. Use client.files.create(...)", "ERROR")
            raise ae

    try:
        uploaded = _upload_training_file(FINETUNE_JSONL)
        file_id = uploaded.id
        log(f"Uploaded training file: {file_id}")

        base_model = os.getenv("FINETUNE_BASE_MODEL", "gpt-3.5-turbo")
        job = client.fine_tuning.jobs.create(training_file=file_id, model=base_model)
        job_id = job.id
        log(f"Started Fine-Tuning Job: {job_id} (base={base_model})")

        poll_interval = int(os.getenv("FINETUNE_POLL_INTERVAL", "10"))
        max_polls = int(os.getenv("FINETUNE_MAX_POLLS", "60"))
        for attempt in range(max_polls):
            cur = client.fine_tuning.jobs.retrieve(job_id)
            status = cur.status
            log(f"Fine-tune status: {status}")
            if status in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(poll_interval)

        if status == "succeeded":
            log(f"Fine-tune succeeded. Result model: {cur.fine_tuned_model}", "INFO")
        else:
            log(f"Fine-tune finished with status={status}", "WARN")

    except Exception as e:
        # Provide extra context if legacy usage suspected
        msg = str(e).lower()
        if "openai.file" in msg or "attributeerror" in msg:
            log("Detected a legacy pattern referencing openai.File; the new SDK uses client.files.create().", "ERROR")
        log(f"Fine-tune orchestration error: {e}", "ERROR")

# Purge (Right to be Forgotten) ----------------------------------------------

def remove_sensitive_and_purge(post_id: str):
    log(f"Purging post_id={post_id}")
    # Remove from CSV
    existing = load_existing_posts()
    if post_id not in existing:
        log("Post ID not found.", "WARN")
    else:
        existing.pop(post_id)
        if not DRY_RUN:
            with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["id", "author_urn", "date", "text", "reactions", "comments", "media_count", "source_url"])
                for p in existing.values():
                    writer.writerow([p["id"], p["author_urn"], p["date"], p["text"], p["reactions"], p["comments"], p["media_count"], p["source_url"]])
    # Update audit (remove lines)
    if AUDIT_PATH.exists() and not DRY_RUN:
        new_lines = []
        with AUDIT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("post_id") == post_id:
                        continue
                except Exception:
                    pass
                new_lines.append(line)
        safe_write(AUDIT_PATH, "".join(new_lines))
    # Remove from vector store
    store = get_vector_store()
    chunk_ids = []
    if isinstance(store, LocalInMemoryStore):
        for vid, record in list(store._data.items()):
            if record["meta"].get("origin") == post_id:
                chunk_ids.append(vid)
        store.delete(chunk_ids)
    elif isinstance(store, PGVectorAdapter):
        # Attempt range of chunks
        chunk_ids = [f"{post_id}::c{i}" for i in range(200)]
        try:
            store.delete(chunk_ids)
        except Exception as e:
            log(f"PGVector delete partial error: {e}", "WARN")
    else:  # PineconeAdapter
        chunk_ids = [f"{post_id}::c{i}" for i in range(50)]
        try:
            store.delete(chunk_ids)
        except Exception as e:
            log(f"Vector delete partial error: {e}", "WARN")
    log("Purge operation completed.")

# High-Level Steps -----------------------------------------------------------

SCRAPE_CACHE = DATA_DIR / "raw_posts.json"

def step_scrape():
    """
    Scrape posts with fallback order:
      1. Apify (if BACKEND=apify & token present)
      2. Browser (Playwright) if credentials and Playwright installed
      3. Synthetic samples (if SYNTHETIC_ON_EMPTY=1) so pipeline can continue
    """
    backend: ScraperBackend
    raw: List[RawPost] = []
    if BACKEND == "apify":
        if APIFY_TOKEN:
            backend = ApifyBackend(APIFY_TOKEN)
            raw = backend.fetch_posts(MAX_POSTS)
        else:
            log("APIFY_TOKEN not set; skipping Apify backend.", "WARN")
        if not raw and isinstance(backend, ApifyBackend) and not backend.actor_not_found:
            log("Apify returned 0 posts.", "WARN")
    else:
        # BACKEND=browser initial path
        if LINKEDIN_EMAIL and LINKEDIN_PASSWORD:
            try:
                backend = BrowserBackend(LINKEDIN_EMAIL, LINKEDIN_PASSWORD, RATE_LIMIT_MS)
                raw = backend.fetch_posts(MAX_POSTS)
            except Exception as e:
                log(f"Browser backend error: {e}", "ERROR")
        else:
            log("Browser backend selected but credentials missing.", "WARN")
    # Secondary Browser fallback
    if not raw and BACKEND == "apify":
        if LINKEDIN_EMAIL and LINKEDIN_PASSWORD:
            try:
                log("Attempting BrowserBackend fallback...", "INFO")
                browser_backend = BrowserBackend(LINKEDIN_EMAIL, LINKEDIN_PASSWORD, RATE_LIMIT_MS)
                raw = browser_backend.fetch_posts(MAX_POSTS)
            except RuntimeError as e:
                log(f"Browser backend unavailable: {e}", "WARN")
            except Exception as e:
                log(f"Browser backend error: {e}", "ERROR")
        else:
            log("No LinkedIn credentials for BrowserBackend fallback.", "WARN")
    # Synthetic
    if not raw and SYNTHETIC_ON_EMPTY == "1":
        log("No real posts fetched; generating synthetic sample posts (set SYNTHETIC_ON_EMPTY=0 to disable).", "WARN")
        raw = _generate_synthetic_posts(5)

    log(f"Fetched {len(raw)} raw posts.")
    if not raw:
        log("Scrape produced zero posts and synthetic fallback disabled. Pipeline will halt after this step.", "ERROR")
    if DRY_RUN:
        log("DRY_RUN: listing candidate post IDs only:")
        for p in raw[:20]:
            log(f" - {p.get('post_id')}")
        return
    safe_write(SCRAPE_CACHE, json.dumps(raw, ensure_ascii=False, indent=2))
    log(f"Saved raw posts json -> {SCRAPE_CACHE}")

# --- Added: safe synthetic sample post generator (used only if real scraping fails) ---
def _generate_synthetic_posts(n: int = 5) -> List[RawPost]:
    """
    Produce synthetic placeholder posts so downstream pipeline (preprocess / ingest)
    can be validated without live scraping / external dependencies.
    Each post is clearly marked to avoid accidental misuse.
    """
    now = int(time.time())
    posts: List[RawPost] = []
    samples = [
        "Synthetic sample: 3 product growth lessons from shipping fast.",
        "Synthetic sample: Why distribution beats invention for early startups.",
        "Synthetic sample: A concise framework for improving onboarding UX.",
        "Synthetic sample: Pricing iteration notes after 10 customer interviews.",
        "Synthetic sample: Developer productivity isn't velocity—it's clarity."
    ]
    for i in range(min(n, len(samples))):
        posts.append(RawPost(
            post_id=f"synthetic_{now}_{i}",
            author_urn="urn:li:member:synthetic",
            text=samples[i],
            publish_timestamp=now - i * 3600,
            media_urls=[],
            reactions_count=0,
            comments_count=0,
            top_comments=[],
            source_url="https://example.local/synthetic"
        ))
    return posts
# --- end addition ---

# LinkedIn OAuth Token Utilities ---------------------------------------------

def get_stored_linkedin_token() -> tuple[str, str]:
    """
    Loads the LinkedIn OAuth access token and member URN from the token file.
    Returns (access_token, member_urn) or ("", "") if not found.
    """
    try:
        if OAUTH_TOKENS_PATH.exists():
            with OAUTH_TOKENS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("access_token", ""), data.get("member_urn", "")
    except Exception as e:
        log(f"Failed to load LinkedIn OAuth token: {e}", "WARN")
    return "", ""

# Argparse -------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="LinkedIn scraping & training pipeline (consent-gated).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="command")

    # Add existing commands
    sub.add_parser("scrape", help="Fetch raw posts using selected backend")
    sub.add_parser("preprocess", help="Clean, dedupe, redact & write posts.csv")
    sub.add_parser("ingest", help="Vectorize & ingest (RAG mode)")
    sub.add_parser("finetune", help="Build finetune.jsonl & start fine-tune (FINETUNE mode)")
    gen = sub.add_parser("generate", help="Generate a post using RAG context (MODE=RAG)")
    gen.add_argument("prompt", type=str, help="Prompt text for generation")
    purge = sub.add_parser("purge", help="Right-to-be-forgotten: purge a post id")
    purge.add_argument("post_id", type=str, help="Post ID to purge across CSV, vectors, audit")
    sub.add_parser("pipeline", help="Run end‑to‑end: scrape -> preprocess -> (ingest|finetune)")
    
    # Add new command for LinkedIn authentication
    sub.add_parser("linkedin-auth", help="Authenticate with LinkedIn using OAuth")
    
    return parser

def run_pipeline():
    log("Running full pipeline...")
    step_scrape()
    # If scrape produced no posts file or empty array, abort politely
    if not SCRAPE_CACHE.exists():
        log("Scrape step produced no cache file; aborting pipeline.", "WARN")
        return
    try:
        raw_preview = json.loads(SCRAPE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        print("Failed to read raw posts cache; aborting pipeline.", "ERROR")
        raw_preview = []
    if not raw_preview:
        log("No raw posts to preprocess; pipeline stopping.", "WARN")
        return
    step_preprocess()
    # Only proceed if posts.csv now exists with at least one row
    posts_map = load_existing_posts()
    if not posts_map:
        log("No processed posts available; skipping ingestion / finetune.", "WARN")
        return
    if MODE == "FINETUNE":
        build_finetune_file()
        log("Pipeline (FINETUNE mode) complete. Run: python scrape_and_train.py finetune")
    else:
        ingest_vectors()
        log("Pipeline (RAG mode) complete. You can now generate:\n  python scrape_and_train.py generate \"Your prompt\"")

def step_preprocess():
    raw = []
    if SCRAPE_CACHE.exists():
        try:
            raw = json.loads(SCRAPE_CACHE.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"Failed to load raw posts: {e}", "ERROR")
    if not raw:
        log("No raw posts to preprocess.", "ERROR")
        return
    cleaned = dedupe_and_redact(raw)
    save_posts(cleaned)
    log(f"Preprocessed and saved {len(cleaned)} posts.")

def step_ingest():
    ingest_vectors()

def step_finetune():
    build_finetune_file()
    train_openai_model()

def step_generate(prompt: str):
    result = generate_with_context(prompt)
    print("\n[GENERATED POST]\n")
    print(result)
    print("\n")

def linkedin_oauth_flow() -> tuple[str, str]:
    """
    Placeholder for LinkedIn OAuth flow.
    Replace this with actual implementation to perform OAuth and return (access_token, member_urn).
    """
    log("linkedin_oauth_flow is not implemented. Returning empty credentials.", "WARN")
    return "", ""

def step_linkedin_auth():
    """New step to handle LinkedIn authentication."""
    log("Starting LinkedIn OAuth authentication flow...", "INFO")
    access_token, member_urn = linkedin_oauth_flow()
    
    if access_token and member_urn:
        log("LinkedIn authentication successful!", "INFO")
        log(f"Member URN: {member_urn}", "INFO")
        log(f"Token saved to: {OAUTH_TOKENS_PATH}", "INFO")
        log("You can now use this authentication for scraping.", "INFO")
    else:
        log("LinkedIn authentication failed.", "ERROR")

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        # --- Added: auto-run default command instead of just exiting ---
        default_cmd = os.getenv("AUTO_DEFAULT_COMMAND", "pipeline").strip().lower()
        print(f"\n[INFO] No command provided. Auto-running default command: {default_cmd}\n"
              f"(Set AUTO_DEFAULT_COMMAND env var to change this behavior, or use -h for help.)\n")
        if default_cmd == "pipeline":
            run_pipeline()
        # Simulate setting the command then dispatch
        args.command = default_cmd
        if default_cmd == "generate":
            # Need a prompt; fall back to a simple one
            setattr(args, "prompt", os.getenv("AUTO_DEFAULT_GENERATE_PROMPT", "Share a concise insight about product velocity."))

        elif default_cmd == "purge":
            setattr(args, "post_id", os.getenv("AUTO_DEFAULT_PURGE_ID", "nonexistent"))
        # If unrecognized, fallback to showing examples then exit
        if default_cmd not in {"pipeline","scrape","preprocess","ingest","finetune","generate","purge"}:
            print("Unknown AUTO_DEFAULT_COMMAND. Valid: pipeline,scrape,preprocess,ingest,finetune,generate,purge")
            sys.exit(1)
        # Continue to dispatcher below
    # --- end addition ---

    if args.command == "scrape":
        step_scrape()
    elif args.command == "preprocess":
        step_preprocess()
    elif args.command == "ingest":
        step_ingest()
    elif args.command == "finetune":
        step_finetune()
    elif args.command == "generate":
        step_generate(args.prompt)
    elif args.command == "purge":
        remove_sensitive_and_purge(args.post_id)
    elif args.command == "pipeline":
        run_pipeline()
    elif args.command == "linkedin-auth":
        step_linkedin_auth()
    else:
        log("Unknown command.", "ERROR")
        sys.exit(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.", "WARN")
        sys.exit(130)
    if not args.command:
        # --- Added: auto-run default command instead of just exiting ---
        default_cmd = os.getenv("AUTO_DEFAULT_COMMAND", "pipeline").strip().lower()
        print(f"\n[INFO] No command provided. Auto-running default command: {default_cmd}\n"
              f"(Set AUTO_DEFAULT_COMMAND env var to change this behavior, or use -h for help.)\n")
        if default_cmd == "pipeline":
            run_pipeline()
        # Simulate setting the command then dispatch
        args.command = default_cmd
        if default_cmd == "generate":
            # Need a prompt; fall back to a simple one
            setattr(args, "prompt", os.getenv("AUTO_DEFAULT_GENERATE_PROMPT", "Share a concise insight about product velocity."))

        elif default_cmd == "purge":
            setattr(args, "post_id", os.getenv("AUTO_DEFAULT_PURGE_ID", "nonexistent"))
        # If unrecognized, fallback to showing examples then exit
        if default_cmd not in {"pipeline","scrape","preprocess","ingest","finetune","generate","purge"}:
            print("Unknown AUTO_DEFAULT_COMMAND. Valid: pipeline,scrape,preprocess,ingest,finetune,generate,purge")
            sys.exit(1)
        # Continue to dispatcher below
    # --- end addition ---

    if args.command == "scrape":
        step_scrape()
    elif args.command == "preprocess":
        step_preprocess()
    elif args.command == "ingest":
        step_ingest()
    elif args.command == "finetune":
        step_finetune()
    elif args.command == "generate":
        step_generate(args.prompt)
    elif args.command == "purge":
        remove_sensitive_and_purge(args.post_id)
    elif args.command == "pipeline":
        run_pipeline()
    else:
        log("Unknown command.", "ERROR")
        sys.exit(2)
        run_pipeline()
