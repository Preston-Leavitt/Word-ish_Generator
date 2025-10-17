from fastapi import FastAPI, HTTPException, Request, Body, Response  # Removed UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse  # Updated
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import os, logging, secrets, json, random, datetime
from typing import Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager
from pathlib import Path
import time, requests  # added for robust LinkedIn callback handling
import base64  # added for id_token fallback decoding
import hashlib  # <-- added

from .schemas import GenerationRequest, GenerationResponse, ErrorResponse, Template  # Removed VideoRequest, VideoJob
from .templates import load_templates, get_template_by_id
try:
    from .prompts import SYSTEM_MESSAGE, build_user_prompt
except Exception:
    from .prompts import SYSTEM_MESSAGE, build_user_prompt_with_dm as build_user_prompt
from .openai_client import OpenAIClient
# Removed: from .video_client import SoraClient
from .safety import run_safety_checks
from .utils import extract_json_from_text, normalize_hashtags

def _inject_context(base_pd: str, user_ctx: str) -> str:
    """
    Injects user context into the personal detail string if not already present.
    """
    if not user_ctx:
        return base_pd or ""
    if base_pd and user_ctx in base_pd:
        return base_pd
    if base_pd:
        return f"{base_pd}\n\n{user_ctx}"
    return user_ctx
from .drafts import (
    Draft, DRAFT_STORE, AUTO_PREFS, create_auto_draft, create_manual_draft,
    publish_draft, cancel_draft, edit_draft, get_user_pending_drafts,
    count_user_auto_drafts_today, create_varied_auto_draft
)
from .linkedin_helpers import (
    USER_LINK_STORE, OAUTH_STATE_STORE
)

# Read feature flag for LinkedIn auto-connect (default to "0" if not set)
ENABLE_LINKEDIN_AUTOCONNECT = os.getenv("ENABLE_LINKEDIN_AUTOCONNECT", "0")
# NEW: deployed host domain
HOST_DOMAIN = os.getenv("HOST_DOMAIN", "prestonleavitt.dev")

# Initialize FastAPI app (removed first premature instantiation to avoid losing middleware/routes)
# app = FastAPI(title="LinkedIn Viral Post Generator", version="1.0.0")  # <-- removed

# Scheduler & draft feature globals
scheduler: Optional[AsyncIOScheduler] = None
DAILY_AUTO_LIMIT = int(os.getenv("DRAFT_AUTOGEN_DAILY_LIMIT", "5"))

logger = logging.getLogger("drafts")
logging.basicConfig(level=logging.INFO)

# Global variables
TEMPLATE_STORE: Dict[str, Template] = {}
openai_client = None
# Removed VIDEO_JOBS and Sora-related globals

# --- added: simple in-memory auth/profile stores ---
USER_DB: Dict[str, Dict[str, Any]] = {}       # email -> {user_id, email, pw_hash, key_points, about_me, display_name}
USER_BY_ID: Dict[str, Dict[str, Any]] = {}    # user_id -> record
AUTH_TOKENS: Dict[str, str] = {}              # token -> user_id

# --- added: users persistence (disk) ---
BASE_DIR = Path(__file__).resolve().parent.parent  # points to d:\Linkedin
USERS_DATA_DIR = Path(BASE_DIR) / "data"
USERS_DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_JSON = USERS_DATA_DIR / "users.json"
# --- added: auto-generate persistence paths/state ---
AUTO_JSON = USERS_DATA_DIR / "auto.json"
AUTO_STATE: Dict[str, Dict[str, Any]] = {}  # user_id -> {enabled: bool, last_post_at: iso|None, next_run_at: iso|None}
# --- end added ---

def _persist_auto_state():
    """Persist AUTO_STATE to disk."""
    try:
        payload = {
            "version": 1,
            "auto_state": AUTO_STATE
        }
        _atomic_write_json(AUTO_JSON, payload)
        logging.info("[AutoState] persisted count=%s file=%s", len(AUTO_STATE), AUTO_JSON)
    except Exception as e:
        logging.error("[AutoState] persist_failed err=%s", e)

def _load_auto_state():
    """Load AUTO_STATE from disk."""
    global AUTO_STATE
    try:
        if AUTO_JSON.is_file():
            with AUTO_JSON.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            AUTO_STATE.clear()
            AUTO_STATE.update(data.get("auto_state") or {})
            logging.info("[AutoState] loaded count=%s file=%s", len(AUTO_STATE), AUTO_JSON)
        else:
            logging.info("[AutoState] no auto.json file found (fresh start) path=%s", AUTO_JSON)
    except Exception as e:
        logging.error("[AutoState] load_failed err=%s", e)

def _schedule_auto_gen_at(user_id: str, run_at: datetime.datetime):
    """
    Schedule the auto-generation job for a user at a specific datetime.
    """
    if not scheduler:
        logging.warning("[AutoGen] scheduler not initialized")
        return
    job_id = f"auto_gen_{user_id}"
    # Remove old job if exists
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    scheduler.add_job(
        func=auto_generate_cycle,
        trigger="date",
        run_date=run_at,
        id=job_id,
        kwargs={"user_id": user_id},
        replace_existing=True,
        misfire_grace_time=60
    )
    logging.info("[AutoGen] scheduled job for user=%s at %s", user_id, run_at.isoformat())

def _bootstrap_auto_jobs():
    """Bootstrap scheduled jobs for users with auto-generation enabled."""
    if not scheduler:
        return
    now = datetime.datetime.utcnow()
    for user_id, st in AUTO_STATE.items():
        if not st.get("enabled"):
            continue
        nxt = st.get("next_run_at")
        if nxt:
            try:
                dt = datetime.datetime.fromisoformat(nxt)
                # If next_run_at is in the past, roll it forward to the next valid slot
                if dt <= now:
                    dt = _schedule_next_for_user(user_id, base_time=now)
                    st["next_run_at"] = dt.isoformat()
                else:
                    _schedule_auto_gen_at(user_id, dt)
            except Exception as e:
                logging.warning("[AutoState] bootstrap_job_failed user=%s err=%s", user_id, e)

def _calc_next_run_from_last_post(last_post_at: datetime.datetime) -> datetime.datetime:
    """
    Calculate the next run datetime based on daily limit, spreading runs across 24h.
    interval = round(1440 minutes / DAILY_AUTO_LIMIT). Add a small jitter.
    """
    try:
        per_day = max(1, DAILY_AUTO_LIMIT)
        interval_minutes = max(1, int(round(1440 / per_day)))
    except Exception:
        interval_minutes = 1440  # fallback: once per day
    # small jitter up to 10% of interval (max 10 minutes)
    jitter = min(10, max(0, interval_minutes // 10))
    return last_post_at + datetime.timedelta(minutes=interval_minutes + random.randint(0, jitter))

def _atomic_write_json(path: Path, data: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _persist_users():
    try:
        payload = {
            "version": 1,
            # include auto_generate_enabled + per-user auto config
            "users": [
                {
                    **rec,
                    "auto_generate_enabled": bool(AUTO_STATE.get(rec.get("user_id",""), {}).get("enabled", False))
                }
                for rec in USER_BY_ID.values()
            ]
        }
        _atomic_write_json(USERS_JSON, payload)
        logging.info("[Users] persisted count=%s file=%s", len(USER_BY_ID), USERS_JSON)
    except Exception as e:
        logging.error("[Users] persist_failed err=%s", e)

def _load_users():
    global USER_DB, USER_BY_ID
    try:
        if USERS_JSON.is_file():
            with USERS_JSON.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            users = data.get("users") or []
            USER_DB.clear()
            USER_BY_ID.clear()
            for rec in users:
                uid = rec.get("user_id")
                email = (rec.get("email") or "").lower()
                if uid:  # was: if uid and email
                    # normalize into our in-memory shape; allow empty email
                    USER_BY_ID[uid] = {
                        "user_id": uid,
                        "email": email,
                        "pw_hash": rec.get("pw_hash") or "",
                        "key_points": rec.get("key_points") or "",
                        "about_me": rec.get("about_me") or "",
                        "display_name": rec.get("display_name") or "",
                        "notify_on_draft": bool(rec.get("notify_on_draft", False)),
                        "auto_generate_enabled": bool(rec.get("auto_generate_enabled", False)),
                        # per-user auto config (optional)
                        "auto_posts_per_day": int(rec.get("auto_posts_per_day") or 0) or None,
                        "auto_window_start": (rec.get("auto_window_start") or None),
                        "auto_window_end": (rec.get("auto_window_end") or None),
                    }
                    # map email -> record only if email present
                    if email:
                        USER_DB[email] = USER_BY_ID[uid]
                    # sync AUTO_STATE enabled flag
                    st = AUTO_STATE.setdefault(uid, {"enabled": False, "last_post_at": None, "next_run_at": None})
                    st["enabled"] = bool(rec.get("auto_generate_enabled", False))
            logging.info("[Users] loaded count=%s file=%s", len(USER_BY_ID), USERS_JSON)
        else:
            logging.info("[Users] no users file found (fresh start) path=%s", USERS_JSON)
    except Exception as e:
        logging.error("[Users] load_failed err=%s", e)

def _hash_password(pw: str) -> str:
    return hashlib.sha256((pw or "").encode("utf-8")).hexdigest()

def _issue_token(user_id: str) -> str:
    tok = secrets.token_urlsafe(32)
    AUTH_TOKENS[tok] = user_id
    return tok

# NEW: cookie helpers for consistent prod/dev behavior
def _set_auth_cookie(response: Response, token: str):
    """
    Set auth cookie consistently:
    - Name: auth_token
    - Dev: secure=False, host-only cookie
    - Prod: secure=True, domain=HOST_DOMAIN
    """
    prod = (os.getenv("ENV", "").lower() == "prod")
    kwargs = {
        "key": "auth_token",
        "value": token,
        "httponly": True,
        "samesite": "lax",
        "path": "/",
        "max_age": 60 * 60 * 24 * 30
    }
    if prod:
        kwargs["secure"] = True
        if HOST_DOMAIN:
            kwargs["domain"] = HOST_DOMAIN
    else:
        kwargs["secure"] = False
    response.set_cookie(**kwargs)

def _clear_auth_cookie(response: Response):
    """
    Best-effort clear for both current and legacy cookie names,
    with/without domain to cover dev + prod.
    """
    try:
        response.delete_cookie(key="auth_token", path="/")
    except Exception:
        pass
    try:
        if HOST_DOMAIN:
            response.delete_cookie(key="auth_token", domain=HOST_DOMAIN, path="/")
    except Exception:
        pass
    # Legacy cookie cleanup
    try:
        response.delete_cookie(key="session", path="/")
    except Exception:
        pass
    try:
        if HOST_DOMAIN:
            response.delete_cookie(key="session", domain=HOST_DOMAIN, path="/")
    except Exception:
        pass

def _user_from_auth_header(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
        uid = AUTH_TOKENS.get(token)
        if uid:
            return uid
    # NEW: header token fallback
    try:
        header_tok = request.headers.get("X-Auth-Token")
        if header_tok:
            uid = AUTH_TOKENS.get(header_tok)
            if uid:
                return uid
    except Exception:
        pass
    # NEW: cookie fallbacks (auth_token preferred, legacy 'session' supported)
    try:
        cookie_tok = request.cookies.get("auth_token") or request.cookies.get("session")
        if cookie_tok:
            uid = AUTH_TOKENS.get(cookie_tok)
            if uid:
                return uid
    except Exception:
        pass
    return None

def _get_profile_key_points(user_id: Optional[str]) -> str:
    if not user_id:
        return ""
    rec = USER_BY_ID.get(user_id) or {}
    kp = (rec.get("key_points") or "").strip()
    if len(kp) > 2000:
        kp = kp[:2000]
    return kp
# --- end added ---

# --- added: helper to read About Me from profile ---
def _get_profile_about_me(user_id: Optional[str]) -> str:
    if not user_id:
        return ""
    rec = USER_BY_ID.get(user_id) or {}
    am = (rec.get("about_me") or "").strip()
    if len(am) > 1000:
        am = am[:1000]
    return am
# --- end helper ---

# Track last manual template used per user
LAST_MANUAL_TEMPLATE: Dict[str, str] = {}
ABOUT_CONTEXT_STORE: Dict[str, str] = {}  # <-- added: persist latest user "About You" context

# Added: Track recent template/tone/goal/audience history per user for diversity
META_HISTORY: Dict[str, list] = {}
DIVERSITY_WINDOW = 4  # Used for plan diversification
TONE_POOL = [
    "candid", "inspirational", "authoritative", "humorous", "empathetic", "analytical", "optimistic", "pragmatic"
]
GOAL_POOL = [
    "engagement", "authority", "lead-gen", "storytelling", "curiosity", "conversation", "education"
]
AUDIENCE_POOL = [
    "founders", "executives", "marketers", "engineers", "students", "job seekers", "investors", "general professionals"
]

BASE_DIR = Path(__file__).resolve().parent.parent  # points to d:\Linkedin
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Ensure static dir exists (no video dirs anymore)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI startup and shutdown."""
    global TEMPLATE_STORE, openai_client, scheduler
    TEMPLATE_STORE = load_templates()
    _load_users()  # ...existing...
    try:
        openai_client = OpenAIClient()
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {e}")
    logger.info(f"Loaded {len(TEMPLATE_STORE)} templates")
    # Initialize scheduler
    scheduler = AsyncIOScheduler()
    scheduler.start()
    logging.info("APScheduler started for draft auto-publish")
    # Add auto connect scan job if not present
    def _run_auto_connect_cycle():
        """
        Placeholder for auto-connect scan logic.
        Implement the actual logic here if needed.
        """
        logger.info("[AutoConnect] _run_auto_connect_cycle executed (no-op placeholder)")
        # Actual implementation would go here

    try:
        if not scheduler.get_job("auto_connect_scan"):
            scheduler.add_job(_run_auto_connect_cycle, "interval", seconds=120, id="auto_connect_scan", replace_existing=True)
    except Exception:
        pass
    # After templates loaded:
    if not (INDEX_HTML.exists()):
        logging.warning("[Startup] index.html missing path=%s", INDEX_HTML)
    else:
        logging.info("[Startup] index.html found size=%s bytes", INDEX_HTML.stat().st_size)
    # --- added: load auto state and bootstrap jobs after scheduler is ready ---
    _load_auto_state()
    _bootstrap_auto_jobs()
    # --- end added ---
    yield
    scheduler.shutdown()

app = FastAPI(
    title="LinkedIn Viral Post Generator",
    version="1.0.0",
    lifespan=lifespan
)

# Re-apply CORS AFTER final app instantiation (was previously applied to discarded instance)
app.add_middleware(
    CORSMiddleware,
    # NEW: restrict to deployed origins; keep credentials enabled
    allow_origins=[f"https://{HOST_DOMAIN}", f"https://www.{HOST_DOMAIN}"],
    allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Mount static using absolute path to avoid cwd issues
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def _sanitize_user_context(user_context):
    """
    Basic sanitizer for user context strings.
    - Trims leading/trailing whitespace
    - Normalizes line endings
    - Collapses empty lines
    - Caps length to 500 chars
    """
    if not user_context:
        return ""
    try:
        s = str(user_context)
    except Exception:
        return ""
    s = s.strip()
    # Normalize line endings to \n
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Strip each line and drop empties
    lines = [ln.strip() for ln in s.split("\n")]
    s = "\n".join([ln for ln in lines if ln])
    if len(s) > 500:
        s = s[:500]
    return s

# --- added: strict JSON regen/repair helper for generations ---
def _regen_strict_json(user_prompt: str, temperature: float = 0.2) -> Dict:
    """
    Force-generate strict JSON with required keys when the first call fails.
    """
    if openai_client is None:
        return {}
    required_keys = "post, hooks, hashtags, image_prompt, tl;dr, cta, follow_up_angle, dm_cta, dm_flow"
    sys = (
        "You output ONLY minified JSON with these exact keys: "
        "post, hooks, hashtags, image_prompt, tl;dr, cta, follow_up_angle, dm_cta, dm_flow. "
        "No commentary, no markdown, no extra text."
    )
    usr = (
        "Regenerate a valid response for this prompt and return STRICT JSON only.\n\n"
        "PROMPT:\n" + user_prompt
    )
    try:
        raw = openai_client.generate_completion(
            system_message=sys,
            user_message=usr,
            temperature=temperature
        ) or ""
        return extract_json_from_text(raw) or {}
    except Exception as e:
        logging.warning("[GenRepair] strict_json_regen_failed err=%s", e)
        return {}
# --- end added ---

# Lightweight readiness probe
@app.get("/__ping")
async def __ping():
    return {"ok": True, "templates": len(TEMPLATE_STORE)}

# Root handler: redirect to static index to avoid path issues
@app.get("/")
async def root():
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

# Optional: direct index.html route (serves the actual file)
@app.get("/index.html")
async def index_html():
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML, media_type="text/html")
    return HTMLResponse(
        f"<h3>Frontend index.html not found at {INDEX_HTML}</h3>"
        "<p>Ensure static assets are placed in the /static directory.</p>",
        status_code=200
    )

@app.get("/templates")
async def get_templates():
    """Get available templates."""
    return {"templates": [t.model_dump() for t in TEMPLATE_STORE.values()]}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "templates_loaded": len(TEMPLATE_STORE)}

# REMOVE the duplicate root JSON endpoint that overrides the HTML UI
# @app.get("/")
# async def root():
#     return {
#         "message": "LinkedIn Viral Post Generator API",
#         "version": "1.0.0",
#         "docs": "/docs",
#         "endpoints": {
#             "generate": "POST /generate",
#             "templates": "GET /templates",
#             "health": "GET /health"
#         }
#     }

# Provide same info under /api instead (non-conflicting)
@app.get("/api")
async def api_root():
    return {
        "message": "LinkedIn Viral Post Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "generate": "POST /generate",
            "templates": "GET /templates",
            "health": "GET /health",
            "drafts": {
                "toggle": "POST /api/drafts/auto-generate-toggle",
                "generate": "POST /api/drafts/generate",
                "pending": "GET /api/drafts/pending",
                "edit": "POST /api/drafts/{id}/edit",
                "cancel": "POST /api/drafts/{id}/cancel",
                "post_now": "POST /api/drafts/{id}/post-now"
            },
            "linkedin": {
                "connect": "GET /api/linkedin/connect",
                "status": "GET /api/linkedin/status",
                "disconnect": "POST /api/linkedin/disconnect"
            },
            "dm": {
                "suggest": "POST /api/dm/suggest"
            },
            "auth": {  # <-- added
                "register": "POST /api/auth/register",
                "login": "POST /api/auth/login",
                "logout": "POST /api/auth/logout",
                "me_get": "GET /api/me",
                "me_update": "POST /api/me"
            }
        }
    }

# =============== Draft / Generation Logic (unchanged) ===============
def _validate_user(request: Request) -> str:
    """Accept Bearer token first, fallback to X-User-Id for legacy."""
    uid = _user_from_auth_header(request)
    if uid:
        return uid
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing Authorization or X-User-Id")
    return user_id

def _schedule_publish(draft: Draft):
    """Schedule delayed publish (idempotent)."""
    if not scheduler:
        logger.warning("Scheduler not initialized")
        return
    # Remove old job if exists
    if draft.job_id and scheduler.get_job(draft.job_id):
        scheduler.remove_job(draft.job_id)
    job_id = f"publish_{draft.id}"
    scheduler.add_job(
        func=_attempt_publish_job,
        trigger="date",
        run_date=draft.publish_at,
        id=job_id,
        kwargs={"draft_id": draft.id, "user_id": draft.user_id},
        replace_existing=True,
        misfire_grace_time=60
    )
    draft.job_id = job_id

def _attempt_publish_job(draft_id: str, user_id: str):
    """Job entrypoint - safe publish attempt."""
    drafts = DRAFT_STORE.get(user_id, {})
    draft = drafts.get(draft_id)
    if not draft:
        return
    publish_draft(draft)  # idempotent inside
    # schedule next using per-user config (single authoritative path)
    try:
        if getattr(draft, "auto_generated", False):
            now = datetime.datetime.utcnow()
            st = AUTO_STATE.setdefault(user_id, {"enabled": True, "last_post_at": None, "next_run_at": None})
            st["last_post_at"] = now.isoformat()
            if st.get("enabled"):
                next_run = _schedule_next_for_user(user_id, base_time=now)
                st["next_run_at"] = next_run.isoformat()
            _persist_auto_state()
    except Exception as e:
        logging.warning("[AutoGen] post_hook_schedule_failed user=%s err=%s", user_id, e)

def _core_generate(
    template: Template,
    tone: str = "candid",
    audience: str = "founders",
    goal: str = "engagement",
    key_facts: str = "",
    personal_detail: str = "",
    temperature: float = 0.5
) -> Dict:
    """
    Reuse existing OpenAI generation logic for drafts.
    Returns dict with keys: post, hooks, hashtags, image_prompt, tl;dr, cta, follow_up_angle
    """
    # Build a minimal request-like object
    class _Req:
        def __init__(self):
            self.template_id = template.id
            self.tone = tone
            self.audience = audience
            self.goal = goal
            self.key_facts = key_facts or "Quick insight about growth"
            self.personal_detail = personal_detail or "Personal experience"
            self.temperature = temperature
    req = _Req()
    user_prompt = build_user_prompt(req, template)

    # --- Added: prompt debug print for core generation ---
    try:
        trimmed_user = user_prompt if len(user_prompt) <= 12000 else (user_prompt[:12000] + "\n...[truncated]...")
        print(f"\n====== OPENAI PROMPT (core_generate) template={template.id} ======\n"
              f"[System]\n{SYSTEM_MESSAGE}\n\n[User]\n{trimmed_user}\n"
              f"--- temperature={req.temperature} ======\n")
        logging.debug("[OpenAI][core_generate] template=%s len_user=%s", template.id, len(user_prompt))
    except Exception:
        pass
    # -----------------------------------------------------

    # First attempt
    response_text = openai_client.generate_completion(
        system_message=SYSTEM_MESSAGE,
        user_message=user_prompt,
        temperature=req.temperature
    )
    data = extract_json_from_text(response_text) or {}

    # Heuristic: invalid or fallback -> retry + repair
    def _is_bad(d: Dict) -> bool:
        p = (d.get("post") or "").strip().lower()
        return not p or "draft generation fallback content" in p

    if not data or _is_bad(data):
        # Second attempt with lower temperature
        try:
            retry_text = openai_client.generate_completion(
                system_message=SYSTEM_MESSAGE,
                user_message=user_prompt,
                temperature=0.2
            ) or ""
            data = extract_json_from_text(retry_text) or {}
        except Exception as e:
            logging.warning("[OpenAI][core_generate] retry_exception err=%s", e)

    if not data or _is_bad(data):
        # Final attempt: force strict JSON regen/repair
        repaired = _regen_strict_json(user_prompt, temperature=0.2)
        if repaired:
            data = repaired

    # Defensive defaults
    if not isinstance(data, dict):
        data = {}
    data.setdefault("post", "Draft generation fallback content.")
    data.setdefault("hooks", ["Hook A", "Hook B", "Hook C"])
    data.setdefault("hashtags", ["growth", "startup"])
    data.setdefault("image_prompt", "Professional abstract background")
    # Normalize tl;dr key
    if "tl;dr" not in data:
        data["tl;dr"] = data.get("tl_dr", "Summary not provided")
    data.setdefault("cta", "Share your thoughts below.")
    data.setdefault("follow_up_angle", "Explore a deeper tactic next time.")
    return data;

# --- Added: autonomous draft generator (no DM flow, model chooses meta) ---
def _draft_ai_generate(
    template: Template,
    personal_detail: str = "",
    temperature: float = 0.6,
    **kwargs
) -> Dict:
    """
    Draft-specific generation:
      - Model decides template_style (describe pattern), tone, goal, audience, key_facts.
      - Returns JSON WITHOUT dm_flow, dm_cta, image_prompt.
      - Ensures keys: post, hooks, hashtags, template_style, tone, goal, audience, key_facts, tl;dr, cta.
    """
    if not openai_client:
        return {
            "post": "Generation unavailable (OpenAI client not initialized).",
            "hooks": ["Fallback hook 1", "Fallback hook 2", "Fallback hook 3"],
            "hashtags": ["ai", "draft"],
            "template_style": "fallback",
            "tone": "informative",
            "goal": "engagement",
            "audience": "general professionals",
            "key_facts": ["No facts available"],
            "tl;dr": "Fallback summary.",
            "cta": "Share your thoughts below."
        }

    user_ctx = personal_detail.strip() if personal_detail else ""
    instruction = f"""
You are an assistant generating a single high-quality LinkedIn post.

You must:
1. Infer/decide the most suitable: template_style (short descriptive label), tone, goal, audience.
2. Create 3-5 concise key_facts (array) grounded ONLY in the provided user context (do NOT fabricate roles not present).
3. Write the post body (aim 180-260 words) with strong scroll-stopping opening.
4. Provide 3 alternative hooks (hooks array) – they may differ from the opening line.
5. Provide 6-10 relevant lowercase hashtags (array, no '#').
6. Provide a concise tl;dr (summary).
7. Provide a short cta (call to action).
8. DO NOT include any image prompt, DM flow, messaging funnel, or promotional hard sell.
9. If the context is empty, stay neutral and avoid assuming founder / CEO unless explicitly present.

Return ONLY strict minified JSON with keys:
post (string),
hooks (array of strings),
hashtags (array of strings),
template_style (string),
tone (string),
goal (string),
audience (string),
key_facts (array of strings),
tl;dr (string),
cta (string)

User context (verbatim, may be empty):
\"\"\"{user_ctx}\"\"\".
"""
    sys_msg = "You are a focused LinkedIn content generator. Follow instructions exactly. No extra commentary."
    # Print / log
    try:
        trimmed = instruction if len(instruction) <= 12000 else instruction[:12000] + "\n...[truncated]..."
        print(f"\n====== OPENAI PROMPT (draft_ai_generate) ======\n[System]\n{sys_msg}\n\n[User]\n{trimmed}\n--- temperature={temperature} ======\n")
        logging.debug("[OpenAI][draft_ai_generate] ctx_len=%s", len(user_ctx))
    except Exception:
        pass

    raw = openai_client.generate_completion(
        system_message=sys_msg,
        user_message=instruction,
        temperature=temperature
    ) or ""

    data = extract_json_from_text(raw) or {}
    # Defensive defaults
    data.setdefault("post", "Fallback draft post – insufficient model output.")
    data.setdefault("hooks", ["Hook variant 1", "Hook variant 2", "Hook variant 3"])
    data.setdefault("hashtags", ["growth", "startup", "learning"])
    data.setdefault("template_style", data.get("template_style") or "implied_expertise")
    data.setdefault("tone", data.get("tone") or "informative")
    data.setdefault("goal", data.get("goal") or "engagement")
    data.setdefault("audience", data.get("audience") or "professionals")
    # Normalize key_facts to list
    kf = data.get("key_facts")
    if isinstance(kf, str):
        kf = [kf]
    if not isinstance(kf, list) or not kf:
        kf = ["No key facts provided"]
    data["key_facts"] = kf[:6]
    # tl;dr variants
    if "tl;dr" not in data:
        if "tl_dr" in data:
            data["tl;dr"] = data.pop("tl_dr")
        elif "tldr" in data:
            data["tl;dr"] = data.pop("tldr")
        else:
            data["tl;dr"] = "Quick summary not provided."
    data.setdefault("cta", data.get("cta") or "What do you think?")
    # Ensure hashtags simple
    if isinstance(data.get("hashtags"), list):
        data["hashtags"] = [h.lstrip("#").lower()[:40] for h in data["hashtags"][:10]]
    return data

# --- Added: AI meta planning (select template, tone, goal, audience, key_facts) ---
def _ai_plan_template_meta(templates: Dict[str, Template], user_context: str, user_id: str = "anon") -> Dict[str, Any]:
    """
    Uses a lightweight planning prompt to decide:
      template_id, tone, goal, audience, key_facts (array 3-6 items)
    With diversity pressure to avoid repeating recent choices.
    """
    if not openai_client or not templates:
        t = random.choice(list(templates.values()))
        base_plan = {
            "template_id": t.id,
            "tone": random.choice(TONE_POOL),
            "goal": random.choice(GOAL_POOL),
            "audience": random.choice(AUDIENCE_POOL),
            "key_facts": ["No key facts provided"]
        }
        diversified = _diversify_plan(user_id, base_plan, templates)
        _record_plan(user_id, diversified)
        return diversified

    # Summarize recent history for prompt pressure
    hist = META_HISTORY.get(user_id, [])
    recent_lines = []
    for h in hist[-5:]:
        recent_lines.append(f"- template:{h['template_id']} tone:{h['tone']} goal:{h['goal']} audience:{h['audience']}")
    recent_block = "\n".join(recent_lines) if recent_lines else "None"

    # Build compact template catalog
    lines = []
    for t in list(templates.values())[:20]:
        desc = getattr(t, "description", "") or ""
        lines.append(f"{t.id}: {desc[:70].replace(chr(10),' ')}")
    catalog = "\n".join(lines)

    prompt = f"""
You are planning a LinkedIn post.

Available templates (id: brief):
{catalog}

Recent selections (avoid repeating if possible):
{recent_block}

User context (verbatim – do NOT invent roles / titles not explicitly here):
\"\"\"{(user_context or '').strip()}\"\"\".

Task:
Choose a template_id and produce varied tone, goal, audience (avoid repeating recent unless context forces it).
Return STRICT JSON ONLY:
{{
  "template_id": "<one of the listed ids>",
  "tone": "<concise tone (vary if possible)>",
  "goal": "<one of engagement, authority, lead-gen, storytelling, curiosity, conversation, education>",
  "audience": "<concise audience>",
  "key_facts": ["fact1","fact2","fact3"... 3-6 concise factual bullets; neutral if absent; no fabrication]
}}

Rules:
- If context empty: neutral, do not assume founder/CEO.
- Avoid reusing identical (template_id, tone, goal, audience) tuple to the last one unless unavoidable.
Respond with STRICT JSON only.
"""
    sys_msg = "You output only valid JSON for planning. No commentary."
    raw = openai_client.generate_completion(
        system_message=sys_msg,
        user_message=prompt,
        temperature=0.4  # allow variation
    ) or ""
    plan = extract_json_from_text(raw) or {}
    # Validation & defensive fill
    if plan.get("template_id") not in templates:
        plan["template_id"] = random.choice(list(templates.keys()))
    if not plan.get("tone"):
        plan["tone"] = random.choice(TONE_POOL)
    if plan.get("goal") not in GOAL_POOL:
        plan["goal"] = random.choice(GOAL_POOL)
    if not plan.get("audience"):
        plan["audience"] = random.choice(AUDIENCE_POOL)
    kf = plan.get("key_facts")
    if isinstance(kf, str):
        kf = [kf]
    if not isinstance(kf, list) or not kf:
        kf = ["No key facts provided"]
    plan["key_facts"] = kf[:6]

    # Diversify vs history
    plan = _diversify_plan(user_id, plan, templates)
    _record_plan(user_id, plan)
    logging.debug("[Planner] diversified_plan user=%s plan=%s", user_id,
                  {k: plan[k] for k in ['template_id','tone','goal','audience']})
    return plan

# --- Added: internal helper to diversify / mutate repetitive plan ---
def _diversify_plan(user_id: str, plan: Dict[str, Any], templates: Dict[str, Template]) -> Dict[str, Any]:
    hist = META_HISTORY.get(user_id, [])
    if not hist:
        return plan
    recent = {h["template_id"] for h in hist[-DIVERSITY_WINDOW:]}
    # If template repeats and alternates exist
    if plan["template_id"] in recent and len(templates) > len(recent):
        alt_ids = [tid for tid in templates.keys() if tid not in recent]
        if alt_ids:
            plan["template_id"] = random.choice(alt_ids)
    # Tone diversity
    recent_tones = {h["tone"] for h in hist[-DIVERSITY_WINDOW:]}
    if plan["tone"] in recent_tones and len(TONE_POOL) > len(recent_tones):
        alt_tones = [t for t in TONE_POOL if t not in recent_tones]
        if alt_tones:
            plan["tone"] = random.choice(alt_tones)
    # Goal diversity
    recent_goals = {h["goal"] for h in hist[-DIVERSITY_WINDOW:]}
    if plan["goal"] in recent_goals and len(GOAL_POOL) > len(recent_goals):
        alt_goals = [g for g in GOAL_POOL if g not in recent_goals]
        if alt_goals:
            plan["goal"] = random.choice(alt_goals)
    # Audience diversity (only if context is sparse; avoid overriding explicit user audience words)
    if user_id in ABOUT_CONTEXT_STORE:
        ctx = ABOUT_CONTEXT_STORE[user_id].lower()
        explicit_audience_words = [w for w in AUDIENCE_POOL if w in ctx]
    else:
        explicit_audience_words = []
    recent_aud = {h["audience"] for h in hist[-DIVERSITY_WINDOW:]}
    if (plan["audience"] in recent_aud) and not explicit_audience_words:
        alt_aud = [a for a in AUDIENCE_POOL if a not in recent_aud]
        if alt_aud:
            plan["audience"] = random.choice(alt_aud)
    return plan

def _record_plan(user_id: str, plan: Dict[str, Any]):
    hist = META_HISTORY.setdefault(user_id, [])
    hist.append({
        "template_id": plan.get("template_id"),
        "tone": plan.get("tone"),
        "goal": plan.get("goal"),
        "audience": plan.get("audience"),
        "ts": datetime.datetime.utcnow().isoformat()
    })
    if len(hist) > DIVERSITY_WINDOW * 3:
        del hist[:len(hist) - DIVERSITY_WINDOW * 3]

# --- Added: unified plan + generate helper using templates and _core_generate ---
def _plan_and_generate_with_template(user_id: str, about_me: str, key_points: str) -> Dict[str, Any]:
    plan = _ai_plan_template_meta(TEMPLATE_STORE, about_me, user_id=user_id)
    template_obj = TEMPLATE_STORE[plan["template_id"]]
    # derive key facts from saved key points (Drafts)
    kp_lines = [ln.strip() for ln in (key_points or "").splitlines() if ln.strip()]
    key_facts_list = kp_lines[:6] if kp_lines else ["No key points provided"]
    key_facts_str = "; ".join(key_facts_list)

    raw = _core_generate(
        template=template_obj,
        tone=plan["tone"],
        audience=plan["audience"],
        goal=plan["goal"],
        key_facts=key_facts_str,
        personal_detail=about_me,
        temperature=0.6
    )
    for k in ["image_prompt", "dm_flow", "dm_cta", "follow_up_angle"]:
        raw.pop(k, None)
    raw["template_style"] = plan["template_id"]
    raw["tone"] = plan["tone"]
    raw["goal"] = plan["goal"]
    raw["audience"] = plan["audience"]
    raw["key_facts"] = key_facts_list  # reflect Drafts key points in response
    return raw
# --- end helper ---

# --- LinkedIn OAuth Endpoints (unchanged) ---

@app.get("/api/linkedin/connect")
async def linkedin_connect(request: Request):
    """
    LinkedIn OAuth initiation.
    Modes:
      ?popup=1&u=<user_id> -> popup window (direct redirect to LinkedIn)
      ?redirect=1          -> 302 redirect (full page)
      default              -> JSON {auth_url,state}
    Accepts X-User-Id header OR query param u (popup safe).
    """
    from urllib.parse import urlencode

    raw_headers = dict(request.headers)
    masked_headers = {
        k: (
            "***" if "authorization" in k.lower()
            else (v[:12] + "...") if k.lower() == "cookie" and len(v) > 15
            else v
        )
        for k, v in raw_headers.items()
    }
    logging.debug("[LinkedIn][connect] headers=%s params=%s", masked_headers, dict(request.query_params))

    # --- Modified: allow query param 'u' as fallback user id for popup flows ---
    user_id = request.headers.get("X-User-Id") or request.query_params.get("u")
    if user_id:
        # Basic sanitize (alnum, underscore, hyphen, max 48)
        import re
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,48}", user_id):
            return JSONResponse({"detail": "Invalid user id format"}, status_code=400)
    if not user_id:
        if os.getenv("ALLOW_DEV_ANON_CONNECT") == "1":
            user_id = "anonymous-dev"
            logging.info("[LinkedIn][connect] dev fallback user_id=%s", user_id)
        else:
            return JSONResponse(
                {
                    "detail": "Missing X-User-Id header (or ?u= param in popup mode)",
                    "hint": "Send header X-User-Id or append ?u=<id> when opening popup."
                },
                status_code=401
            )
    # -----------------------------------------------------

    client_id = os.getenv("CLIENT_ID") or os.getenv("LINKEDIN_CLIENT_ID")
    redirect_uri = (
        os.getenv("REDIRECT_URI")
        or os.getenv("LINKEDIN_REDIRECT_URI")
        # NEW: default to deployed URL instead of localhost
        or f"https://{HOST_DOMAIN}/api/linkedin/callback"
    )
    scopes = (os.getenv("LINKEDIN_SCOPES") or "r_liteprofile").strip()
    if not client_id or not redirect_uri:
        return JSONResponse(
            {
                "error": "linkedin_config_missing",
                "detail": "CLIENT_ID or REDIRECT_URI missing on server",
                "have_client_id": bool(client_id),
                "have_redirect_uri": bool(redirect_uri)
            },
            status_code=500
        )

    state = f"{user_id}:{secrets.token_urlsafe(16)}"
    OAUTH_STATE_STORE[state] = user_id
    query = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state
    }
    auth_url = "https://www.linkedin.com/oauth/v2/authorization?" + urlencode(query)
    logging.info("[LinkedIn][connect] auth_url=%s state=%s popup=%s", auth_url, state, request.query_params.get("popup"))

    # --- Modified popup branch: single popup window directly redirected (no intermediate JS helper) ---
    if request.query_params.get("popup") == "1":
        from starlette.responses import RedirectResponse
        return RedirectResponse(auth_url)
    # -----------------------------------------------------

    if request.query_params.get("redirect") == "1":
        from starlette.responses import RedirectResponse
        return RedirectResponse(auth_url)

    return JSONResponse({"auth_url": auth_url, "state": state}, status_code=200)

@app.get("/api/linkedin/callback")
async def linkedin_callback(request: Request):
    # Read raw params (avoid opaque framework 400s before we can log/diagnose)
    qp = dict(request.query_params)
    logging.info("[LinkedIn][callback] raw_query_params=%s", qp)

    code = qp.get("code")
    state = qp.get("state")
    if not code:
        return HTMLResponse(
            json.dumps({
                "message": "no code received",
                "query_params": qp
            }, indent=2),
            status_code=400,
            media_type="application/json"
        )

    # Env / config
    client_id = os.getenv("CLIENT_ID") or os.getenv("LINKEDIN_CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET") or os.getenv("LINKEDIN_CLIENT_SECRET")
    redirect_uri = (
        os.getenv("REDIRECT_URI")
        or os.getenv("LINKEDIN_REDIRECT_URI")
        # NEW: default to deployed URL instead of localhost
        or f"https://{HOST_DOMAIN}/api/linkedin/callback"
    )
    scopes_env = os.getenv("LINKEDIN_SCOPES", "")
    is_prod = (os.getenv("ENV", "").lower() == "prod")

    # 1. Exchange code for token
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    try:
        token_resp = requests.post(
            token_url,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        )
        token_json = token_resp.json() if token_resp.content else {}
    except Exception as e:
        logging.error(f"[LinkedIn][callback] token_exchange_exception={e}")
        return HTMLResponse(
            json.dumps({"error": "token_exchange_failed", "detail": str(e)}, indent=2),
            status_code=502,
            media_type="application/json"
        )

    access_token = token_json.get("access_token")
    masked_token = (access_token[:8] + "...") if access_token else None
    logging.info(
        "[LinkedIn][callback] token_response status=%s masked_access_token=%s expires_in=%s scope=%s raw_keys=%s",
        token_resp.status_code,
        masked_token,
        token_json.get("expires_in"),
        token_json.get("scope"),
        list(token_json.keys())
    )

    if token_resp.status_code != 200 or not access_token:
        body = {
            "error": "access_token_missing",
            "token_status": token_resp.status_code,
            "token_json": token_json if not is_prod else {"keys": list(token_json.keys())}
        }
        return HTMLResponse(json.dumps(body, indent=2), status_code=502, media_type="application/json")

    # 2. Compute expires_at
    expires_in = token_json.get("expires_in")
    expires_at = None
    if expires_in:
        try:
            expires_at = int(time.time()) + int(expires_in)
            logging.info("[LinkedIn][callback] computed_expires_at=%s", expires_at)
        except Exception:
            pass

    # 3. Fetch profile (userinfo vs me)
    use_userinfo = ("openid" in scopes_env) or ("profile" in scopes_env)
    if use_userinfo:
        profile_url = "https://api.linkedin.com/v2/userinfo"
        profile_headers = {"Authorization": f"Bearer {access_token}"}
    else:
        profile_url = "https://api.linkedin.com/v2/me"
        profile_headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Restli-Protocol-Version": "2.0.0"
        }

    try:
        prof_resp = requests.get(profile_url, headers=profile_headers, timeout=15)
        try:
            profile_json = prof_resp.json()
        except Exception:
            profile_json = {"raw": prof_resp.text}
    except Exception as e:
        logging.error(f"[LinkedIn][callback] profile_fetch_exception={e}")
        return HTMLResponse(
            json.dumps({
                "error": "profile_fetch_exception",
                "detail": str(e),
                "token_json": token_json if not is_prod else {"keys": list(token_json.keys())}
            }, indent=2),
            status_code=502,
            media_type="application/json"
        )

    if prof_resp.status_code != 200:
        logging.error(f"[LinkedIn][callback] profile_fetch_failed status={prof_resp.status_code} body={profile_json}")
        return HTMLResponse(
            json.dumps({
                "error": "profile_fetch_failed",
                "profile_status": prof_resp.status_code,
                "token_json": token_json if not is_prod else {"keys": list(token_json.keys())},
                "profile_json": profile_json if not is_prod else {"keys": list(profile_json.keys())}
            }, indent=2),
            status_code=502,
            media_type="application/json"
        )

    # 4. Extract stable member_id (LinkedIn OIDC may return 'sub' instead of 'id'; fallback to id_token)
    member_id = profile_json.get("id") or profile_json.get("sub")

    if not member_id and token_json.get("id_token"):
        id_token = token_json["id_token"]
        decoded_claims = None
        # Try PyJWT decode (no signature verification for diagnostics)
        try:
            try:
                import jwt  # type: ignore
                try:
                    decoded_claims = jwt.decode(id_token, options={"verify_signature": False})
                except Exception as e:
                    logging.error(f"[LinkedIn][callback] jwt_decode_failed={e}")
            except ImportError:
                pass
            # Manual base64 fallback
            if decoded_claims is None:
                parts = id_token.split(".")
                if len(parts) >= 2:
                    payload_part = parts[1]
                    padding = '=' * (-len(payload_part) % 4)
                    try:
                        import json as _json
                        payload_bytes = base64.urlsafe_b64decode(payload_part + padding)
                        decoded_claims = _json.loads(payload_bytes.decode("utf-8"))
                    except Exception as e:
                        logging.error(f"[LinkedIn][callback] id_token_manual_decode_failed={e}")
        except Exception as e:
            logging.error(f"[LinkedIn][callback] id_token_processing_error={e}")

        if decoded_claims:
            member_id = decoded_claims.get("sub") or decoded_claims.get("id")

    if not member_id:
        logging.error("[LinkedIn][callback] member_id_unresolved_after_fallbacks")
        return HTMLResponse(
            json.dumps({
                "error": "member_id_unresolved",
                "message": "Profile missing id/sub and id_token did not contain identifiable subject",
                "token_json": token_json if not is_prod else {"keys": list(token_json.keys())},
                "profile_json": profile_json if not is_prod else {"keys": list(profile_json.keys())}
            }, indent=2),
            status_code=502,
            media_type="application/json"
        )

    member_urn = f"urn:li:person:{member_id}"
    logging.info("[LinkedIn][callback] profile_ok member_id=%s member_urn=%s", member_id, member_urn)

    # 5. Persist (in-memory store or placeholder upsert)
    user_key = None
    if state and state in OAUTH_STATE_STORE:
        user_key = OAUTH_STATE_STORE.pop(state)
    else:
        # Derive fallback user key if state not tracked
        user_key = f"state:{state or 'anonymous'}"

    record = {
        "access_token": access_token,        # In production: encrypt / store securely
        "expires_at": expires_at,
        "id": member_id,
        "member_urn": member_urn
    }
    # In-memory persistence
    USER_LINK_STORE[user_key] = record

    # Example placeholder for another persistence layer (uncomment & adapt):
    # try:
    #     db.user_links.update_one({"user_id": user_key}, {"$set": record}, upsert=True)
    # except Exception as e:
    #     logging.error(f"[LinkedIn][callback] db_upsert_failed={e}")

    success_payload = {
        "user_key": user_key,
        "member_urn": member_urn,
        "expires_at": expires_at,
        "masked_access_token": masked_token,
        "debug_note": "Access token masked; full token stored server-side."
    }

    html = f"""
    <html>
      <head><title>LinkedIn OAuth Success</title></head>
      <body style="font-family:Arial;padding:20px;">
        <h2>LinkedIn OAuth Success</h2>
        <p>Authentication completed. You may close this window.</p>
        <pre style="background:#f4f4f4;padding:12px;border:1px solid #ccc;">{json.dumps(success_payload, indent=2)}</pre>
        <script>
          try {{
            if (window.opener) {{
              window.opener.postMessage({json.dumps(success_payload)}, "*");
              setTimeout(()=>window.close(), 600);
            }}
          }} catch(e) {{}}
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html, status_code=200)

@app.post("/api/linkedin/disconnect")
async def linkedin_disconnect(request: Request):
    """Remove stored LinkedIn linkage for this user."""
    user_id = _validate_user(request)
    removed = bool(USER_LINK_STORE.pop(user_id, None))
    return {"disconnected": removed}

# Diagnostic endpoint (optional)
@app.get("/api/linkedin/debug")
async def linkedin_debug(request: Request):
    user_id = _validate_user(request)
    return {
        "linked": user_id in USER_LINK_STORE,
        "stored_keys": list(USER_LINK_STORE.get(user_id, {}).keys()),
        "states_cached": len(OAUTH_STATE_STORE),
    }

# (Add this endpoint – was missing, causing 404 on /api/linkedin/status)
@app.get("/api/linkedin/status")
async def linkedin_status(request: Request):
    """
    Return LinkedIn connection status for the current user.
    Frontend polls this; previously 404 because route was removed.
    """
    user_id = _validate_user(request)
    data = USER_LINK_STORE.get(user_id)
    if not data:
        return {"connected": False}
    return {"connected": True}
# =============== Draft Endpoints (unchanged logic) ===============

async def auto_generate_cycle(user_id: str):
    """
    Scheduled entrypoint that triggers an auto-generation run for user_id.
    Previously called the toggle endpoint and often hit limits without generating.
    Now performs a single scheduled attempt with clear guards.
    """
    try:
        created = _scheduled_auto_attempt(user_id)
        if created:
            logging.info("[AutoDraft][Scheduled] generated user=%s", user_id)
        else:
            logging.info("[AutoDraft][Scheduled] no-op user=%s (skipped by guards)", user_id)
    except Exception as e:
        logging.error(f"[AutoDraft][Scheduled] auto_generate_cycle_failed user={user_id} err={e}")

def _scheduled_auto_attempt(user_id: str) -> bool:
    """
    Attempt a single scheduled auto-generation for the given user.
    Respects:
      - enabled flag
      - pending auto-generated draft guard
      - per-user posts-per-day limit
    On success: schedules draft publish and the next run slot.
    Returns True if a draft was created, else False.
    """
    st = AUTO_STATE.get(user_id) or {}
    if not st.get("enabled"):
        logging.info("[AutoDraft][Scheduled] disabled user=%s", user_id)
        return False

    # Load and persist lightweight context into ABOUT_CONTEXT_STORE
    user_context = _sanitize_user_context(_get_profile_key_points(user_id))
    if user_context:
        ABOUT_CONTEXT_STORE[user_id] = user_context
    headline = _get_linkedin_headline(user_id)

    # Daily limit guard (per-user configured)
    used_today = count_user_auto_drafts_today(user_id)
    limit = _effective_posts_per_day(user_id)
    if used_today >= limit:
        logging.info("[AutoDraft][Scheduled] skip_limit user=%s used=%s limit=%s", user_id, used_today, limit)
        # Still compute/schedule next slot to keep cadence moving forward
        try:
            _schedule_next_for_user(user_id)
        except Exception as e:
            logging.warning("[AutoDraft][Scheduled] schedule_next_failed user=%s err=%s", user_id, e)
        return False

    # Pending guard
    pending_auto = [
        d for d in get_user_pending_drafts(user_id)
        if getattr(d, "auto_generated", False)
    ]
    if pending_auto:
        logging.info("[AutoDraft][Scheduled] skip_pending user=%s count=%s", user_id, len(pending_auto))
        try:
            _schedule_next_for_user(user_id)
        except Exception as e:
            logging.warning("[AutoDraft][Scheduled] schedule_next_failed user=%s err=%s", user_id, e)
        return False

    # Generator that uses About Me + Drafts Key Points via unified planner
    def gen_with_context(**kw):
        about_me = _get_profile_about_me(user_id)
        key_points = ABOUT_CONTEXT_STORE.get(user_id, user_context)
        return _plan_and_generate_with_template(user_id, about_me, key_points)

    # Try LLM-select template first, then varied, then single fallback
    draft = None
    try:
        selected_template = _llm_pick_template(TEMPLATE_STORE, user_context=user_context, goal="engagement")
        if selected_template:
            logging.info("[AutoDraft][Scheduled] using_llm_selected_template id=%s user=%s", selected_template.id, user_id)
            draft = _safe_create_auto_draft(
                user_id=user_id,
                template=selected_template,
                generator=lambda **kw: gen_with_context(**kw)
            )
    except Exception as e:
        logging.warning("[AutoDraft][Scheduled] llm_template_pick_or_create_failed user=%s err=%s", user_id, e)

    if not draft:
        try:
            logging.info("[AutoDraft][Scheduled] attempting_varied user=%s templates=%s", user_id, len(TEMPLATE_STORE))
            draft = create_varied_auto_draft(
                user_id=user_id,
                templates=TEMPLATE_STORE,
                generator=lambda **kw: gen_with_context(**kw)
            )
        except Exception as e:
            logging.error("[AutoDraft][Scheduled] varied_failed user=%s err=%s (fallback to single template)", user_id, e)
            try:
                tmpl = random.choice(list(TEMPLATE_STORE.values()))
                draft = _safe_create_auto_draft(
                    user_id=user_id,
                    template=tmpl,
                    generator=lambda **kw: gen_with_context(**kw)
                )
            except Exception as e2:
                logging.error("[AutoDraft][Scheduled] fallback_failed user=%s err=%s", user_id, e2)
                # Still move schedule forward
                try:
                    _schedule_next_for_user(user_id)
                except Exception as e3:
                    logging.warning("[AutoDraft][Scheduled] schedule_next_failed user=%s err=%s", user_id, e3)
                return False

    # Normalize status and schedule publish
    if getattr(draft, "status", None) not in ("pending",):
        try:
            draft.status = "pending"
        except Exception:
            pass

    try:
        _schedule_publish(draft)
    except Exception as e:
        logging.error("[AutoDraft][Scheduled] publish_schedule_failed user=%s draft_id=%s err=%s", user_id, getattr(draft, "id", "?"), e)
        # Draft exists but publish wasn't scheduled; still move cadence
        try:
            _schedule_next_for_user(user_id)
        except Exception as e2:
            logging.warning("[AutoDraft][Scheduled] schedule_next_failed user=%s err=%s", user_id, e2)
        return True  # draft created

    # Move cadence forward to next slot after creating the draft
    try:
        _schedule_next_for_user(user_id)
    except Exception as e:
        logging.warning("[AutoDraft][Scheduled] schedule_next_failed user=%s err=%s", user_id, e)

    logging.info("[AutoDraft][Scheduled] success user=%s draft_id=%s publish_at=%s",
                 user_id, draft.id, getattr(draft, "publish_at", None))
    return True

# --- Added: safe wrapper to handle create_auto_draft signature differences ---
def _safe_create_auto_draft(user_id: str, template: Template, generator):
    """
    Tries create_auto_draft with 'template=' then falls back to 'templates=' if the
    imported function expects a plural parameter (avoids TypeError seen in logs).
    """
    try:
        return create_auto_draft(user_id=user_id, template=template, generator=generator)
    except TypeError as e:
        if "unexpected keyword argument 'template'" in str(e):
            try:
                return create_auto_draft(user_id=user_id, templates={template.id: template}, generator=generator)
            except Exception as e2:
                logging.error("[AutoDraft] safe_create_auto_draft_fallback_failed user=%s template=%s err=%s",
                              user_id, getattr(template, "id", "?"), e2)
                raise
        raise

# --- Added: internal LLM template selection helper (replaces missing import) ---
def _llm_pick_template(templates: Dict[str, Template], user_context: str = "", goal: str = "engagement") -> Optional[Template]:
    """
    Choose a template via LLM. Returns Template or None (caller falls back to varied/random).
    """
    if not openai_client or not templates:
        return None
    try:
        catalog_lines = []
        for t in list(templates.values())[:20]:
            name = getattr(t, "name", t.id)
            desc = getattr(t, "description", "") or ""
            snippet = (desc[:90] + ("..." if len(desc) > 90 else ""))
            catalog_lines.append(f"- {t.id}: {name} | {snippet}")
        catalog = "\n".join(catalog_lines)
        prompt = f"""
You are selecting the best LinkedIn post template for generating a high quality draft.

User context: {user_context or "N/A"}
Primary goal: {goal}

Templates:
{catalog}

Respond ONLY with compact JSON: {{"template_id":"<one_of_listed_ids>"}}
If uncertain, choose the most generally useful template.
"""
        raw = openai_client.generate_completion(
            system_message="You help pick a template id.",
            user_message=prompt,
            temperature=0
        ) or ""
        parsed = extract_json_from_text(raw) or {}
        tid = parsed.get("template_id")
        if tid and tid in templates:
            logging.info("[TemplateSelect] LLM chose template_id=%s", tid)
            return templates[tid]
        logging.warning("[TemplateSelect] unusable_response raw=%s", raw[:180])
    except Exception as e:
        logging.warning("[TemplateSelect] exception=%s", e)
    return None

def _get_linkedin_headline(user_id: str) -> Optional[str]:
    record = USER_LINK_STORE.get(user_id)
    if not record or not record.get("access_token"):
        return None
    if record.get("headline"):
        return record["headline"]
    try:
        resp = requests.get(
            "https://api.linkedin.com/v2/me",
            headers={
                "Authorization": f"Bearer {record['access_token']}",
                "X-Restli-Protocol-Version": "2.0.0"
            },
            timeout=10
        )
        if resp.status_code == 200:
            pj = resp.json()
            headline = (
                pj.get("headline")
                or pj.get("localizedHeadline")
                or pj.get("firstName", {}).get("localized", {}).get("en_US")
            )
            if headline:
                record["headline"] = headline
                return headline
    except Exception as e:
        logging.warning(f"[LinkedIn] headline_fetch_failed user={user_id} err={e}")
    return None

# --- Added: shared helper to perform "auto style" generation for both auto & manual ---
def _generate_auto_style(
    user_id: str,
    composed_pd: str,
    treat_as_manual: bool,
    user_context: str = "",
    headline: Optional[str] = None  # was: str | None
) -> Draft:
    """
    Core auto-style generation pipeline (LLM pick -> varied -> fallback).
    treat_as_manual:
        True  -> skip daily auto limit & pending-auto guard.
        False -> enforce normal auto-generation constraints (handled upstream).
    composed_pd: merged personal detail (user_context + headline etc).
    """
    if not TEMPLATE_STORE:
        raise HTTPException(status_code=500, detail="No templates loaded")

    def gen_with_context(**kw):
        kw["personal_detail"] = _inject_context(kw.get("personal_detail", ""), ABOUT_CONTEXT_STORE.get(user_id, effective_ctx))
        return _core_generate(**kw)
    # Override composed_pd with robust injection (ignore passed composed_pd if we have stored context)
    effective_ctx = _sanitize_user_context(user_context) or ABOUT_CONTEXT_STORE.get(user_id, "")
    if effective_ctx and effective_ctx != ABOUT_CONTEXT_STORE.get(user_id):
        ABOUT_CONTEXT_STORE[user_id] = effective_ctx
    injected_pd = _inject_context(
        base_pd=(f"LinkedIn headline: {headline}" if headline else ""),
        user_ctx=ABOUT_CONTEXT_STORE.get(user_id, effective_ctx)
    )
    composed_pd = injected_pd

    def gen_with_context(**kw):
        # Planner uses About Me + Drafts key points
        about_me = _get_profile_about_me(user_id)
        key_points = ABOUT_CONTEXT_STORE.get(user_id, effective_ctx)
        return _plan_and_generate_with_template(user_id, about_me, key_points)

    # Replace prior varied-first logic with planner-driven single attempt + fallback
    draft = None
    try:
        # Use templates= signature to match drafts.create_auto_draft
        draft = create_auto_draft(
            user_id=user_id,
            templates={tid: t for tid, t in TEMPLATE_STORE.items()},
            generator=lambda **kw: gen_with_context(**kw)
        )
    except Exception as e:
        logging.error("[GenPipeline] primary_auto_draft_failed user=%s err=%s", user_id, e)

    if not draft:
        # Last resort manual creation path
        tmpl = random.choice(list(TEMPLATE_STORE.values()))
        draft = create_manual_draft(
            user_id=user_id,
            template=tmpl,
            params={"manual": True},
            generator=lambda **kw: gen_with_context(**kw)
        )

    if getattr(draft, "status", None) not in ("pending",):
        try: draft.status = "pending"
        except Exception: pass
    if treat_as_manual and hasattr(draft, "auto_generated"):
        try: draft.auto_generated = False
        except Exception: pass
    return draft

@app.post("/api/drafts/generate")
async def force_generate_draft(request: Request, body: Dict = Body(default={})):
    # Rewritten to mirror auto-generate flow
    user_id = _validate_user(request)
    # --- changed: use account key_points instead of body.user_context ---
    user_context = _sanitize_user_context(_get_profile_key_points(user_id))
    if user_context:
        ABOUT_CONTEXT_STORE[user_id] = user_context
    headline = _get_linkedin_headline(user_id)
    pd = (body.get("personal_detail") or "").strip()
    # We now ignore manual merging here and let unified pipeline inject context strongly
    composed_pd = ""  # placeholder; pipeline will reconstruct
    draft = _generate_auto_style(
        user_id=user_id,
        composed_pd=composed_pd,
        treat_as_manual=True,
        user_context=user_context,
        headline=headline
    )

    _schedule_publish(draft)
    draft_dict = draft.to_public_dict()
    if "content" not in draft_dict and "post" in draft_dict:
        draft_dict["content"] = draft_dict["post"]
    draft_dict["chosen_template_id"] = getattr(draft, "template_id", getattr(draft, "template", None)) or draft_dict.get("chosen_template_id")
    draft_dict["auto_flow_like"] = True
    draft_dict["about_context_used"] = ABOUT_CONTEXT_STORE.get(user_id, "")
    if draft_dict.get("chosen_template_id"):
        LAST_MANUAL_TEMPLATE[user_id] = str(draft_dict["chosen_template_id"])

    logging.info("[Drafts] manual_auto_style user=%s draft_id=%s template=%s ctx_len=%s",
                 user_id, draft.id, draft_dict.get("chosen_template_id"), len(draft_dict.get("about_context_used") or ""))
    return {"draft": draft_dict}

# --- added: auth + profile endpoints ---
@app.post("/api/auth/register")
async def auth_register(response: Response, body: Dict = Body(...)):
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required")
    if email in USER_DB:
        raise HTTPException(status_code=409, detail="email already registered")
    user_id = f"u_{secrets.token_urlsafe(8)}"
    rec = {
        "user_id": user_id,
        "email": email,
        "pw_hash": _hash_password(password),
        "key_points": "",
        "about_me": "",
        "display_name": ""
    }
    USER_DB[email] = rec
    USER_BY_ID[user_id] = rec
    # --- persist to disk ---
    _persist_users()
    token = _issue_token(user_id)
    # CHANGED: use consistent auth cookie helper (was 'session' cookie with differing options)
    try:
        _set_auth_cookie(response, token)
    except Exception:
        pass
    return {
        "user_id": user_id,
        "email": email,
        "token": token,
        "profile": {"key_points": "", "about_me": "", "display_name": ""}
    }

@app.post("/api/auth/login")
async def auth_login(response: Response, body: Dict = Body(...)):
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    rec = USER_DB.get(email)
    if not rec or rec.get("pw_hash") != _hash_password(password):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = _issue_token(rec["user_id"])
    # CHANGED: use consistent auth cookie helper (one code path for dev/prod)
    try:
        _set_auth_cookie(response, token)
    except Exception:
        pass
    return {
        "user_id": rec["user_id"],
        "email": email,
        "token": token,
        "profile": {
            "key_points": rec.get("key_points") or "",
            "about_me": rec.get("about_me") or "",
            "display_name": rec.get("display_name") or ""
        }
    }

@app.post("/api/auth/logout")
async def auth_logout(request: Request, response: Response):
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
        AUTH_TOKENS.pop(token, None)
    # CHANGED: clear both possible cookie tokens and remove mappings
    try:
        cookie_tok = request.cookies.get("auth_token") or request.cookies.get("session")
        if cookie_tok:
            AUTH_TOKENS.pop(cookie_tok, None)
        _clear_auth_cookie(response)
    except Exception:
        pass
    return {"ok": True}

@app.get("/api/me")
async def me_get(request: Request):
    user_id = _validate_user(request)
    rec = USER_BY_ID.get(user_id) or {}
    st = AUTO_STATE.get(user_id) or {}
    auto_enabled = bool(st.get("enabled", rec.get("auto_generate_enabled", False)))
    # lightweight LinkedIn info if linked
    link = USER_LINK_STORE.get(user_id)
    return {
        "user_id": user_id,
        "email": rec.get("email"),
        "display_name": rec.get("display_name") or "",
        "about_me": rec.get("about_me") or "",
        "key_points": rec.get("key_points") or "",
        "notify_on_draft": bool(rec.get("notify_on_draft", False)),  # <-- added
        "linkedin_connected": bool(link),                              # <-- added
        "linkedin_member_urn": (link or {}).get("member_urn"),         # <-- added
        "linkedin_headline": (link or {}).get("headline"),             # <-- added
        "auto_generate_enabled": auto_enabled,
        "auto_next_run_at": st.get("next_run_at"),
        "auto_posts_per_day": _effective_posts_per_day(user_id),
        "auto_window_start": rec.get("auto_window_start") or None,
        "auto_window_end": rec.get("auto_window_end") or None
    }

@app.post("/api/me")
async def me_update(request: Request, body: Dict = Body(...)):
    user_id = _validate_user(request)
    kp = (body.get("key_points") or "").strip()
    am = (body.get("about_me") or "").strip()
    dn = (body.get("display_name") or "").strip()
    if len(kp) > 2000: kp = kp[:2000]
    if len(am) > 4000: am = am[:4000]
    if len(dn) > 120: dn = dn[:120]
    rec = USER_BY_ID.setdefault(user_id, {"user_id": user_id})
    if "key_points" in body: rec["key_points"] = kp
    if "about_me" in body: rec["about_me"] = am
    if "display_name" in body: rec["display_name"] = dn
    # new: notifications preference
    if "notify_on_draft" in body:
        rec["notify_on_draft"] = bool(body.get("notify_on_draft"))

    ABOUT_CONTEXT_STORE[user_id] = rec.get("key_points") or ""

    # accept per-user auto config
    changed_schedule = False
    if "auto_posts_per_day" in body:
        try:
            ppd = int(body.get("auto_posts_per_day") or 0)
        except Exception:
            ppd = 0
        ppd = max(1, min(5, ppd))  # cap 1..5
        rec["auto_posts_per_day"] = ppd
        changed_schedule = True
    if "auto_window_start" in body:
        rec["auto_window_start"] = (body.get("auto_window_start") or "").strip() or None
        changed_schedule = True
    if "auto_window_end" in body:
        rec["auto_window_end"] = (body.get("auto_window_end") or "").strip() or None
        changed_schedule = True

    # accept auto_generate_enabled via profile save
    if "auto_generate_enabled" in body:
        _set_auto_enabled(user_id, bool(body.get("auto_generate_enabled")))

    _persist_users()

    # If auto is enabled and schedule-affecting fields changed, recompute next slot
    st = AUTO_STATE.get(user_id) or {}
    if changed_schedule and st.get("enabled"):
        try:
            _schedule_next_for_user(user_id)
        except Exception as e:
            logging.warning("[Profile] recompute_next_failed user=%s err=%s", user_id, e)

    st = AUTO_STATE.get(user_id) or {}
    return {"ok": True, "profile": {
        "display_name": rec.get("display_name") or "",
        "about_me": rec.get("about_me") or "",
        "key_points": rec.get("key_points") or "",
        "notify_on_draft": bool(rec.get("notify_on_draft", False)),  # <-- added
        "auto_generate_enabled": bool(st.get("enabled", rec.get("auto_generate_enabled", False))),
        "auto_next_run_at": st.get("next_run_at"),
        "auto_posts_per_day": _effective_posts_per_day(user_id),
        "auto_window_start": rec.get("auto_window_start") or None,
        "auto_window_end": rec.get("auto_window_end") or None
    }}

def _parse_hhmm(s: Optional[str]) -> Optional[int]:
    """
    Parse 'HH:MM' into minutes since midnight (0..1439). Returns None if invalid.
    """
    if not s or not isinstance(s, str): return None
    try:
        parts = s.strip().split(":")
        if len(parts) != 2: return None
        h = int(parts[0]); m = int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h*60 + m
    except Exception:
        return None
    return None

def _effective_posts_per_day(user_id: str) -> int:
    """

    Force 1 post/day regardless of per-user/env settings.
    """
    return 1

def _get_window_minutes(user_id: str) -> Tuple[int, int]:  # was: tuple[int, int]
    """
    Return (start_min, end_min). If invalid or unset, use full-day window (0..1440).
    Support overnight windows (end < start) by treating end + 1440 when computing slots.
    """
    rec = USER_BY_ID.get(user_id) or {}
    s = _parse_hhmm(rec.get("auto_window_start"))
    e = _parse_hhmm(rec.get("auto_window_end"))
    if s is None or e is None:
        return 0, 1440
    if s == e:
        # Degenerate -> treat full day
        return 0, 1440
    return s, e

def _next_slot_within_window(now: datetime.datetime, s_min: int, e_min: int, per_day: int) -> datetime.datetime:
    """
    Compute the next datetime slot within a daily window (minutes since midnight).
    Supports overnight windows where e_min <= s_min and evenly spaces 'per_day' slots.
    Returns the next slot as a naive UTC datetime (consistent with other scheduling code).
    """
    per_day = max(1, int(per_day or 1))

    # Compute window length in minutes, handling overnight wrap
    if e_min <= s_min:
        window_len = (e_min + 1440) - s_min
    else:
        window_len = e_min - s_min

    # Interval between slots (integer minutes)
    interval = max(1, int(round(window_len / per_day)))

    # Start of today's window as datetime
    today = now.date()
    start_dt = datetime.datetime.combine(
        today, datetime.time(hour=(s_min // 60), minute=(s_min % 60))
    )

    # Build candidate slots for today
    slots = [start_dt + datetime.timedelta(minutes=i * interval) for i in range(per_day)]

    # If window wraps past midnight, some slots may fall into next calendar day already;
    # the candidate generation above still produces correct datetimes.
    for slot in slots:
        if slot > now:
            return slot

    # No slot left today -> return first slot tomorrow
    return slots[0] + datetime.timedelta(days=1)

# NEW: preferred-time minutes helper (fallback to 09:00 if unset/invalid)
def _get_preferred_minutes(user_id: str) -> int:
    rec = USER_BY_ID.get(user_id) or {}
    m = _parse_hhmm(rec.get("auto_window_start"))
    return m if m is not None else 9 * 60

def _schedule_next_for_user(user_id: str, base_time: Optional[datetime.datetime] = None) -> datetime.datetime:
    """
    Compute and schedule the next_run_at for a user.
    Daily cadence schedules at preferred time (auto_window_start) ±30 minutes.
    """
    now = base_time or datetime.datetime.utcnow()
    per_day = _effective_posts_per_day(user_id)

    # Daily preferred-time scheduler (±30m jitter)
    if per_day == 1:
        pref_min = _get_preferred_minutes(user_id)
        jitter = random.randint(-30, 30)
        today = now.date()
        dt = datetime.datetime.combine(
            today,
            datetime.time(hour=pref_min // 60, minute=pref_min % 60)
        ) + datetime.timedelta(minutes=jitter) - datetime.timedelta(minutes=60)
        if dt <= now:
            dt = dt + datetime.timedelta(days=1)
        st = AUTO_STATE.setdefault(user_id, {"enabled": False, "last_post_at": None, "next_run_at": None})
        st["next_run_at"] = dt.isoformat()
        _schedule_auto_gen_at(user_id, dt)
        _persist_auto_state()
        return dt

    now = base_time or datetime.datetime.utcnow()
    per_day = _effective_posts_per_day(user_id)
    s_min, e_min = _get_window_minutes(user_id)
    next_run = _next_slot_within_window(now, s_min, e_min, per_day)
    st = AUTO_STATE.setdefault(user_id, {"enabled": False, "last_post_at": None, "next_run_at": None})
    st["next_run_at"] = next_run.isoformat()
    _schedule_auto_gen_at(user_id, next_run)
    _persist_auto_state()
    return next_run

def _set_auto_enabled(user_id: str, enabled: bool) -> Dict[str, Any]:
    """
    Central setter to enable/disable auto-generate, persist state, and (un)schedule jobs.
    Also stores the flag on the user profile (users.json), like About Me.
    """
    AUTO_PREFS[user_id] = enabled
    st = AUTO_STATE.setdefault(user_id, {"enabled": False, "last_post_at": None, "next_run_at": None})
    st["enabled"] = enabled
    if not enabled:
        _unschedule_auto_gen_job(user_id)
        st["next_run_at"] = None
    else:
        # schedule next using per-user posts/day and window
        try:
            _schedule_next_for_user(user_id)
        except Exception as e:
            logging.warning("[AutoGen] schedule_next_for_user_failed user=%s err=%s", user_id, e)
    rec = USER_BY_ID.setdefault(user_id, {"user_id": user_id})
    rec["auto_generate_enabled"] = bool(enabled)
    _persist_auto_state()
    _persist_users()
    return {"enabled": bool(enabled), "next_run_at": st.get("next_run_at")}

def _unschedule_auto_gen_job(user_id: str):
    """Remove scheduled auto-generation job for a user if it exists."""
    if scheduler:
        job_id = f"auto_gen_{user_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

# --- Added: restore /api/drafts/pending endpoint (fix 404) ---
@app.get("/api/drafts/pending")
async def list_pending_drafts(request: Request):
    """
    Return pending drafts. If none, widen to any not posted/cancelled so UI isn't empty.
    ?all=1 -> force widening.
    """
    user_id = _validate_user(request)
    all_flag = request.query_params.get("all") == "1"

    drafts_pending = get_user_pending_drafts(user_id)
    if not drafts_pending or all_flag:
        store = DRAFT_STORE.get(user_id, {})
        widened = [
            d for d in store.values()
            if getattr(d, "status", None) not in ("posted", "cancelled")
        ]
        if not drafts_pending:
            drafts_pending = widened

    def _sort_key(d):
        return getattr(d, "publish_at", None) or getattr(d, "created_at", None) or datetime.datetime.utcnow()

    result = []
    for d in sorted(drafts_pending, key=_sort_key)[:25]:
        pd = d.to_public_dict()
        # Normalize content field for frontend
        if "content" not in pd:
            pd["content"] = pd.get("post") or pd.get("text") or "(no content)"
        if isinstance(pd.get("publish_at"), datetime.datetime):
            pd["publish_at"] = pd["publish_at"].isoformat()
        result.append(pd)

    logging.debug("[Drafts] pending_list user=%s count=%s widened=%s",
                  user_id, len(result), not bool(get_user_pending_drafts(user_id)))
    return {"pending": result}

# --- Added: restore edit / cancel / post-now endpoints (fix 404 on cancel) ---
@app.post("/api/drafts/{draft_id}/edit")
async def edit_draft_endpoint(draft_id: str, request: Request, body: Dict = Body(...)):
    user_id = _validate_user(request)
    content = body.get("content")
    if content is None:
        raise HTTPException(status_code=400, detail="content required")
    d = edit_draft(user_id, draft_id, content)
    if not d:
        raise HTTPException(status_code=404, detail="Draft not found or not editable")
    return {"draft": d.to_public_dict()}

@app.post("/api/drafts/{draft_id}/cancel")
async def cancel_draft_endpoint(draft_id: str, request: Request):
    user_id = _validate_user(request)
    d = cancel_draft(user_id, draft_id)
    if not d:
        raise HTTPException(status_code=404, detail="Draft not found or not cancellable")
    if d.job_id and scheduler and scheduler.get_job(d.job_id):
        scheduler.remove_job(d.job_id)
    return {"draft": d.to_public_dict()}

@app.post("/api/drafts/{draft_id}/post-now")
async def post_now_endpoint(draft_id: str, request: Request):
    user_id = _validate_user(request)
    d = DRAFT_STORE.get(user_id, {}).get(draft_id)
    if not d:
        raise HTTPException(status_code=404, detail="Draft not found")
    if d.status not in ("pending",):
        raise HTTPException(status_code=400, detail="Draft not in pending state")
    if d.job_id and scheduler and scheduler.get_job(d.job_id):
        scheduler.remove_job(d.job_id)
    publish_draft(d, immediate=True)
    return {"draft": d.to_public_dict()}

@app.post("/generate")
async def generate_post(request: Request, req: GenerationRequest):
    """
    Generate a LinkedIn post + DM funnel JSON for the UI.
    """
    if openai_client is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Resolve template (be tolerant of get_template_by_id signature)
    template = TEMPLATE_STORE.get(req.template_id)
    if not template:
        try:
            template = get_template_by_id(req.template_id)  # may be (id)->Template
        except TypeError:
            # Fallback: manual scan
            template = next((t for t in TEMPLATE_STORE.values() if getattr(t, "id", None) == req.template_id), None)
    if not template:
        raise HTTPException(status_code=400, detail="Invalid template_id")

    # --- use About Me (profile) as personal_detail and Drafts Key Points as key_facts ---
    try:
        user_id_for_ctx = _user_from_auth_header(request) or request.headers.get("X-User-Id")
    except Exception:
        user_id_for_ctx = None
    about_me = _get_profile_about_me(user_id_for_ctx) if user_id_for_ctx else ""
    key_points = _get_profile_key_points(user_id_for_ctx) if user_id_for_ctx else ""

    merged_pd = about_me  # Personal Detail comes from profile About Me
    try:
        req_for_prompt = GenerationRequest(**{
            **req.model_dump(),
            "personal_detail": merged_pd,
            "key_facts": key_points  # Key Facts come from Drafts Key Points
        })
    except Exception:
        class _Req:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        req_for_prompt = _Req({**req.model_dump(), "personal_detail": merged_pd, "key_facts": key_points})

    # Build prompt + call model
    user_prompt = build_user_prompt(req_for_prompt, template)
    temperature = getattr(req, "temperature", None)
    if temperature is None:
        temperature = 0.7

    raw_text = openai_client.generate_completion(
        system_message=SYSTEM_MESSAGE,
        user_message=user_prompt,
        temperature=temperature
    ) or ""

    data = extract_json_from_text(raw_text) or {}

    # Normalize/defensive defaults
    def _ensure_dm_flow(tone: str):
        t = (tone or "professional").lower()
        opener = "Appreciated your recent post—great perspective." if t == "professional" else "Loved your latest post—super insightful."
        return {
            "initial_message": data.get("dm_flow", {}).get("initial_message") or f"{opener} Thought this might help—happy to swap notes.",
            "followup_no_reply_1": data.get("dm_flow", {}).get("followup_no_reply_1") or "Quick nudge—sharing a relevant tip I use often. If helpful, I can send more.",
            "followup_no_reply_2": data.get("dm_flow", {}).get("followup_no_reply_2") or "No pressure—dropping this and will leave it with you.",
            "followup_question": data.get("dm_flow", {}).get("followup_question") or "Curious—what’s the bottleneck you see most right now?",
            "qualification_question": data.get("dm_flow", {}).get("qualification_question") or "Would it help to walk through your current approach and spot quick wins?",
            "book_meeting_template": data.get("dm_flow", {}).get("book_meeting_template") or "If useful, here’s a quick link to find a time this week: {{calendly_link}}"
        }

    # Post length cap
    post = str(data.get("post") or "").strip()
    if len(post) > 1300:
        post = post[:1297].rstrip() + "..."
    data["post"] = post or "Quick take: shipping beats perfect. Here’s what actually moved the needle for us."

    # Hooks: ensure 3 max, each <=120 chars
    hooks = data.get("hooks") or []
    if isinstance(hooks, str):
        hooks = [hooks]
    hooks = [str(h)[:120] for h in hooks][:3]
    if len(hooks) < 3:
        defaults = ["Hard-won lessons from the last sprint", "What we changed to get unstuck", "A tiny tweak that changed everything"]
        hooks += defaults[:max(0, 3 - len(hooks))]
   
    data["hooks"] = hooks

    # Hashtags: 3-6, lowercase, no '#'
    tags = data.get("hashtags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags = normalize_hashtags(tags)[:6]
    if len(tags) < 3:
        tags += ["growth", "learning", "startup"][:3 - len(tags)]
    data["hashtags"] = tags

    # tl;dr normalization
    if "tl;dr" not in data:
        if "tl_dr" in data:
            data["tl;dr"] = data.pop("tl_dr")
        elif "tldr" in data:
            data["tl;dr"] = data.pop("tldr")
        else:
            data["tl;dr"] = "Quick summary: key takeaway and one actionable next step."

    # CTA default
    data["cta"] = data.get("cta") or "Share your take below."

    # Image prompt default
    data["image_prompt"] = data.get("image_prompt") or "Clean, minimal abstract background with subtle LinkedIn-blue accents."

    # Follow-up angle
    data["follow_up_angle"] = data.get("follow_up_angle") or "Break down the exact playbook with before/after examples."

    # DM CTA token
    data["dm_cta"] = (data.get("dm_cta") or "GUIDE").strip().upper()[:24]

    # DM Flow
    data["dm_flow"] = _ensure_dm_flow(req.tone)

    # Safety checks (best-effort)
    try:
        run_safety_checks(data.get("post") or "")
    except Exception:
        pass

    return JSONResponse(data, status_code=200)

@app.post("/api/dm/suggest")
async def dm_suggest(request: Request, body: Dict = Body(...)):
    """
    Suggest a concise, respectful LinkedIn DM reply.
    Request JSON:
      {
        "context": "<relationship / purpose / constraints>",
        "inbound": "<their latest message>",
        "history": [{"role":"partner"|"me","content":"..."}],  // optional recent thread
        "tone": "professional" | "friendly" | "candid"          // optional
        "temperature": 0.0-1.0                                   // optional
      }
    """
    if openai_client is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
   
    _ = _validate_user(request)  # require X-User-Id

    context = (body.get("context") or "").strip()
    inbound = (body.get("inbound") or "").strip()
    history = body.get("history") or []
    tone = (body.get("tone") or "professional").strip().lower()
    temperature = float(body.get("temperature") or 0.4)

    if not inbound and not history and not context:
        raise HTTPException(status_code=400, detail="Provide at least inbound message or context")

    # Prepare compact conversation summary
    hist_lines = []
    for h in history[-8:]:
        role = "Them" if (h.get("role") == "partner") else "Me"
        content = str(h.get("content") or "").strip()
        if content:
            # Truncate overly long history lines to keep prompt small
            hist_lines.append(f"{role}: {content[:500]}")

    conversation_block = "\n".join(hist_lines) if hist_lines else "None"

    sys_msg = (
        "You draft concise, respectful LinkedIn DM replies.\n"
        "Requirements: keep it natural, non-pushy, <= 450 characters, "
        "no emojis unless context invites it, politely move toward a clear next step if appropriate, "
        "avoid hard-sell language. Output only the reply text."
    )
    user_msg = f"""
Context (optional):
{context or "None"}

Conversation so far (most recent last):
{conversation_block}

Their latest message:
{inbound or "(empty)"}

Tone preference: {tone}

Task:
Write my reply (single message, <= 450 chars). Avoid over-explaining. Be direct, kind, and clear.
"""

    raw = openai_client.generate_completion(
        system_message=sys_msg,
        user_message=user_msg,
        temperature=temperature
    ) or ""

    suggestion = raw.strip()
    # Soft cap
    if len(suggestion) > 500:
        suggestion = suggestion[:497].rstrip() + "..."

    try:
        run_safety_checks(suggestion)
    except Exception:
        pass

    return {"suggestion": suggestion}

# New: status endpoints for persisted auto-generate toggle
@app.get("/api/drafts/auto-generate-status")
async def auto_generate_status(request: Request):
    user_id = _validate_user(request)
    st = AUTO_STATE.get(user_id) or {}
    enabled = bool(st.get("enabled", USER_BY_ID.get(user_id, {}).get("auto_generate_enabled", AUTO_PREFS.get(user_id, False))))
    # Daily cadence -> report full-day interval
    try:
        per_day = _effective_posts_per_day(user_id)
        if per_day == 1:
            interval_minutes = 1440
        else:
            s_min, e_min = _get_window_minutes(user_id)
            window_len = (e_min + (1440 if e_min <= s_min else 0)) - s_min
            interval_minutes = max(1, int(round(window_len / max(1, per_day))))
    except Exception:
        interval_minutes = 1440
    rec = USER_BY_ID.get(user_id) or {}
    return {
        "enabled": enabled,
        "next_run_at": st.get("next_run_at"),
        "interval_minutes": interval_minutes,
        "posts_per_day": _effective_posts_per_day(user_id),
        "window_start": rec.get("auto_window_start") or None,
        "window_end": rec.get("auto_window_end") or None
    }

@app.post("/api/drafts/auto-generate-status")
async def auto_generate_status_post(request: Request, body: Dict = Body(default={})):
    # Mirror GET for clients preferring POST
    user_id = _validate_user(request)
    st = AUTO_STATE.get(user_id) or {}
    enabled = bool(st.get("enabled", USER_BY_ID.get(user_id, {}).get("auto_generate_enabled", AUTO_PREFS.get(user_id, False))))
    try:
        per_day = _effective_posts_per_day(user_id)
        if per_day == 1:
            interval_minutes = 1440
        else:
            s_min, e_min = _get_window_minutes(user_id)
            window_len = (e_min + (1440 if e_min <= s_min else 0)) - s_min
            interval_minutes = max(1, int(round(window_len / max(1, per_day))))
    except Exception:
        interval_minutes = 1440
    rec = USER_BY_ID.get(user_id) or {}
    return {
        "enabled": enabled,
        "next_run_at": st.get("next_run_at"),
        "interval_minutes": interval_minutes,
        "posts_per_day": _effective_posts_per_day(user_id),
        "window_start": rec.get("auto_window_start") or None,
        "window_end": rec.get("auto_window_end") or None
    }

def _require_admin(request: Request):
    """
    Simple admin gate:
    - If ADMIN_TOKEN is set, require header X-Admin-Token to match.
    - If ADMIN_TOKEN is not set, allow only localhost callers.
    """
    expected = os.getenv("ADMIN_TOKEN") or ""
    tok = request.headers.get("X-Admin-Token") or ""
    if expected:
        if tok != expected:
            raise HTTPException(status_code=403, detail="admin auth required")
        return
    # No token configured -> allow only local
    try:
        host = (request.client.host or "").lower()
    except Exception:
        host = ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="admin auth required (localhost only)")

def _parse_relative_offset(body: Dict[str, Any]) -> Optional[datetime.timedelta]:
    """
    Supports {in_seconds|in_minutes|in_hours}. Returns timedelta or None.
    Priority: seconds > minutes > hours.
    """
    if "in_seconds" in body:
        try: return datetime.timedelta(seconds=int(body.get("in_seconds") or 0))
        except Exception: pass
    if "in_minutes" in body:
        try: return datetime.timedelta(minutes=int(body.get("in_minutes") or 0))
        except Exception: pass
    if "in_hours" in body:
        try: return datetime.timedelta(hours=int(body.get("in_hours") or 0))
        except Exception: pass
    return None

# --- Admin: force-set next_run_at for testing ---
@app.post("/api/admin/auto/next-run")
async def admin_set_next_run(request: Request, body: Dict = Body(...)):
    """
    Set or adjust next_run_at for a user and reschedule the auto-generation job.
    Body:
      {
        "user_id": "<target user id>",
        "next_run_at": "2025-10-07T15:30:00",   // optional ISO-8601 (UTC)
        "in_seconds": 30,                        // optional relative offset (s)
        "in_minutes": 1,                         // optional relative offset (m)
        "in_hours": 0                            // optional relative offset (h)
      }
    If both next_run_at and relative fields are provided, next_run_at takes precedence.
    """
    _require_admin(request)
    user_id = (body.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    # Resolve target datetime
    target_dt: Optional[datetime.datetime] = None
    iso = body.get("next_run_at")
    if isinstance(iso, str) and iso.strip():
        try:
            target_dt = datetime.datetime.fromisoformat(iso.strip())
        except Exception:
            raise HTTPException(status_code=400, detail="invalid next_run_at format (use ISO-8601)")
    else:
        delta = _parse_relative_offset(body)
        if delta is None:
            raise HTTPException(status_code=400, detail="provide next_run_at or a relative offset")
        target_dt = datetime.datetime.utcnow() + delta

    # Ensure naive UTC if a timezone-aware dt sneaks in
    if getattr(target_dt, "tzinfo", None) is not None:
        target_dt = target_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    # Update state and reschedule
    st = AUTO_STATE.setdefault(user_id, {"enabled": True, "last_post_at": None, "next_run_at": None})
    st["next_run_at"] = target_dt.isoformat()
    try:
        _schedule_auto_gen_at(user_id, target_dt)
        _persist_auto_state()
    except Exception as e:
        logging.error("[Admin] schedule_next_failed user=%s err=%s", user_id, e)
        raise HTTPException(status_code=500, detail="failed to schedule")

    return {
        "ok": True,
        "user_id": user_id,
        "enabled": bool(st.get("enabled", True)),
        "next_run_at": st["next_run_at"]
    }

# --- Admin: inspect auto state (optional filter by user_id) ---
@app.get("/api/admin/auto/state")
async def admin_get_auto_state(request: Request, user_id: Optional[str] = None):
    _require_admin(request)
    if user_id:
        st = AUTO_STATE.get(user_id) or {}
        return {"user_id": user_id, "state": st}
    # shallow copy to avoid mutation races
    return {"auto_state": {uid: dict(st) for uid, st in AUTO_STATE.items()}}
@app.post("/api/drafts/auto-generate-toggle")
async def auto_generate_toggle(request: Request, body: Dict = Body(...)):
    """
    Toggle auto-generate drafts on/off for the current user.
    Request JSON: { "enabled": true|false, "generate_now": false }  // generate_now optional
    Response: { enabled, next_run_at, generated_draft_id?, notice? }
    """
    user_id = _validate_user(request)
    enabled = bool(body.get("enabled"))
    generate_now = bool(body.get("generate_now", False))

    # Flip state and schedule/unschedule
    _set_auto_enabled(user_id, enabled)
    st = AUTO_STATE.get(user_id) or {}
    next_run_at = st.get("next_run_at")

    # Ensure a scheduled time exists when enabling
    if enabled and not next_run_at:
        try:
            next_run_at = _schedule_next_for_user(user_id).isoformat()
        except Exception:
            next_run_at = None

    generated_id = None
    notice = None

    # Optional immediate generation attempt (guards enforced inside)
    if enabled and generate_now:
        try:
            if _scheduled_auto_attempt(user_id):
                store = DRAFT_STORE.get(user_id, {})
                if store:
                    d = sorted(
                        store.values(),
                        key=lambda x: getattr(x, "created_at", getattr(x, "publish_at", datetime.datetime.utcnow())),
                        reverse=True
                    )[0]
                    generated_id = getattr(d, "id", None)
            else:
                notice = "Auto enabled; generation skipped by guards."
        except Exception as e:
            logging.warning("[AutoToggle] generate_now_failed user=%s err=%s", user_id, e)
            notice = "Auto enabled; immediate generation failed."

    if not notice:
        notice = "Auto-generate enabled. Next run: {}".format(next_run_at or "(scheduled)") if enabled else "Auto-generate disabled."

    return {
        "enabled": enabled,
        "next_run_at": next_run_at,
        "generated_draft_id": generated_id,
        "notice": notice
    }
# --- end toggle endpoint ---