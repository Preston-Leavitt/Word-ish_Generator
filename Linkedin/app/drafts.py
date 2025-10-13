from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import uuid
from typing import Dict, List, Callable, Optional
from .schemas import Template
from .linkedin_helpers import safe_publish_to_linkedin  # NEW
import random

# In-memory stores (replace with DB later)
DRAFT_STORE: Dict[str, Dict[str, "Draft"]] = {}
AUTO_PREFS: Dict[str, bool] = {}

@dataclass
class Draft:
    id: str
    user_id: str
    content: str
    title: str
    created_at: datetime
    publish_at: datetime
    last_edited_at: datetime
    status: str  # pending | posted | cancelled
    auto_generated: bool
    job_id: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    meta: Dict[str, Optional[str]] = field(default_factory=dict)  # NEW: arbitrary metadata (e.g., linkedin_post_urn)

    def to_public_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "publish_at": self.publish_at.isoformat(),
            "last_edited_at": self.last_edited_at.isoformat(),
            "status": self.status,
            "auto_generated": self.auto_generated,
            "job_id": self.job_id,
            "meta": self.meta,  # include meta for frontend LinkedIn status
        }

def _now():
    return datetime.now(timezone.utc)

def _derive_title(content: str) -> str:
    first_line = (content.splitlines() or [""]).pop(0).strip()
    return first_line[:70] or "Draft"

def create_auto_draft(user_id: str, templates: Dict[str, Template], generator: Callable[..., Dict]) -> Draft:
    template = next(iter(templates.values())) if templates else Template(id="default", name="Default", structure="", rules="")
    generated = generator(template=template)
    draft = Draft(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content=generated["post"],
        title=_derive_title(generated["post"]),
        created_at=_now(),
        publish_at=_now() + timedelta(hours=1),
        last_edited_at=_now(),
        status="pending",
        auto_generated=True,
        logs=[f"{_now().isoformat()} created (auto)"]
    )
    DRAFT_STORE.setdefault(user_id, {})[draft.id] = draft
    return draft

def create_manual_draft(user_id: str, template: Template, params: Dict, generator: Callable[..., Dict]) -> Draft:
    generated = generator(
        template=template,
        tone=params.get("tone", "candid"),
        audience=params.get("audience", "founders"),
        goal=params.get("goal", "engagement"),
        key_facts=params.get("key_facts", ""),
        personal_detail=params.get("personal_detail", ""),
        temperature=params.get("temperature", 0.6),
    )
    draft = Draft(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content=generated["post"],
        title=_derive_title(generated["post"]),
        created_at=_now(),
        publish_at=_now() + timedelta(hours=1),
        last_edited_at=_now(),
        status="pending",
        auto_generated=False,
        logs=[f"{_now().isoformat()} created (manual)"]
    )
    DRAFT_STORE.setdefault(user_id, {})[draft.id] = draft
    return draft

def publish_draft(draft: Draft, immediate: bool = False):
    if draft.status != "pending":
        return
    # Attempt LinkedIn publish (non-blocking)
    try:
        if "linkedin_post_urn" not in draft.meta:
            urn = safe_publish_to_linkedin(draft.user_id, draft.content)
            if urn:
                draft.meta["linkedin_post_urn"] = urn
    except Exception:
        # Logged inside helper; continue
        pass
    draft.status = "posted"
    draft.logs.append(f"{_now().isoformat()} published{' (immediate)' if immediate else ''}")

def cancel_draft(user_id: str, draft_id: str) -> Optional[Draft]:
    draft = DRAFT_STORE.get(user_id, {}).get(draft_id)
    if not draft or draft.status != "pending":
        return None
    draft.status = "cancelled"
    draft.logs.append(f"{_now().isoformat()} cancelled")
    return draft

def edit_draft(user_id: str, draft_id: str, new_content: str) -> Optional[Draft]:
    draft = DRAFT_STORE.get(user_id, {}).get(draft_id)
    if not draft or draft.status != "pending":
        return None
    draft.content = new_content
    draft.title = _derive_title(new_content)
    draft.last_edited_at = _now()
    draft.logs.append(f"{_now().isoformat()} edited")
    return draft

def get_user_pending_drafts(user_id: str) -> List[Draft]:
    return [d for d in DRAFT_STORE.get(user_id, {}).values() if d.status == "pending"]

def count_user_auto_drafts_today(user_id: str) -> int:
    today = _now().date()
    drafts = DRAFT_STORE.get(user_id, {}).values()
    return sum(1 for d in drafts if d.auto_generated and d.created_at.date() == today)

TONES = ["candid","professional","conversational","authoritative","humorous"]
GOALS = ["leadgen","engagement","awareness","thought-leadership"]
AUDIENCE_SEEDS = [
    "bootstrapped SaaS founders",
    "technical CTOs scaling product-market fit",
    "agency owners stuck at 30k MRR",
    "product-led growth marketers",
    "revops leaders in B2B SaaS"
]
KEY_FACT_SEEDS = [
    "Cut churn 22% in 60 days",
    "Scaled from 0 to 10k users with zero ad spend",
    "Rebuilt onboarding and doubled activation",
    "Reduced CAC payback from 14 to 6 months",
    "Outperformed bigger competitors with a 2-person team"
]
PERSONAL_DETAIL_SEEDS = [
    "Former engineer who hated writing",
    "First sales hire at a failing startup",
    "Solo founder juggling support and growth",
    "Quit agency life to build a product",
    "Built first MVP in a weekend hackathon"
]

def create_varied_auto_draft(user_id: str, templates: Dict[str, Template], generator: Callable[..., Dict]) -> Draft:
    """Create a more diverse auto-generated draft with random structure & angle."""
    template = random.choice(list(templates.values())) if templates else Template(id="default", name="Default", structure="", rules="")
    tone = random.choice(TONES)
    goal = random.choice(GOALS)
    audience = random.choice(AUDIENCE_SEEDS)
    key_facts = random.choice(KEY_FACT_SEEDS)
    personal_detail = random.choice(PERSONAL_DETAIL_SEEDS)
    generated = generator(
        template=template,
        tone=tone,
        audience=audience,
        goal=goal,
        key_facts=key_facts,
        personal_detail=personal_detail,
        temperature=round(random.uniform(0.55, 0.85), 2)
    )
    draft = Draft(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content=generated["post"],
        title=_derive_title(generated["post"]),
        created_at=_now(),
        publish_at=_now() + timedelta(hours=1),
        last_edited_at=_now(),
        status="pending",
        auto_generated=True,
        logs=[f"{_now().isoformat()} created (auto-varied)"],
        meta={}
    )
    DRAFT_STORE.setdefault(user_id, {})[draft.id] = draft
    return draft
