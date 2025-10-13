import os, time, json, logging, urllib.parse, requests
from typing import Optional, Dict, Any

logger = logging.getLogger("linkedin")

# In‑memory stores (prototype) - TODO: move to encrypted persistent storage
USER_LINK_STORE: Dict[str, Dict[str, Any]] = {}
OAUTH_STATE_STORE: Dict[str, str] = {}

# Environment (all optional except client id/secret for real flow)
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID", "")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "")
LINKEDIN_SCOPES = os.getenv("LINKEDIN_SCOPES", "r_liteprofile w_member_social")

AUTH_BASE = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
PROFILE_URL = "https://api.linkedin.com/v2/userinfo"
UGC_POST_URL = "https://api.linkedin.com/v2/ugcPosts"

def build_authorize_url(state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": LINKEDIN_CLIENT_ID,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "scope": LINKEDIN_SCOPES,
        "state": state,
    }
    return f"{AUTH_BASE}?{urllib.parse.urlencode(params)}"

def exchange_code_for_token(code: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": LINKEDIN_REDIRECT_URI,
                "client_id": LINKEDIN_CLIENT_ID,
                "client_secret": LINKEDIN_CLIENT_SECRET,
            },
            timeout=15
        )
        if resp.status_code != 200:
            logger.error(f"LinkedIn token exchange failed: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        expires_in = data.get("expires_in", 0)
        return {
            "access_token": data.get("access_token"),
            "expires_at": int(time.time()) + int(expires_in),
        }
    except Exception as e:
        logger.exception(f"Token exchange exception: {e}")
        return None

def fetch_linkedin_profile(access_token: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(
            PROFILE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15
        )
        if resp.status_code != 200:
            logger.error(f"LinkedIn profile fetch failed: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        pid = data.get("id")
        if not pid:
            return None
        data["member_urn"] = f"urn:li:person:{pid}"
        return data
    except Exception as e:
        logger.exception(f"Profile fetch exception: {e}")
        return None

def linkedin_publish_text(access_token: str, author_urn: str, text: str) -> Optional[str]:
    """Publish text to LinkedIn UGC API, return post URN or None. Non-blocking failure."""
    body = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text[:1300]},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
    }
    try:
        resp = requests.post(
            UGC_POST_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0"
            },
            data=json.dumps(body),
            timeout=20
        )
        if resp.status_code not in (201, 200):
            logger.error(f"LinkedIn publish failed: {resp.status_code} {resp.text}")
            return None
        # Location header or JSON id
        urn = resp.headers.get("x-restli-id") or resp.json().get("id")
        return urn
    except Exception as e:
        logger.exception(f"LinkedIn publish exception: {e}")
        return None

def user_linked(user_id: str) -> bool:
    data = USER_LINK_STORE.get(user_id)
    if not data:
        return False
    if data.get("expires_at", 0) <= int(time.time()):
        # Expired token (no refresh implemented)
        return False
    return True

def safe_publish_to_linkedin(user_id: str, content: str) -> Optional[str]:
    """Attempt LinkedIn publish if user linked; never raises."""
    try:
        record = USER_LINK_STORE.get(user_id)
        if not record:
            return None
        if record.get("expires_at", 0) <= int(time.time()):
            return None
        urn = linkedin_publish_text(record["access_token"], record["member_urn"], content)
        return urn
    except Exception as e:
        logger.exception(f"safe_publish_to_linkedin error: {e}")
        return None
