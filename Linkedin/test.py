"""
Manual LinkedIn OAuth Integration Helper (FastAPI / Flask external client)

Purpose:
Interactively test LinkedIn OAuth and (optionally) post a UGC text update.

Flow:
1. Build and display the correct authorization URL (open it in a browser).
2. User pastes the full redirected callback URL (?code=... or ?error=...).
3. Exchange code for access token.
4. Fetch profile (userinfo if openid/profile scope; otherwise /me).
5. Compute member_urn.
6. (Optional) Publish a text-only UGC post.
7. Output copy/paste-ready credentials for your app's USER_LINK_STORE / DB.

Constraints:
- Do NOT import project/server code.
- Only uses: requests, python-dotenv, urllib.parse, time, json, sys, os, secrets (std lib).
- Minimal error handling; network calls wrapped in try/except.
"""

import os
import sys
import time
import json
import secrets
from urllib.parse import urlencode, urlparse, parse_qs
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------
load_dotenv()

CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "http://localhost:8000/api/linkedin/callback")
SCOPES = os.getenv("LINKEDIN_SCOPES", "r_liteprofile w_member_social")

AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
PROFILE_URL_ME = "https://api.linkedin.com/v2/me"
PROFILE_URL_USERINFO = "https://api.linkedin.com/v2/userinfo"
UGC_POST_URL = "https://api.linkedin.com/v2/ugcPosts"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def exit_error(msg: str):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)

def build_authorize_url(state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state
    }
    return f"{AUTH_URL}?{urlencode(params)}"

def exchange_code_for_token(code: str):
    print("\n[INFO] Exchanging authorization code for access token...")
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET
            },
            timeout=20
        )
    except Exception as e:
        exit_error(f"Network error during token exchange: {e}")
    if resp.status_code != 200:
        exit_error(f"Token exchange failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    if "access_token" not in data:
        exit_error(f"No 'access_token' in token response: {data}")
    return data

def fetch_profile(access_token: str, scopes: str):
    # Decide which endpoint to call
    use_userinfo = any(s in scopes.split() for s in ("openid", "profile"))
    url = PROFILE_URL_USERINFO if use_userinfo else PROFILE_URL_ME
    print(f"\n[INFO] Fetching profile from: {url}")
    headers = {"Authorization": f"Bearer {access_token}"}
    if not use_userinfo:
        headers["X-Restli-Protocol-Version"] = "2.0.0"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        exit_error(f"Network error fetching profile: {e}")
    if resp.status_code != 200:
        exit_error(f"Profile fetch failed ({resp.status_code}): {resp.text}")
    return resp.json(), use_userinfo

def maybe_publish_post(access_token: str, member_urn: str):
    print("\n[OPTIONAL] Publish a UGC text post.")
    choice = input("Publish test post? (y/N): ").strip().lower()
    if choice != "y":
        print("[INFO] Skipping UGC post.")
        return None

    msg = input("Enter post text (default: 'Manual integration test via API'): ").strip()
    if not msg:
        msg = "Manual integration test via API"
    msg = msg[:1300]  # LinkedIn typical text limit

    payload = {
        "author": member_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": msg},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "CONNECTIONS"
        }
    }

    print("[INFO] Sending UGC post request...")
    try:
        resp = requests.post(
            UGC_POST_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0"
            },
            data=json.dumps(payload),
            timeout=25
        )
    except Exception as e:
        print(f"[ERROR] Network error creating post: {e}")
        return None

    if resp.status_code not in (200, 201):
        print(f"[ERROR] UGC post failed ({resp.status_code}): {resp.text}")
        return None

    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    urn = resp.headers.get("x-restli-id") or body.get("id")
    print(f"[SUCCESS] Post created. URN: {urn}")
    print("[REMINDER] Manually delete this test post from your LinkedIn feed if desired.")
    return {"urn": urn, "response": body}

# ---------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------
def main():
    print("=== LinkedIn OAuth Manual Integration Helper ===\n")
    if not CLIENT_ID or not CLIENT_SECRET:
        exit_error("Missing LINKEDIN_CLIENT_ID or LINKEDIN_CLIENT_SECRET in .env")

    print("[INFO] Environment configuration:")
    print(f"  CLIENT_ID (truncated): {CLIENT_ID[:6]}...")
    print(f"  REDIRECT_URI: {REDIRECT_URI}")
    print(f"  SCOPES: {SCOPES}")

    state = f"manual_test:{secrets.token_urlsafe(10)}"
    auth_url = build_authorize_url(state)

    print("\n[ACTION] STEP 1: Open the authorization URL in a browser:")
    print(auth_url)
    print("\nAfter authorizing, LinkedIn will redirect your browser to:")
    print(f"  {REDIRECT_URI}")
    print("Copy the FULL redirected URL (with ?code=... or ?error=...) and paste it below.\n")

    # Allow one retry if initial parse is invalid
    attempts = 0
    parsed_qs = None
    raw_input_url = None
    while attempts < 2:
        raw_input_url = input("Paste full callback URL: ").strip()
        if not raw_input_url:
            print("[WARN] Empty input.")
            attempts += 1
            continue
        pr = urlparse(raw_input_url)
        parsed_qs = parse_qs(pr.query)
        # Accept if we see code or error or state param
        if any(k in parsed_qs for k in ("code", "error", "state")):
            break
        print("[WARN] URL did not contain expected query parameters. Try again.")
        attempts += 1

    if not parsed_qs:
        exit_error("Failed to parse a valid callback URL.")

    code = (parsed_qs.get("code") or [None])[0]
    ret_state = (parsed_qs.get("state") or [None])[0]
    err = (parsed_qs.get("error") or [None])[0]
    err_desc = (parsed_qs.get("error_description") or [None])[0]

    print("\n[INFO] Extracted query parameters:")
    print(f"  code: {code}")
    print(f"  state: {ret_state}")
    print(f"  error: {err}")
    print(f"  error_description: {err_desc}")

    if err:
        exit_error(f"Authorization error: {err} | {err_desc or 'No description'}")

    if not code:
        exit_error("No authorization code found in callback URL.")

    if ret_state != state:
        print(f"[WARNING] State mismatch: expected={state} got={ret_state}")

    # Token exchange
    token_json = exchange_code_for_token(code)
    access_token = token_json["access_token"]
    expires_in = int(token_json.get("expires_in", 0) or 0)
    expires_at = int(time.time()) + expires_in if expires_in else None

    print("\n[TOKEN RESPONSE JSON]")
    print(json.dumps(token_json, indent=2))
    if expires_at:
        print(f"\nComputed expires_at (epoch): {expires_at}")
    else:
        print("\nNo expires_in provided; cannot compute expires_at.")

    # Fetch profile
    profile_json, used_userinfo = fetch_profile(access_token, SCOPES)
    print("\n[PROFILE RESPONSE JSON]")
    print(json.dumps(profile_json, indent=2))

    profile_id = profile_json.get("id")
    if not profile_id:
        # userinfo might return 'sub' instead of 'id'
        profile_id = profile_json.get("sub")
    if not profile_id:
        exit_error("Profile response missing 'id' or 'sub'; cannot derive member_urn.")

    member_urn = f"urn:li:person:{profile_id}"
    print(f"\nComputed member_urn: {member_urn}")

    # Optional UGC
    post_result = maybe_publish_post(access_token, member_urn)

    print("\n=== COPY / PASTE INTO YOUR APP (USER_LINK_STORE / DB) ===")
    summary = {
        "access_token": access_token,
        "expires_at": expires_at,
        "id": profile_id,
        "member_urn": member_urn
    }
    print(json.dumps(summary, indent=2))

    if post_result and post_result.get("urn"):
        print(f"\n[NOTE] A test post was created with URN: {post_result['urn']}")
        print("You may delete it manually from LinkedIn if this was only for testing.")

    print("\n[SECURITY REMINDER]")
    print("Access tokens are sensitive. Store them securely (encrypted at rest),")
    print("limit exposure, implement rotation/refresh, and never commit them to source control.")
    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Aborted by user.")
