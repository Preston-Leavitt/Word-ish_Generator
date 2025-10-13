import os
import time
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            # Fallback initialization without extra parameters
            self.client = OpenAI()
        
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # --- added: formatter gate config ---
        self.formatter_enabled = (os.getenv("OPENAI_FORMATTER_ENABLED", "1") == "1")
        self.format_model = os.getenv("OPENAI_FORMATTER_MODEL", "gpt-4o-mini")
        # --- end addition ---
    
    def generate_completion(
        self, 
        system_message: str, 
        user_message: str, 
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> Optional[str]:
        """Generate completion with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=2000,
                    timeout=30
                )
                
                raw = response.choices[0].message.content.strip()
                # --- added: optional post-formatting gate ---
                if self.formatter_enabled and self._should_format(system_message, user_message, raw):
                    try:
                        formatted = self.format_generation_json(raw)
                        if formatted:
                            return formatted.strip()
                    except Exception as fe:
                        print(f"Formatter gate failed, returning raw output: {fe}")
                # --- end addition ---
                return raw
                
            except Exception as e:
                print(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
        
        return None

    # --- added: helper to decide when to run formatter ---
    def _should_format(self, system_message: str, user_message: str, raw_text: str) -> bool:
        """
        Only format JSON generations for LinkedIn posts:
        - System message matches our post generator signature, and
        - The output contains a JSON-like 'post' field.
        Avoids touching DM suggest/planner outputs.
        """
        sig = (system_message or "").lower()
        looks_like_post_json = '"post"' in (raw_text or "")
        is_post_generator = ("viral post generator" in sig) or ("exact keys" in sig and "dm_flow" in sig)
        return is_post_generator and looks_like_post_json

    # --- added: second-pass formatting using gpt-4o-mini (or env override) ---
    def format_generation_json(self, raw_json_text: str, temperature: float = 0.2) -> str:
        print(raw_json_text)
        """
        Rewrites ONLY the 'post' field to enforce the required structure and tone:
        Structure:
        - Hook: 1 line, <90 chars, contrarian statement challenging conventional wisdom
        - Opening: 2–3 lines expanding the hook with a relatable pain point or common mistake
        - Main Body: numbered list (3–5) OR bullet points; each point 1–2 sentences; include concrete examples; add whitespace
        - Social Proof/Results: brief outcome or data point validating the approach
        - Closing: statement (NOT a question) reinforcing the main point and inviting reflection

        Style:
        - Length 80–120 words total
        - Direct, no‑nonsense, CEO‑speak; short sentences; minimal punctuation
        - No bold; 2–3 line breaks for separation; lists for scannability; max 1–2 emojis (optional)
        - Keep total post <= 1300 chars

        Do not modify any keys/values other than 'post'. Do not add hashtags in 'post'.
        Return strict minified JSON with identical keys.
        """
        sys = (
            "You are a strict JSON formatter. Keep the exact JSON shape and keys. "
            "Only rewrite the 'post' field and enforce this structure and tone:\n"
            "STRUCTURE:\n"
            "1) Hook (1 line, <90 chars): contrarian statement challenging conventional wisdom about the topic.\n"
            "2) Opening (2–3 lines): expand the hook with a relatable pain point or common mistake.\n"
            "3) Main Body: numbered list (3–5 points) OR bullet points; each point 1–2 sentences; include specific, concrete examples; add whitespace between sections.\n"
            "4) Social Proof/Results: brief outcome or data point that validates the approach.\n"
            "5) Closing: end with a statement (NOT a question) that reinforces the main point and invites reflection.\n"
            "\n"
            "STYLE:\n"
            "- 80–120 words total; direct, no‑nonsense, CEO‑speak; short sentences; minimal punctuation.\n"
            "- No bold text; 2–3 line breaks for visual separation; lists for scannability; max 1–2 emojis if used at all.\n"
            "- Keep 'post' <= 1300 characters.\n"
            "\n"
            "RULES:\n"
            "- Do NOT modify hooks, hashtags, tl;dr, cta, follow_up_angle, dm_cta, dm_flow, or any keys/values other than 'post'.\n"
            "- Do NOT add hashtags inside 'post'.\n"
            "- IF NO HASHTAGS are present, you have add 1-3 at the end of 'post' on a new line.\n"
            "- Avoid AI tells (formulaic transitions, em dashes, excessive hyphenation).\n"
            "Return STRICT minified JSON only (no commentary)."
        )
        usr = f"""INPUT_JSON:
{raw_json_text}

Task: Return the same JSON, but with 'post' rewritten to meet the structure, style, and rules above. Output STRICT minified JSON only."""
        resp = self.client.chat.completions.create(
            model=self.format_model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ],
            temperature=temperature,
            max_tokens=1200,
            timeout=30
        )
        return resp.choices[0].message.content or ""
