import json
import re
from typing import List, Optional


def normalize_hashtags(hashtags: List[str]) -> List[str]:
    """Normalize hashtags to lowercase, remove punctuation."""
    normalized = []
    for tag in hashtags:
        # Remove # if present, convert to lowercase, remove punctuation
        clean_tag = re.sub(r'[^\w]', '', tag.lstrip('#').lower())
        if clean_tag:
            normalized.append(clean_tag)
    return normalized


def count_characters(text: str) -> int:
    """Count characters in text."""
    return len(text)


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON from text that might contain other content."""
    try:
        # First try to parse as pure JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Look for JSON block in text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    return None


def validate_post_structure(post: str) -> bool:
    """Validate post has proper structure (2-6 paragraphs)."""
    paragraphs = [p.strip() for p in post.split('\n\n') if p.strip()]
    return 2 <= len(paragraphs) <= 6
