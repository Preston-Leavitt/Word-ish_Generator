import pytest
from unittest.mock import Mock, patch
import json

from app.templates import load_templates
from app.prompts import build_user_prompt
from app.safety import run_safety_checks, check_profanity, check_pii
from app.utils import normalize_hashtags, extract_json_from_text
from app.schemas import GenerationRequest, GenerationResponse, Template


def test_template_loading():
    """Test template loading."""
    templates = load_templates()
    assert len(templates) > 0
    assert "you_dont_actually_want_that" in templates


def test_prompt_building():
    """Test prompt building."""
    request = GenerationRequest(
        template_id="you_dont_actually_want_that",
        tone="candid",
        audience="early-stage SaaS founders",
        goal="leadgen",
        key_facts="Grew newsletter to 10k in 9 months",
        personal_detail="I was the only salesperson at my first startup",
        temperature=0.35
    )
    
    template = Template(
        id="you_dont_actually_want_that",
        name="You Don't Actually Want That",
        structure="Hook + Problem + Solution + CTA",
        rules="Start with contrarian hook"
    )
    
    prompt = build_user_prompt(request, template)
    assert "candid" in prompt
    assert "early-stage SaaS founders" in prompt
    assert "10k in 9 months" in prompt


def test_safety_checks():
    """Test safety checks."""
    # Test profanity detection
    issues = check_profanity("This is stupid content")
    assert len(issues) > 0
    
    # Test PII detection
    issues = check_pii("Contact me at test@example.com")
    assert len(issues) > 0
    
    # Test clean content
    issues = run_safety_checks("Great content here", ["Hook 1", "Hook 2", "Hook 3"])
    assert len(issues) == 0


def test_hashtag_normalization():
    """Test hashtag normalization."""
    hashtags = ["#B2BMarketing", "SaaS!", "@growth"]
    normalized = normalize_hashtags(hashtags)
    assert normalized == ["b2bmarketing", "saas", "growth"]


def test_json_extraction():
    """Test JSON extraction from text."""
    text_with_json = 'Here is some text {"key": "value", "number": 123} and more text'
    extracted = extract_json_from_text(text_with_json)
    assert extracted == {"key": "value", "number": 123}


@patch('app.openai_client.OpenAI')
def test_mocked_openai_response(mock_openai):
    """Test with mocked OpenAI response."""
    # Mock response
    mock_response_data = {
        "post": "Stop chasing virality. Make your content a heat-seeking missile.\n\nWhen I started I chased trends and got impressions with zero customers. I swapped to targeted content and cold outreach. In 9 months my newsletter grew to 10k and demo bookings doubled.\n\nAim for clarity, not virality: niche your message, speak their language, and send content directly to them.\n\nDownload the free playbook (link in comments).",
        "hooks": [
            "Stop chasing virality. Make your content a heat-seeking missile.",
            "Why viral content usually fails SaaS founders.",
            "Small audience, huge results: the anti-viral playbook."
        ],
        "hashtags": ["b2bmarketing", "saas", "contentstrategy", "growth"],
        "image_prompt": "close-up of a missile target overlaying a LinkedIn post UI; warm cinematic lighting; clean modern style",
        "tl;dr": "Target your audience, not the algorithm.",
        "cta": "Download the free playbook (link in comments).",
        "follow_up_angle": "Share a 3-part thread showing the exact calendar and outreach script you used."
    }
    
    # Test response validation
    response = GenerationResponse(**mock_response_data)
    assert response.post is not None
    assert len(response.hooks) == 3
    assert len(response.hashtags) >= 3
    assert response.tl_dr is not None


if __name__ == "__main__":
    pytest.main([__file__])
