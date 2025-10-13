import re
from typing import List

PROFANITY_LIST = [
    "damn", "hell", "crap", "stupid", "idiot", "moron", "jerk", "suck",
    "hate", "kill", "die", "death", "murder", "violence"
]

NAMED_PERSONS = [
    "elon musk", "jeff bezos", "bill gates", "mark zuckerberg", "tim cook",
    "sundar pichai", "satya nadella", "jack dorsey", "reed hastings"
]

def check_profanity(text: str) -> List[str]:
    """Check for profanity in text."""
    issues = []
    text_lower = text.lower()
    found_profanity = [word for word in PROFANITY_LIST if word in text_lower]
    if found_profanity:
        issues.append(f"Contains profanity: {', '.join(found_profanity)}")
    return issues


def check_pii(text: str) -> List[str]:
    """Check for personally identifiable information."""
    issues = []
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, text):
        issues.append("Contains email address")
    
    # Phone pattern (various formats)
    phone_pattern = r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
    if re.search(phone_pattern, text):
        issues.append("Contains phone number")
    
    return issues


def check_named_persons(text: str) -> List[str]:
    """Check for potential defamation of named living persons."""
    issues = []
    text_lower = text.lower()
    
    for person in NAMED_PERSONS:
        if person in text_lower:
            # Check for negative context
            negative_words = ["fraud", "scam", "lie", "liar", "criminal", "illegal", "steal"]
            for word in negative_words:
                if word in text_lower:
                    issues.append(f"Potential defamation of {person}")
                    break
    
    return issues


def check_length(text: str, max_length: int = 1300) -> List[str]:
    """Check if text exceeds maximum length."""
    issues = []
    if len(text) > max_length:
        issues.append(f"Post exceeds maximum length ({len(text)} > {max_length})")
    return issues


def check_hook_length(hook: str, max_length: int = 120) -> List[str]:
    """Check if hook exceeds maximum length."""
    issues = []
    if len(hook) > max_length:
        issues.append(f"Hook exceeds maximum length ({len(hook)} > {max_length})")
    return issues


def run_safety_checks(post: str, hooks: List[str]) -> List[str]:
    """Run all safety checks on post content."""
    all_issues = []
    
    # Check post content
    all_issues.extend(check_profanity(post))
    all_issues.extend(check_pii(post))
    all_issues.extend(check_named_persons(post))
    all_issues.extend(check_length(post))
    
    # Check hooks
    for i, hook in enumerate(hooks):
        hook_issues = check_hook_length(hook)
        all_issues.extend([f"Hook {i+1}: {issue}" for issue in hook_issues])
    
    return all_issues
