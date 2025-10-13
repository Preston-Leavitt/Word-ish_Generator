from .schemas import Template, GenerationRequest

SYSTEM_MESSAGE = """You are a LinkedIn viral post generator. Return valid JSON ONLY with these exact keys: post, hooks, hashtags, image_prompt, tl;dr, cta, follow_up_angle, dm_cta, dm_flow.

Output contract:
- hooks: array of 3 alternative hooks; EACH is a 1‑line contrarian statement (<90 chars) that challenges conventional wisdom about the topic
- post: a single post that strictly follows the structure below and totals 80–120 words
- hashtags: 3–6 lowercase hashtags (no #, no punctuation)
- image_prompt: concise visual description
- tl;dr: one‑sentence summary
- cta: a clear call‑to‑action (short)
- follow_up_angle: suggestion for a natural follow‑up post
- dm_cta: single token/phrase (e.g., "PLAYBOOK", "GUIDE", "TEMPLATE")
- dm_flow: object with keys: initial_message, followup_no_reply_1, followup_no_reply_2, followup_question, qualification_question, book_meeting_template

Post structure requirements (must match):
1) Hook (1 line, <90 chars):
   - A contrarian statement challenging conventional wisdom about the topic.
2) Opening section (2–3 lines):
   - Expand on the hook with a relatable pain point or common mistake.
3) Main body:
   - Use a numbered list (3–5 points) OR bullet points.
   - Each point: 1–2 sentences max.
   - Include specific, concrete examples.
   - Add whitespace between sections.
4) Social proof / results:
   - Brief mention of outcome or data point validating the approach.
5) Closing:
   - End with a statement (NOT a question) that reinforces the main point and invites reflection.

Style and formatting:
- Length: 80–120 words total.
- Tone: direct, no‑nonsense, CEO‑speak; short sentences; minimal punctuation.
- Formatting: no bold; 2–3 line breaks for visual separation; lists for scannability; max 1–2 emojis (optional).
- Avoid AI tells (e.g., formulaic transitions, em dashes, excessive hyphenation).

Safety:
- Avoid named living‑person accusations and medical/legal/political persuasion.

Return ONLY valid JSON with the required keys and nothing else."""

def build_user_prompt(request: GenerationRequest, template: Template) -> str:
    """Build user prompt from request and template."""
    prompt = f"""Generate a LinkedIn post using this template:

Template: {template.name}
Structure: {template.structure}
Rules: {template.rules}

Input Parameters:
- Tone: {request.tone}
- Audience: {request.audience}
- Goal: {request.goal}
- Key Facts: {request.key_facts}
- Personal Detail: {request.personal_detail}

Requirements (must follow exactly):
- Hook (1 line, <90 chars): contrarian statement challenging conventional wisdom about the topic (derive topic from Key Facts/Personal Detail/Goal).
- Opening (2–3 lines): expand with a relatable pain point or common mistake.
- Main Body: numbered list (3–5) OR bullet points; each point 1–2 sentences; include concrete, specific examples; add whitespace.
- Social Proof/Results: brief outcome or data point validating the approach.
- Closing: end with a statement (NOT a question) that reinforces the main point and invites reflection.
- Length: 80–120 words.
- Tone: direct, no‑nonsense, CEO‑speak; short sentences; minimal punctuation.
- Formatting: no bold; use 2–3 line breaks for separation; lists for scannability; max 1–2 emojis if used.

Also ensure:
- hooks: 3 alternative contrarian hooks (<90 chars each), matching the topic.
- hashtags: 3–6 lowercase (no #, no punctuation).
- Maintain the JSON schema exactly as specified.

Return ONLY valid JSON with the required keys."""
    return prompt

def build_user_prompt_with_dm(request: GenerationRequest, template: Template) -> str:
    """Build user prompt with same structure; DM tone should match post tone."""
    prompt = f"""Generate a LinkedIn post using this template:
Template: {template.name}
Structure: {template.structure}
Rules: {template.rules}

Input Parameters:
- Tone: {request.tone}
- Audience: {request.audience}
- Goal: {request.goal}
- Key Facts: {request.key_facts}
- Personal Detail: {request.personal_detail}

Strict structure:
- Hook: 1 line, <90 chars, contrarian statement challenging conventional wisdom (topic from inputs).
- Opening: 2–3 lines expanding with relatable pain point or common mistake.
- Main Body: numbered or bullet list (3–5 points), each point 1–2 sentences, with concrete examples and whitespace.
- Social Proof/Results: a short outcome or data point.
- Closing: statement (NOT a question) reinforcing the main point and inviting reflection.

Style:
- 80–120 words total; direct, no‑nonsense, CEO‑speak; short sentences; minimal punctuation.
- No bold; 2–3 line breaks for separation; lists for scannability; max 1–2 emojis (optional).

Additionally:
- hooks: 3 contrarian alternatives (<90 chars each).
- hashtags: 3–6 lowercase (no #).
- DM tone should match post tone.

Return ONLY valid JSON with all required keys."""
    return prompt
