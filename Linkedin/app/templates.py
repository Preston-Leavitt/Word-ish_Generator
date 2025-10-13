import fitz  # PyMuPDF
from typing import Dict, List
from .schemas import Template

# Fallback templates if PDF parsing fails
FALLBACK_TEMPLATES = {
    "you_dont_actually_want_that": Template(
        id="you_dont_actually_want_that",
        name="You Don't Actually Want That",
        structure="Hook + Problem + Solution + Personal Story + CTA",
        rules="Start with contrarian hook, expose hidden problem, provide solution, share personal experience"
    ),
    "the_simple_framework": Template(
        id="the_simple_framework",
        name="The Simple Framework",
        structure="Hook + Framework Steps + Example + CTA",
        rules="Present a simple framework, break into 3-5 steps, provide concrete example"
    ),
    "mistake_story": Template(
        id="mistake_story",
        name="Mistake Story",
        structure="Hook + Mistake + Lesson + Application + CTA",
        rules="Share personal mistake, extract lesson, show how to apply"
    ),
    "implied_expertise": Template(
        id="implied_expertise",
        name="Implied Expertise",
        structure="Hook + Anecdote + Insight + Actionable Tip + CTA",
        rules="Start with a short anecdote that implies expertise, extract the non-obvious insight, give one practical tip"
    ),
    "learning_the_hard_way": Template(
        id="learning_the_hard_way",
        name="Learning The Hard Way",
        structure="Hook + Struggle + Turning Point + Result + Takeaway + CTA",
        rules="Admit a painful struggle, show how you changed approach, highlight concrete results and the lesson"
    ),
    "broken_mindset": Template(
        id="broken_mindset",
        name="A Broken Mindset",
        structure="Provocative Hook + List of Beliefs + Reframe + Steps to Change + CTA",
        rules="Name a damaging belief, explain why it fails, offer 3 practical reframes or steps"
    ),
    "bucking_the_trend": Template(
        id="bucking_the_trend",
        name="Bucking The Trend",
        structure="Hook + Trend Callout + Counterargument + Evidence + Advice + CTA",
        rules="Call out a popular trend, argue against it with crisp evidence, give alternative approach"
    ),
    "resource_curation": Template(
        id="resource_curation",
        name="Resource Curation",
        structure="Hook + Short List of Resources + Why Each Helps + CTA",
        rules="Give X high-quality resources, 1 sentence why each matters, invite engagement/download"
    ),
    "quick_hypothetical": Template(
        id="quick_hypothetical",
        name="A Quick Hypothetical",
        structure="Hook + Hypothetical Scenario + Breakdown + Lesson + CTA",
        rules="Pose a vivid what-if, walk through consequences, finish with the takeaway and action"
    ),
    "youve_been_lied_to": Template(
        id="youve_been_lied_to",
        name="You've Been Lied To",
        structure="Contrarian Hook + Myth Debunk + Evidence + Real Truth + CTA",
        rules="Open with a punchy contradiction, debunk with 1–2 facts or examples, state the real rule"
    ),
    "weve_all_been_there": Template(
        id="weve_all_been_there",
        name="We've All Been There",
        structure="Relatable Hook + Short Story + Empathy + Practical Next Step + CTA",
        rules="Normalize a common failure, show empathy, give a small immediate step readers can take"
    ),
    "secret_weapon": Template(
        id="secret_weapon",
        name="My Secret Weapon",
        structure="Hook + Reveal (unexpected tool) + How I used it + Results + CTA",
        rules="Reveal an unconventional tactic/tool, show quick example and impact, invite DMs for details"
    ),
    "problem_agitate_solution": Template(
        id="problem_agitate_solution",
        name="Problem Agitate Solution (PAS)",
        structure="Hook (problem) + Agitation (why it hurts) + Solution + Microproof + CTA",
        rules="Clearly state the pain, amplify consequence, give a concise solution and one proof point"
    ),
    "giving_words_to_the_unvoiced": Template(
        id="giving_words_to_the_unvoiced",
        name="Giving Words to the Unvoiced",
        structure="Hook + Observe an Unsaid Issue + Frame the Impact + Suggested Fix + CTA",
        rules="Surface a hidden or ignored problem, explain its impact, offer one systemic or policy fix"
    ),
    "an_irresistible_solution": Template(
        id="an_irresistible_solution",
        name="An Irresistible Solution",
        structure="Hook + Problem Recap + Unique Solution + Benefits List + CTA",
        rules="Present a crisp solution to a familiar problem and list 3 tangible benefits"
    ),
    "anticipating_skepticism": Template(
        id="anticipating_skepticism",
        name="Anticipating Skepticism",
        structure="Hook (skeptical question) + Answer + Evidence + Rule of 3 Reasons + CTA",
        rules="Pose the reader's doubt, answer directly, give 3 quick reasons to reduce friction"
    ),
    "immersive_storytelling": Template(
        id="immersive_storytelling",
        name="The Art of Immersive Storytelling",
        structure="Hook + Scene Setup + Rising Tension + Turning Point + Emotional Lesson + CTA",
        rules="Use vivid sensory detail, build a short narrative arc, close with an emotionally resonant lesson"
    ),
    "stacking_transformations": Template(
        id="stacking_transformations",
        name="Stacking Transformations",
        structure="Hook + Before/After Bullets + Process Steps + Composite Result + CTA",
        rules="Show multiple small wins stacked over time; list the steps that compound to the outcome"
    ),
    "radical_candour": Template(
        id="radical_candour",
        name="Radical Candour",
        structure="Bold Hook + Brutal Truth + Constructive Advice + Boundaries + CTA",
        rules="Make a risky honest claim, pair it with a constructive path forward and a clear boundary"
    ),
    "you_dont_wanna_miss_this": Template(
        id="you_dont_wanna_miss_this",
        name="You Don't Wanna Miss This",
        structure="Hook (FOMO) + What You're Missing + Quick Proof + How to Join + CTA",
        rules="Create urgency with a clear opportunity, show social proof, provide a simple next step"
    ),
    "gripping_story": Template(
        id="gripping_story",
        name="A Gripping Story",
        structure="Hook + Character Intro + Conflict + Cliffhanger or Resolution + Moral + CTA",
        rules="Tell a tight human story that hooks fast and leaves reader moved or curious"
    ),
    "imagine_this_scenario": Template(
        id="imagine_this_scenario",
        name="Imagine This Scenario",
        structure="Hypnotic Hook + Future Vision + Steps to Get There + Call to Action",
        rules="Paint a future-state vividly, then give 3 actionable steps to start moving toward it"
    )

}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""


def parse_pdf_templates(pdf_paths: List[str]) -> Dict[str, Template]:
    """Parse PDF files to extract templates."""
    templates = {}
    
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        if text:
            # Simple parsing - look for template patterns
            # This is a simplified implementation
            template_id = pdf_path.split('/')[-1].replace('.pdf', '').lower().replace(' ', '_')
            
            template = Template(
                id=template_id,
                name=template_id.replace('_', ' ').title(),
                structure=extract_structure_from_text(text),
                rules=extract_rules_from_text(text)
            )
            templates[template_id] = template
    
    return templates


def extract_structure_from_text(text: str) -> str:
    """Extract template structure from PDF text."""
    # Simplified extraction - look for common patterns
    if "hook" in text.lower():
        return "Hook + Problem + Solution + CTA"
    return "Introduction + Body + Conclusion + CTA"


def extract_rules_from_text(text: str) -> str:
    """Extract template rules from PDF text."""
    # Simplified extraction
    return "Follow template structure, keep paragraphs short, end with clear CTA"


def load_templates(pdf_paths: List[str] = None) -> Dict[str, Template]:
    """Load templates from PDFs or use fallbacks."""
    if pdf_paths:
        templates = parse_pdf_templates(pdf_paths)
        if templates:
            return templates
    
    # Use fallback templates
    return FALLBACK_TEMPLATES


def get_template_by_id(template_id: str, templates: Dict[str, Template]) -> Template:
    """Get template by ID."""
    return templates.get(template_id)
