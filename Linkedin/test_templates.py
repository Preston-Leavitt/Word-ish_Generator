from app.templates import load_templates

def test_templates():
    """Test that templates can be loaded."""
    templates = load_templates()
    print(f"Loaded {len(templates)} templates")
    
    for template_id, template in templates.items():
        print(f"Template: {template_id} - {template.name}")
        print(f"  Structure: {template.structure}")
        print(f"  Rules: {template.rules}")
        print()

if __name__ == "__main__":
    test_templates()
