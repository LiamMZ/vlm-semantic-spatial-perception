from typing import Dict, Optional


def render_prompt_template(template: Optional[str], replacements: Dict[str, str]) -> str:
    """
    Replace placeholder tokens in a prompt template.

    Placeholders use the format <<TOKEN>>. Any missing token renders as empty string.
    """
    if template is None:
        raise KeyError("Prompt template is missing")

    rendered = template
    for key, value in replacements.items():
        placeholder = f"<<{key}>>"
        if placeholder not in rendered:
            raise KeyError(f"Prompt template missing placeholder: {placeholder}")
        rendered = rendered.replace(placeholder, value or "")
    return rendered

