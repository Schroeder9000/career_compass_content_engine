"""Versioned launch facts shared by editorial prompts and website copy."""
import json
from pathlib import Path

PRODUCT_FACTS = json.loads((Path(__file__).parent / "data" / "product_facts.json").read_text())


def product_context() -> str:
    return "\n\nCURRENT PRODUCT FACTS — override historical examples:\n" + json.dumps(
        PRODUCT_FACTS, ensure_ascii=False, indent=2
    )
