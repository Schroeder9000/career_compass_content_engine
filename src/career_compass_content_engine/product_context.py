"""Versioned launch facts shared by editorial prompts and website copy."""
import json
from pathlib import Path

PRODUCT_FACTS = json.loads((Path(__file__).parent / "data" / "product_facts.json").read_text())


def product_context() -> str:
    return ("\n\nCURRENT PRODUCT FACTS — accuracy reference only, not required copy. "
            "These describe the app's audience, not the broader blog audience. "
            "Do not insert product facts or a sales pitch when the article does not need them. "
            "Override conflicting historical examples:\n") + json.dumps(
        PRODUCT_FACTS, ensure_ascii=False, indent=2
    )
