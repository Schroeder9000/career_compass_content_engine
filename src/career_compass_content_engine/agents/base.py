from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class AgentContext:
    """Shared context passed between agents."""

    brand_voice: str
    brand_colors: str
    target_audience: str
    distribution_channels: str


@dataclass
class AgentResult:
    """Typed response from each agent."""

    content: Dict[str, Any]
    raw_response: str


@dataclass
class BaseAgent:
    """Base agent with a chat model and a lightweight interface."""

    llm: BaseChatModel
    name: str
    system_prompt: str

    def run(self, prompt: str) -> AgentResult:  # pragma: no cover - thin wrapper
        response = self.llm.invoke(prompt)
        # Normalize raw_response: if the model returned structured content, serialize it
        raw = response.content
        try:
            # If content is already a string, keep as-is; otherwise JSON-serialize
            if isinstance(raw, str):
                raw_response = raw
            else:
                import json

                raw_response = json.dumps(raw)
        except Exception:
            raw_response = str(response)

        return AgentResult(content={"message": response.content}, raw_response=raw_response)

    def build_prompt(self, instructions: str) -> str:
        return f"{self.system_prompt}\n\n{instructions}"
