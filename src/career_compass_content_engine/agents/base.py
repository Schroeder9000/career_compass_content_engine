from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


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
        # o1 models do not support system messages — prepend the persona to the
        # human message instead. All other models use the standard two-message format.
        model_name = getattr(self.llm, "model_name", "") or ""
        if model_name.startswith("o1"):
            messages = [HumanMessage(content=f"{self.system_prompt}\n\n{prompt}")]
        else:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]

        response = self.llm.invoke(messages)
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
