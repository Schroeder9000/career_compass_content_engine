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
        return AgentResult(content={"message": response.content}, raw_response=str(response))

    def build_prompt(self, instructions: str) -> str:
        return f"{self.system_prompt}\n\n{instructions}"
