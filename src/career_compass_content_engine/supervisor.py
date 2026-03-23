from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .agents.base import AgentContext
from .config import Settings, configure_logging, get_settings
from .rag import RAGPipeline

logger = logging.getLogger(__name__)


def build_llm(settings: Settings, task_type: str = "default") -> ChatOpenAI:
    """
    Build LLM optimized for specific task types.
    
    Task types:
    - research: Strategic analysis and planning (highest reasoning)
    - writing: Long-form content generation (high creativity and quality)
    - editing: Refinement and polish (balanced quality and consistency)
    - image: Visual concept design (creative but structured)
    - metadata: Structured data generation (precision and consistency)
    """
    # Use the most advanced models for highest quality output
    # o1 for research (deep reasoning), GPT-4 for content generation
    model_config = {
        "research": {
            "model": "o1",  # Highest reasoning for strategic analysis
            "temperature": 1,  # o1 models use temperature=1
        },
        "writing": {
            "model": "gpt-4o",  # Best for long-form creative content
            "temperature": 0.7,  # Higher creativity for engaging content
        },
        "editing": {
            "model": "gpt-4o",  # Best quality for refinement
            "temperature": 0.5,  # Balanced for consistency with creativity
        },
        "image": {
            "model": "gpt-4o",  # Best for creative visual concepts
            "temperature": 0.8,  # High creativity for unique visuals
        },
        "metadata": {
            "model": "gpt-4o",  # Best for structured output
            "temperature": 0.3,  # Lower for consistent formatting
        },
        "evaluation": {
            "model": "gpt-4o",  # Deterministic scoring requires consistency
            "temperature": 0,   # Zero temperature for reproducible evaluation scores
        },
        "revision": {
            "model": "gpt-4o",  # Same quality model for revision editing
            "temperature": 0.8,  # Higher than first edit — force genuinely different rewrites
        },
        "default": {
            "model": "gpt-4o",  # Default to best general model
            "temperature": 0.7,
        },
    }
    
    config = model_config.get(task_type, model_config["default"])
    return ChatOpenAI(
        model=config["model"],
        api_key=settings.openai_api_key,
        temperature=config["temperature"],
    )


@dataclass
class WorkflowResult:
    research: Dict[str, str]
    blog_post: str
    image_plan: Dict[str, str]
    publish_result: Dict[str, str]
    # Token / cost telemetry (populated when langchain-community is installed)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class Supervisor:
    """
    Thin orchestration wrapper around the LangGraph content pipeline.

    Builds the StateGraph once on construction, then delegates each
    run_workflow() call to graph.invoke(). All agent instantiation,
    state management, and node routing live in graph.py.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        from .graph import build_graph  # deferred to avoid circular import at module load

        configure_logging()
        self.settings = settings or get_settings()
        self.settings.configure_tracing()
        self.rag = rag_pipeline or RAGPipeline.from_examples()
        self._graph = build_graph(settings=self.settings, rag_pipeline=self.rag)

    async def run_workflow(self, existing_content: str, context: AgentContext) -> WorkflowResult:
        """Invoke the compiled graph and return a structured WorkflowResult.

        This is an async method — await it directly or call
        ``asyncio.run(supervisor.run_workflow(...))`` from synchronous code.
        Token usage and estimated USD cost are captured via
        ``get_openai_callback`` and attached to the returned WorkflowResult.
        """
        run_id = str(uuid.uuid4())
        config = RunnableConfig(
            run_name="career-compass-blog-workflow",
            metadata={"run_id": run_id},
            tags=["career-compass", "blog-generation"],
            configurable={"thread_id": run_id},
        )
        with get_openai_callback() as cb:
            final_state = await self._graph.ainvoke(
                {
                    "existing_content": existing_content,
                    "context": context,
                },
                config=config,
            )
        logger.info(
            "[workflow] run %s complete — tokens: %d (prompt %d / completion %d)  est. cost: $%.4f",
            run_id, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost,
        )
        return WorkflowResult(
            research=final_state.get("research_payload", {}),
            blog_post=final_state.get("edited_content", ""),
            image_plan=final_state.get("image_plan", {}),
            publish_result=final_state.get("publish_result", {}),
            total_tokens=cb.total_tokens,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            total_cost_usd=cb.total_cost,
            run_id=run_id,
        )
