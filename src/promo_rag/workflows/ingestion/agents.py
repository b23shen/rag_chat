"""Agents in the ingestion workflow: model role, prompt and output schema for each.

Tools and middleware are shared (promo_rag.agents) and referenced by name.
"""

from pathlib import Path

from promo_rag.agents import AgentConfig, register_agent

PROMPTS = Path(__file__).parent / "prompts"

FACT_EXTRACTOR = register_agent(
    AgentConfig(
        name="fact_extractor",
        description="Fills a facts schema (PromotionFacts or StoreNoticeFacts) from one page.",
        prompt=PROMPTS / "extract_facts.md",
        model="helper",
        # No default schema: each call passes the schema for the page's doc_type.
        response_schema=None,
        middleware=("model_fallback",),
    )
)
