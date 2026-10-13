"""search_promotion_text: the M2 search (vector top 20 → Cohere rerank → top k) as a tool.

For rules and fine print: how to return an item, which stores a coupon excludes, what a
campaign gives. Counts, lists and dates go to the SQL tools, since search ranks by topic
and returns only the top k.

The query must stand on its own: the model writes it, so a follow-up ("what about
Friday?") is rewritten into a full question before it is searched.
"""

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from promo_rag.search.retrieve import SearchError, search

# On the golden set, every answerable question's top rerank score was at least 0.61; 7 of
# the 20 unanswerable ones scored lower (M2 handoff). Fitted on those same questions, so
# a hint for the model, not a rule: near misses still score as high as real answers.
LOW_RELEVANCE = 0.61


class SearchArgs(BaseModel):
    query: str = Field(
        min_length=2,
        description="A full question that stands on its own, in the user's language or "
        "Chinese. Rewrite follow-ups with what they refer to: after a question about the "
        "LINE Pay offer, 'and on Friday?' becomes 'LINE Pay 優惠週五可以用嗎?'.",
    )
    k: int = Field(default=5, ge=1, le=10, description="How many passages to return.")


def search_promotion_text(session: Session, args: SearchArgs) -> dict:
    """The top k passages, best first, each with its page URL to cite, and a note when the
    scores say the pages may not answer the question."""
    try:
        hits = search(session, args.query, k=args.k)
    except SearchError as e:
        return {"error": f"search is unavailable: {e}"}
    result = {
        "results": [
            {
                "rank": h.rank,
                "score": round(h.score, 3),
                "url": h.source_url,
                "title": h.title,
                "text": h.text,
            }
            for h in hits
        ]
    }
    if not hits:
        result["note"] = "No passages found."
    elif not hits[0].reranked:
        result["note"] = (
            "The reranker was unavailable: passages are in vector order, and scores can't "
            "tell whether the pages answer the question."
        )
    elif hits[0].score < LOW_RELEVANCE:
        result["note"] = (
            f"Low relevance: the best passage scores {hits[0].score:.2f}, below "
            f"{LOW_RELEVANCE}. The pages may not answer this; if the passages don't, say so."
        )
    return result
