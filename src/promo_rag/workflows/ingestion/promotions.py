"""Keeping promotions tied to live pages."""

import uuid

from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session

from promo_rag.db.models import Promotion, PromotionSource


def unlink_promotions(session: Session, document_id: uuid.UUID) -> int:
    """Detach a document from its promotions, and delete promotions no page describes any more.

    Called when a document version is archived (its page changed or disappeared) and before
    a document is re-extracted. Returns how many promotions were deleted.
    """
    linked = session.scalars(
        select(PromotionSource.promotion_id).where(PromotionSource.document_id == document_id)
    ).all()
    session.execute(delete(PromotionSource).where(PromotionSource.document_id == document_id))
    still_described = exists().where(PromotionSource.promotion_id == Promotion.id)
    result = session.execute(delete(Promotion).where(Promotion.id.in_(linked), ~still_described))
    return result.rowcount or 0
