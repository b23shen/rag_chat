"""What the ingestion actions need from the outside world, in one place."""

from promo_rag.config import get_settings
from promo_rag.db.session import new_session
from promo_rag.storage.blob import BlobStore, LocalBlobStore


def blob_store() -> BlobStore:
    return LocalBlobStore(get_settings().blob_dir)


def session():
    return new_session()
