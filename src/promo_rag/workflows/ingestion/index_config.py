"""The settings an index is built with. Their hash is the corpus version's fingerprint.

Change any setting and the fingerprint changes, so the chunks (and their embeddings) are
built again as a new corpus version, and eval results are never compared across settings.
"""

import hashlib
import json

from pydantic import BaseModel, Field

from promo_rag.config import get_settings
from promo_rag.search.segment import Segmenter, word_list_version


def _embed_model() -> str:
    return get_settings().embed_model


def _embed_dims() -> int:
    return get_settings().embed_dims


# gemini-embedding-2's task prefixes for questions (ai.google.dev/gemini-api/docs/embeddings).
# "none" sends the bare question, as a control for how much the prefix matters.
QUERY_FORMATS = {
    "search": "task: search result | query: {query}",
    "qa": "task: question answering | query: {query}",
    "none": "{query}",
}


class IndexConfig(BaseModel):
    # Bump when the chunking code changes in a way that changes its output.
    chunker: str = "sentence-pack-v2"  # v2: posting date and section in headers
    chunk_tokens: int = 512
    overlap_tokens: int = 100
    # Gemini tokens are about 1.1 Chinese characters each on these pages (measured with
    # count_tokens on 4 pages: 1.07–1.16). Used to estimate sizes without an API call.
    chars_per_token: float = 1.1
    header: bool = True  # start each chunk with "title | dates | stores"
    segmenter: Segmenter = "jieba"
    word_list: str = Field(default_factory=word_list_version)
    embed_model: str = Field(default_factory=_embed_model)
    embed_dims: int = Field(default_factory=_embed_dims)
    # gemini-embedding-2 takes the task as a text prefix. The query format is an eval
    # experiment ("task: search result" vs "task: question answering").
    embed_doc_format: str = "title: {title} | text: {text}"
    embed_query_format: str = QUERY_FORMATS["search"]

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
