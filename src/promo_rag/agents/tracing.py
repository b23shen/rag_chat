"""Langfuse tracing for agent calls.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. It must never block or break
a call: the SDK sends traces from a background thread, and a setup error only logs a warning.
"""

import logging

from promo_rag.config import get_settings

log = logging.getLogger(__name__)
_client = None


def langfuse_handler():
    """A LangChain callback that sends this call's trace to Langfuse, or None if tracing is off."""
    global _client
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        if _client is None:
            _client = Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key.get_secret_value(),
                host=s.langfuse_host,
            )
        return CallbackHandler(public_key=s.langfuse_public_key)
    except Exception as e:  # noqa: BLE001 - tracing problems must not stop the call
        log.warning("Langfuse tracing is off: %s", e)
        return None


def flush_traces() -> None:
    """Send buffered traces before a batch job exits. A no-op if tracing is off."""
    if _client is not None:
        _client.flush()
