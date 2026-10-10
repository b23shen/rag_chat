"""Check that Postgres, pgvector and the Gemini key all work before building anything.

Usage: uv run python scripts/smoke_test.py
"""

import math
import sys
from datetime import date

import httpx
import psycopg
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from promo_rag.config import get_settings

# API errors (bad key, quota, unknown model) and network errors (timeouts, DNS).
GEMINI_ERRORS = (errors.APIError, httpx.HTTPError)


def check_postgres(s) -> bool:
    print("\n[1] Postgres + pgvector")
    try:
        with psycopg.connect(s.pg_conninfo, connect_timeout=5) as conn:
            version = conn.execute("SHOW server_version").fetchone()[0]
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if row is None:
                print(
                    f"  Postgres {version} OK, but pgvector is not enabled in {s.pg_db}. Run scripts/init_db.sh."
                )
                return False
            # Cosine distance between two tiny vectors, computed by pgvector.
            dist = conn.execute("SELECT '[0.9,0.1]'::vector <=> '[0.8,0.2]'::vector").fetchone()[0]
            print(f"  Postgres {version}, pgvector {row[0]}, database {s.pg_db}")
            print(
                f"  cosine distance [0.9,0.1] vs [0.8,0.2] = {dist:.3f} (similarity {1 - dist:.2f})"
            )
            return True
    except psycopg.Error as e:
        print(f"  Can't connect: {e}")
        return False


def list_models(client: genai.Client) -> None:
    print("\n[2] Gemini models this key can use")
    chat, embed = [], []
    for m in client.models.list():
        actions = m.supported_actions or []
        name = m.name.removeprefix("models/")
        if "generateContent" in actions:
            chat.append(name)
        if "embedContent" in actions:
            embed.append(name)
    print("  chat:      " + ", ".join(sorted(chat)))
    print("  embedding: " + ", ".join(sorted(embed)))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def check_embeddings(client: genai.Client, s) -> bool:
    print(f"\n[3] Embeddings ({s.embed_model}, {s.embed_dims} dims)")
    texts = [
        "task: search result | query: cashback",
        "title: none | text: money back offer",
        "title: none | text: store opening hours",
    ]
    try:
        resp = client.models.embed_content(
            model=s.embed_model,
            # One Content per text. gemini-embedding-2 reads a plain list[str] as the parts of
            # ONE input and returns a single blended vector.
            contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
            config=types.EmbedContentConfig(output_dimensionality=s.embed_dims),
        )
    except GEMINI_ERRORS as e:
        print(f"  Failed: {e}")
        return False
    vecs = [e.values for e in resp.embeddings]
    if len(vecs) != len(texts):
        print(f"  Sent {len(texts)} texts but got {len(vecs)} vectors back")
        return False
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    print(f"  got {len(vecs)} vectors of {len(vecs[0])} dims; length of the first = {norm:.3f}")
    print(f"  'cashback' vs 'money back offer':    {cosine(vecs[0], vecs[1]):.3f}")
    print(f"  'cashback' vs 'store opening hours': {cosine(vecs[0], vecs[2]):.3f}")
    return len(vecs[0]) == s.embed_dims


class PromoDates(BaseModel):
    title: str
    start_date: date | None
    end_date: date | None


def check_structured_output(client: genai.Client, s) -> bool:
    print("\n[4] Structured output (Gemini -> Pydantic)")
    if not s.helper_model:
        print("  Skipped: set HELPER_MODEL in .env to one of the chat models above.")
        return False
    try:
        resp = client.models.generate_content(
            model=s.helper_model,
            contents="Extract the promotion: 'Mid-Autumn coupon, valid 20 Sep to 6 Oct 2026 at all stores.'",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PromoDates,
            ),
        )
    except GEMINI_ERRORS as e:
        print(f"  Failed: {e}")
        return False
    facts: PromoDates = resp.parsed
    print(f"  {s.helper_model} -> {facts!r}")
    return facts is not None and facts.start_date is not None


def main() -> int:
    try:
        s = get_settings()
    except ValidationError as e:
        missing = ", ".join(str(err["loc"][0]).upper() for err in e.errors())
        print(f"Missing settings in .env: {missing}")
        return 1

    results = {"postgres": check_postgres(s)}
    client = genai.Client(
        api_key=s.gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(timeout=30_000),
    )
    try:
        list_models(client)
    except GEMINI_ERRORS as e:
        print(f"  Gemini call failed (check GEMINI_API_KEY): {e}")
        return 1
    results["embeddings"] = check_embeddings(client, s)
    results["structured output"] = check_structured_output(client, s)

    print("\nSummary: " + ", ".join(f"{k} {'OK' if v else 'FAILED'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
