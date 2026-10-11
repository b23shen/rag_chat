import csv
import hashlib

from promo_rag.storage.blob import LocalBlobStore, blob_key
from promo_rag.workflows.ingestion.actions.import_snapshot import load_snapshot


def _write_snapshot(folder, pages: dict[str, bytes], recorded_sha: dict[str, str] | None = None):
    (folder / "raw").mkdir()
    with open(folder / "sources.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slug", "url", "sha256"])
        writer.writeheader()
        for slug, data in pages.items():
            sha = (recorded_sha or {}).get(slug) or hashlib.sha256(data).hexdigest()
            writer.writerow({"slug": slug, "url": f"https://example.test/{slug}", "sha256": sha})
            if data is not None:
                (folder / "raw" / f"{slug}.html").write_bytes(data)


def test_load_snapshot_accepts_matching_bytes(tmp_path):
    _write_snapshot(tmp_path, {"a": b"<html>a</html>", "b": b"<html>b</html>"})
    loaded = load_snapshot(tmp_path)
    assert [p.row["slug"] for p in loaded.pages] == ["a", "b"]
    assert loaded.rejected == []


def test_load_snapshot_rejects_changed_bytes(tmp_path):
    _write_snapshot(tmp_path, {"a": b"<html>edited</html>"}, recorded_sha={"a": "0" * 64})
    loaded = load_snapshot(tmp_path)
    assert loaded.pages == []
    assert loaded.rejected == ["a: bytes changed since the fetch (sha256)"]


def test_load_snapshot_rejects_missing_file(tmp_path):
    _write_snapshot(tmp_path, {"a": b"<html>a</html>"})
    (tmp_path / "raw" / "a.html").unlink()
    loaded = load_snapshot(tmp_path)
    assert loaded.rejected == ["a: file missing"]


def test_blob_store_keys_by_content_and_stores_once(tmp_path):
    store = LocalBlobStore(tmp_path)
    data = b"<html>same bytes</html>"
    key1 = store.put(data, ".html")
    key2 = store.put(data, ".html")
    sha = hashlib.sha256(data).hexdigest()
    assert key1 == key2 == blob_key(sha, ".html") == f"raw/{sha[:2]}/{sha}.html"
    assert store.get(key1) == data
    assert len(list(tmp_path.rglob("*.html"))) == 1
