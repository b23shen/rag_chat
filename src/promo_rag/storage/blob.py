"""Blob store for raw page bytes: a local folder in dev, GCS in the cloud (M5).

Keys are derived from the bytes' sha256, so the same bytes are stored once
and putting them again is a no-op.
"""

import hashlib
from pathlib import Path
from typing import Protocol


def blob_key(sha256: str, ext: str = "") -> str:
    return f"raw/{sha256[:2]}/{sha256}{ext}"


class BlobStore(Protocol):
    def put(self, data: bytes, ext: str = "") -> str: ...
    def get(self, key: str) -> bytes: ...


class LocalBlobStore:
    def __init__(self, root: Path):
        self.root = root

    def put(self, data: bytes, ext: str = "") -> str:
        key = blob_key(hashlib.sha256(data).hexdigest(), ext)
        path = self.root / key
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file and rename, so a crash never leaves half a file under the key.
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return key

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()
