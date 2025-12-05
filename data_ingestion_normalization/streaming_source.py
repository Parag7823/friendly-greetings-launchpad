"""Utilities for working with streamed files stored on disk."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import xxhash  # ✅ FIX ISSUE #5: Import xxhash for standardized hashing


@dataclass
class StreamedFile:
    path: str
    filename: Optional[str] = None
    _size: Optional[int] = None
    _sha256: Optional[str] = None
    _xxh3_128: Optional[str] = None  # ✅ FIX ISSUE #5: Add xxh3_128 cache
    _cleanup: bool = False

    def __post_init__(self) -> None:
        self.path = str(self.path)
        if not self.filename:
            self.filename = Path(self.path).name

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = os.path.getsize(self.path)
        return self._size

    @property
    def sha256(self) -> str:
        """DEPRECATED: Use xxh3_128 instead for better performance"""
        if self._sha256 is None:
            self._sha256 = self.compute_sha256()
        return self._sha256

    @property
    def xxh3_128(self) -> str:
        """✅ FIX ISSUE #5: Compute xxh3_128 hash (fastest + collision-resistant)"""
        if self._xxh3_128 is None:
            self._xxh3_128 = self.compute_xxh3_128()
        return self._xxh3_128

    def iter_bytes(self, chunk_size: int = 8 * 1024 * 1024) -> Generator[bytes, None, None]:
        with open(self.path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as handle:
            return handle.read()

    def read_text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        with open(self.path, "r", encoding=encoding, errors=errors) as handle:
            return handle.read()

    def compute_sha256(self) -> str:
        hasher = hashlib.sha256()
        for chunk in self.iter_bytes():
            hasher.update(chunk)
        digest = hasher.hexdigest()
        self._sha256 = digest
        return digest

    def compute_xxh3_128(self) -> str:
        """✅ FIX ISSUE #5: Standardized hash using xxh3_128 (faster than SHA-256, collision-resistant)"""
        hasher = xxhash.xxh3_128()
        for chunk in self.iter_bytes():
            hasher.update(chunk)
        digest = hasher.hexdigest()
        self._xxh3_128 = digest
        return digest

    def open(self, mode: str = "rb"):
        return open(self.path, mode)

    def cleanup(self) -> None:
        if self._cleanup and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass

    def __enter__(self) -> "StreamedFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        filename: str,
        suffix: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> "StreamedFile":
        suffix = suffix or Path(filename).suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir)
        try:
            temp.write(data)
            temp.flush()
        finally:
            temp.close()
        return cls(path=temp.name, filename=filename, _size=len(data), _cleanup=True)
