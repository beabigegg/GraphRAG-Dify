"""
cache.py
~~~~~~~~

This module provides a lightweight SQLite-backed cache for storing and
retrieving JSON responses from the Dify extraction API. The goal of this
cache is to avoid repeated calls to Dify for identical inputs (which can
consume tokens and incur latency) and to make debugging easier by
allowing you to inspect the raw responses that were persisted.

Each entry in the cache is keyed by a hash of the input payload (the
concatenation of the text, images, section identifier and revision
information) to uniquely identify an extraction request. The response is
stored as a JSON string. If you call the extractor with the same
payload again, the cached response will be returned instead of
triggering a new API call.

Usage::

    from .cache import Cache
    cache = Cache("./extraction_cache.db")
    cache.connect()
    cached = cache.get(hash_key)
    if cached is None:
        data = dify.extract(payload)
        cache.set(hash_key, data)
    else:
        data = cached
    cache.close()

The `Cache` class takes care of creating the underlying SQLite table on
first use. Note that the cache stores the full JSON response as
returned by Dify, not just the `structured_output` field, so that you
have access to all debugging information.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, Any


class Cache:
    """A simple SQLite-backed cache for storing JSON responses."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open a SQLite connection and initialize the cache table if needed."""
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the SQLite connection if it exists."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached response by its key.

        Args:
            key: The hash key identifying the payload.

        Returns:
            The parsed JSON object if present in the cache, otherwise ``None``.
        """
        if self.conn is None:
            raise RuntimeError("Cache must be connected before use")
        cur = self.conn.cursor()
        row = cur.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        # The stored value is a JSON string; deserialize it back into a Python object
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def set(self, key: str, response: Any) -> None:
        """Store a JSON response in the cache.

        Args:
            key: The hash key identifying the payload.
            response: The JSON-serializable object to be stored.
        """
        if self.conn is None:
            raise RuntimeError("Cache must be connected before use")
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
            (key, json.dumps(response)),
        )
        self.conn.commit()
