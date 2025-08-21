from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import hashlib
from ..models.schema import TextChunk, ExtractedChunk
from ..dify.client import DifyClient
from .cache import Cache


def run_extraction(
    chunks: List[TextChunk],
    dify: DifyClient,
    cache_db_path: Optional[str] = None,
    debug: bool = False,
) -> List[ExtractedChunk]:
    """Run extraction on a list of text/image chunks.

    This function optionally uses a SQLite-backed cache to avoid repeated
    calls to Dify for identical inputs. It also supports emitting debug
    messages to the console when cache hits/misses occur and when
    unexpected data is encountered.

    Args:
        chunks: A list of :class:`TextChunk` instances to extract entities
            and relations from.
        dify: An initialized :class:`DifyClient` for communicating with
            the Dify API.
        cache_db_path: If provided, the path to a SQLite database file
            where raw Dify responses will be cached. When set, the
            extractor will attempt to read from the cache before making
            an API call and will write new responses back to the cache.
        debug: If ``True``, emit verbose debug information to stdout.

    Returns:
        A list of :class:`ExtractedChunk` instances corresponding to
        each input chunk.
    """
    out: List[ExtractedChunk] = []
    cache: Optional[Cache] = None
    if cache_db_path:
        cache = Cache(cache_db_path)
        cache.connect()

    def _get_structured(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to locate the structured_output within a nested response.

        The Dify API may return the structured output in several locations:

        - At the top level under 'structured_output'
        - Under 'outputs' → 'structured_output'
        - Nested inside a 'data' object, as seen in workflow responses

        Args:
            data: The raw JSON response from Dify (or cached equivalent).

        Returns:
            The structured_output dict if found, otherwise ``None``.
        """
        if not isinstance(data, dict):
            return None
        if 'structured_output' in data:
            return data['structured_output']
        # Handle responses with 'outputs' at the top level
        if 'outputs' in data and isinstance(data['outputs'], dict):
            out = data['outputs']
            if 'structured_output' in out:
                return out['structured_output']
        # Handle nested 'data' wrapper (e.g., workflow API responses)
        if 'data' in data and isinstance(data['data'], dict):
            return _get_structured(data['data'])
        return None

    try:
        for ch in chunks:
            # Build payload following the existing contract: include text,
            # images (converted to the format expected by Dify), section
            # identifier and revision.
            payload = {
                "text": ch.text,
                "images": [
                    {
                        "transfer_method": "local_file",
                        "upload_file_id": img.image_id,
                        "type": "image",
                    }
                    for img in getattr(ch, "images", [])
                ],
                "section_id": ch.section_id if ch.section_id is not None else "default_section",
                "rev": ch.rev,
            }

            # Compute a deterministic hash of the payload for caching.
            payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            key = hashlib.md5(payload_str.encode("utf-8")).hexdigest()

            # Attempt to fetch from cache if enabled
            data: Dict[str, Any]
            if cache:
                cached = cache.get(key)
                if cached is not None:
                    data = cached
                    if debug:
                        print(f"[cache] hit for key={key}")
                else:
                    if debug:
                        print(f"[cache] miss for key={key}, calling Dify")
                    data = dify.extract(payload)
                    cache.set(key, data)
            else:
                data = dify.extract(payload)

            # Dify may wrap the structured result in various ways; extract it.
            structured_data = _get_structured(data)
            if structured_data is None:
                # No structured output found; record raw response for debugging
                if debug:
                    print(f"[debug] structured_output not found in response for key={key}")
                out.append(
                    ExtractedChunk(
                        source={"section_id": ch.section_id, "rev": ch.rev},
                        notes={"raw": data},
                    )
                )
                continue
            # Normalize attribute fields: convert empty strings to None and
            # parse JSON strings for `condition` if needed. Dify may return
            # condition/applies_to as empty string or JSON string, which
            # conflicts with the Pydantic schema expecting dict or None.
            attrs = structured_data.get("attributes", [])
            for attr in attrs:
                # Normalize condition: empty string → None; JSON string → dict
                if isinstance(attr.get("condition"), str):
                    cond_str = attr["condition"].strip()
                    if cond_str == "":
                        attr["condition"] = None
                    elif cond_str.startswith("{"):
                        try:
                            attr["condition"] = json.loads(cond_str)
                        except Exception:
                            # leave as raw string if cannot parse
                            pass
                # Normalize applies_to: empty string → None
                if isinstance(attr.get("applies_to"), str) and attr["applies_to"].strip() == "":
                    attr["applies_to"] = None

            try:
                out.append(ExtractedChunk(**structured_data))
            except Exception as exc:
                # Fallback: store raw structured_data in notes for debugging
                if debug:
                    print(f"[debug] Failed to parse structured_data: {exc}")
                    print(f"[debug] Raw structured_data: {structured_data}")
                out.append(
                    ExtractedChunk(
                        source={"section_id": ch.section_id, "rev": ch.rev},
                        notes={"raw": structured_data},
                    )
                )
    finally:
        if cache:
            cache.close()
    return out


def merge_and_dedup(chunks: List[ExtractedChunk]) -> Dict[str, Any]:
    """Merge and deduplicate entities, relations and attributes.

    This function iterates over the extracted chunks, normalizes and
    deduplicates their constituent entities, relations and attributes. It
    returns a simplified graph representation suitable for upserting into
    a graph database such as Neo4j.

    Args:
        chunks: A list of :class:`ExtractedChunk` instances.

    Returns:
        A dict with three keys: ``entities``, ``relations`` and
        ``attributes``. Each is a list of dictionaries ready for
        consumption by downstream graph loaders.
    """
    entities: Dict[tuple, Dict] = {}
    relations: set[tuple] = set()
    attributes: Dict[tuple, Dict] = {}

    for ex in chunks:
        for e in ex.entities:
            key = (e.type, e.canonical_name.lower())
            entities[key] = {
                "type": e.type,
                "canonical_name": e.canonical_name,
                "zh": e.zh,
                "en": e.en,
            }
        for r in ex.relations:
            relations.add((r.subject, r.predicate, r.object))
        for a in ex.attributes:
            cond = json.dumps(a.condition or {}, sort_keys=True)
            key = (a.owner, a.name, cond)
            attributes[key] = {
                "owner": a.owner,
                "name": a.name,
                "operator": a.operator,
                "value": a.value,
                "unit": a.unit,
                "condition": a.condition,
            }
    return {
        "entities": [
            {"id": f"E{i}", **v} for i, v in enumerate(entities.values(), start=1)
        ],
        "relations": [
            {"subject": s, "predicate": p, "object": o} for (s, p, o) in sorted(relations)
        ],
        "attributes": list(attributes.values()),
    }