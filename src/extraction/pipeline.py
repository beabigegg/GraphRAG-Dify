from __future__ import annotations
from typing import List, Dict, Any
import json
from ..models.schema import TextChunk, ExtractedChunk
from ..dify.client import DifyClient
from ..config import DIFY


def run_extraction(chunks: List[TextChunk], dify: DifyClient) -> List[ExtractedChunk]:
    out: List[ExtractedChunk] = []
    for ch in chunks:
        payload = {
            "text": ch.text,
            "images": [
                {
                "transfer_method": "local_file",
                "upload_file_id": img.image_id,
                "type": "image"
                } 
                for img in ch.images
            ],
            "section_id": ch.section_id if ch.section_id is not None else "default_section",
            "rev": ch.rev,
        }
        data = dify.extract(payload)
        try:
            out.append(ExtractedChunk(**data))
        except Exception:
            # 容錯：若返回非嚴格結構，包成 notes
            out.append(ExtractedChunk(source={"section_id": ch.section_id, "rev": ch.rev}, notes={"raw": data}))
    return out


def merge_and_dedup(chunks: List[ExtractedChunk]) -> Dict[str, Any]:
    # 簡要示例：按 (owner,name,condition) 合併屬性
    entities: Dict[str, Dict] = {}
    relations: set[tuple] = set()
    attributes: Dict[tuple, Dict] = {}

    for ex in chunks:
        for e in ex.entities:
            key = (e.type, e.canonical_name.lower())
            entities[key] = {"type": e.type, "canonical_name": e.canonical_name, "zh": e.zh, "en": e.en}
        for r in ex.relations:
            relations.add((r.subject, r.predicate, r.object))
        for a in ex.attributes:
            cond = json.dumps(a.condition or {}, sort_keys=True)
            key = (a.owner, a.name, cond)
            attributes[key] = {
                "owner": a.owner, "name": a.name, "operator": a.operator,
                "value": a.value, "unit": a.unit, "condition": a.condition,
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