from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class EvidenceImage(BaseModel):
    image_id: str
    page: Optional[int] = None
    bbox: Optional[list[int]] = None  # [x1,y1,x2,y2]
    caption: Optional[str] = None
    purpose: Optional[str] = None


class SourceRef(BaseModel):
    section_id: Optional[str] = None
    page_hint: Optional[str] = None
    rev: Optional[str] = None


class Attribute(BaseModel):
    owner: str
    name: str
    operator: Optional[str] = None
    value: Any | None = None
    unit: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    applies_to: Optional[str] = None


class Entity(BaseModel):
    id: str
    type: str
    canonical_name: str
    zh: Optional[str] = None
    en: Optional[str] = None


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractedChunk(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    attributes: List[Attribute] = Field(default_factory=list)
    refs: List[str] = Field(default_factory=list)
    source: SourceRef = Field(default_factory=SourceRef)
    evidence: dict = Field(default_factory=dict)  # {"images": [EvidenceImage...]}
    notes: dict = Field(default_factory=dict)


class TextChunk(BaseModel):
    chunk_id: str
    section_id: Optional[str] = None
    rev: Optional[str] = None
    text: str
    images: List[EvidenceImage] = Field(default_factory=list)