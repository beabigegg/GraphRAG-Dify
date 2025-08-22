from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvidenceImage(BaseModel):
    """Metadata for an image extracted from a document.

    Each evidence image references a page number and bounding box.
    """
    image_id: str
    page: Optional[int] = None
    bbox: Optional[list[int]] = None  # [x1,y1,x2,y2]
    caption: Optional[str] = None
    purpose: Optional[str] = None


class SourceRef(BaseModel):
    """Reference back to the source document location for a chunk or entity."""
    section_id: Optional[str] = None
    page_hint: Optional[str] = None
    rev: Optional[str] = None


class Attribute(BaseModel):
    """Represents an attribute extracted from a chunk or entity.

    Attributes are stored as simple key–value pairs on the owning entity.
    """
    owner: str
    name: str
    operator: Optional[str] = None
    value: Any | None = None
    unit: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    applies_to: Optional[str] = None


class Entity(BaseModel):
    """Represents a named entity extracted from text."""
    id: str
    type: str
    canonical_name: str
    zh: Optional[str] = None
    en: Optional[str] = None


class Relation(BaseModel):
    """Represents a relationship between two entities."""
    subject: str
    predicate: str
    object: str


class ExtractedChunk(BaseModel):
    """Container for the structured extraction result of a chunk.

    It includes extracted entities, relations, attributes, references back to
    the source, supporting evidence images and any notes.
    """
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    attributes: List[Attribute] = Field(default_factory=list)
    refs: List[str] = Field(default_factory=list)
    source: SourceRef = Field(default_factory=SourceRef)
    evidence: dict = Field(default_factory=dict)  # {"images": [EvidenceImage...]}
    notes: dict = Field(default_factory=dict)


class TextChunk(BaseModel):
    """Represents a unit of text (chunk) to be processed.

    A chunk contains the raw text, optional section identifier and revision,
    any associated images, and an optional embedding vector.  The embedding
    vector can be generated using local embedding models (e.g. via Ollama).
    """
    chunk_id: str
    section_id: Optional[str] = None
    rev: Optional[str] = None
    text: str
    images: List[EvidenceImage] = Field(default_factory=list)
    embedding: Optional[List[float]] = None  # populated after embedding generation