from __future__ import annotations
from typing import Dict, Any, Iterable, List
import json
from neo4j import GraphDatabase, Session

# Import vector index utilities from neo4j_graphrag.  These helpers are
# optional but provide convenient ways to manage vector indexes and store
# embeddings.  If neo4j_graphrag is not installed, you can replace these
# calls with explicit Cypher.
try:
    from neo4j_graphrag.indexes import create_vector_index, upsert_vectors
    from neo4j_graphrag.types import EntityType
except ImportError:
    create_vector_index = None  # type: ignore
    upsert_vectors = None  # type: ignore
    EntityType = None  # type: ignore


class Neo4jClient:
    """Simple wrapper around the Neo4j driver for upserting graphs and vectors."""

    def __init__(self, uri: str, auth: tuple[str, str]):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self) -> None:
        self.driver.close()

    def upsert_graph(self, graph: Dict[str, Any]) -> None:
        """Upsert entities, relations and attributes into Neo4j.

        Entities are merged by (type, canonical_name) keys; relations and
        attributes are created or updated accordingly.
        """
        with self.driver.session() as sess:
            # 實體
            for e in graph.get("entities", []):
                sess.execute_write(_merge_entity, e)
            # 關係
            for r in graph.get("relations", []):
                sess.execute_write(_merge_relation, r)
            # 屬性：寫為屬性節點或屬性關係，這裡簡化為節點屬性
            for a in graph.get("attributes", []):
                sess.execute_write(_attach_attribute, a)

    def upsert_chunks(self, chunks: Iterable[Any], index_name: str = "chunk_vector_index") -> None:
        """Upsert chunk nodes with embeddings and populate a vector index.

        Each chunk must have the following attributes: chunk_id, section_id,
        rev, text, and embedding (list of floats).  The method will merge
        Chunk nodes by id and set properties.  A vector index is created on
        the `embedding` property when available.

        Args:
            chunks: iterable of chunk objects with attributes.
            index_name: name of the Neo4j vector index to create/update.
        """
        # Collect chunk data to avoid multiple iterations
        chunk_list: List[Any] = list(chunks)
        if not chunk_list:
            return
        # Determine embedding dimension from the first chunk
        first_emb = getattr(chunk_list[0], "embedding", None)
        if first_emb is None:
            raise ValueError("Chunk embedding is missing; ensure embeddings are computed before upsert.")
        emb_dim = len(first_emb)
        with self.driver.session() as sess:
            # Create vector index (if create_vector_index is available)
            if create_vector_index is not None:
                create_vector_index(
                    self.driver,
                    index_name=index_name,
                    label="Chunk",
                    embedding_property="embedding",
                    dimensions=emb_dim,
                    similarity_fn="euclidean",
                )
            # Upsert chunk nodes and collect data for vector upsert
            ids: List[str] = []
            embeddings: List[List[float]] = []
            for chunk in chunk_list:
                # Merge the chunk node and set its properties
                sess.run(
                    "MERGE (c:Chunk {id:$id}) "
                    "SET c.section_id=$section_id, c.rev=$rev, c.text=$text, c.embedding=$embedding",
                    id=chunk.chunk_id,
                    section_id=getattr(chunk, "section_id", None),
                    rev=getattr(chunk, "rev", None),
                    text=getattr(chunk, "text", ""),
                    embedding=getattr(chunk, "embedding", None),
                )
                ids.append(chunk.chunk_id)
                embeddings.append(chunk.embedding)
            # Populate the vector index (if upsert_vectors is available)
            if upsert_vectors is not None and EntityType is not None:
                upsert_vectors(
                    self.driver,
                    ids=ids,
                    embedding_property="embedding",
                    embeddings=embeddings,
                    entity_type=EntityType.NODE,
                )


def _merge_entity(tx: Session, e: Dict[str, Any]) -> None:
    tx.run(
        "MERGE (n:Entity {type:$type, canonical_name:$name}) "
        "SET n.zh=$zh, n.en=$en",
        type=e["type"], name=e["canonical_name"], zh=e.get("zh"), en=e.get("en")
    )


def _merge_relation(tx: Session, r: Dict[str, Any]) -> None:
    tx.run(
        "MERGE (s:Entity {id:$s}) MERGE (o:Entity {id:$o}) "
        "MERGE (s)-[rel:REL {predicate:$p}]->(o)",
        s=r["subject"], p=r["predicate"], o=r["object"]
    )


def _attach_attribute(tx: Session, a: Dict[str, Any]) -> None:
    # 將嵌套的 dict 轉成 JSON 字串；空值轉成 None
    props: Dict[str, Any] = {}
    for k, v in a.items():
        if k == "owner":
            continue
        if isinstance(v, dict):
            props[k] = json.dumps(v, ensure_ascii=False)
        else:
            # 將空字串統一為 None，以免出現其他類型問題
            props[k] = v if v not in ("", None) else None
    tx.run(
        "MATCH (e:Entity {id:$owner}) SET e += $props",
        owner=a["owner"], props=props
    )