from __future__ import annotations
from typing import Dict, Any
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri: str, auth: tuple[str,str]):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def upsert_graph(self, graph: Dict[str, Any]):
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


def _merge_entity(tx, e: Dict[str,Any]):
    tx.run(
        "MERGE (n:Entity {type:$type, canonical_name:$name}) "
        "SET n.zh=$zh, n.en=$en",
        type=e["type"], name=e["canonical_name"], zh=e.get("zh"), en=e.get("en")
    )

def _merge_relation(tx, r: Dict[str,Any]):
    tx.run(
        "MERGE (s:Entity {id:$s}) MERGE (o:Entity {id:$o}) "
        "MERGE (s)-[rel:REL {predicate:$p}]->(o)",
        s=r["subject"], p=r["predicate"], o=r["object"]
    )

def _attach_attribute(tx, a: Dict[str,Any]):
    tx.run(
        "MATCH (e:Entity {id:$owner}) SET e += $props",
        owner=a["owner"], props={k:v for k,v in a.items() if k not in {"owner"}}
    )