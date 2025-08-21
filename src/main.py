from __future__ import annotations
import argparse
from pathlib import Path
from tqdm import tqdm
from .config import DIFY, NEO4J
from .io.docx_parser import load_docx_text_chunks
from .dify.client import DifyClient
from .extraction.pipeline import run_extraction, merge_and_dedup
from .graph.neo4j_client import Neo4jClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="path to DOCX")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--section-hint", default=None)
    ap.add_argument("--push-graph", action="store_true")
    args = ap.parse_args()

    docx_path = Path(args.doc)
    assert docx_path.exists(), f"Not found: {docx_path}"

    print("[1/4] 解析 DOCX…")
    chunks = load_docx_text_chunks(str(docx_path), rev=args.rev)

    print(f"已生成 {len(chunks)} 個 chunk。")
    dify = DifyClient(DIFY.base_url, DIFY.api_key, DIFY.app_type, DIFY.response_mode, DIFY.user_id)

    print("[2/4] 呼叫 Dify 進行抽取…")
    extracted = []
    for ch in tqdm(chunks):
        ex = run_extraction([ch], dify)
        extracted.extend(ex)

    print("[3/4] 合併與去重…")
    graph = merge_and_dedup(extracted)
    print(f"entities={len(graph['entities'])}, relations={len(graph['relations'])}, attributes={len(graph['attributes'])}")

    if args.push_graph:
        print("[4/4] 寫入 Neo4j…")
        neo = Neo4jClient(NEO4J.uri, (NEO4J.user, NEO4J.password))
        neo.upsert_graph(graph)
        neo.close()
        print("完成！")

if __name__ == "__main__":
    main()