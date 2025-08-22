from __future__ import annotations

"""Command line entry point for the GraphRAG‑Dify project.

This script extends the original minimal project by adding vector support.
Each chunk produced from the input document is embedded using a local
embedding model (via Ollama) and then stored into Neo4j along with the
structured extraction results.  The resulting graph therefore contains
rich entity/relationship data and a vector index for semantic search.
"""

import argparse
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm

from .config import DIFY, NEO4J
from .io.docx_parser import load_docx_text_chunks
from .dify.client import DifyClient
from .extraction.pipeline import run_extraction, merge_and_dedup
from .graph.neo4j_client import Neo4jClient


def _embed_text(text: str, model: str = "mxbai-embed-large", host: str = "http://localhost:11434") -> List[float]:
    """Call a local Ollama embedding model to generate a vector for the given text.

    Args:
        text: The input text to embed.
        model: Name of the embedding model to use.  Defaults to ``mxbai-embed-large``.
        host: Base URL of the Ollama service.  Defaults to ``http://localhost:11434``.

    Returns:
        A list of floating point numbers representing the embedding.
    """
    url = f"{host}/api/embed"
    payload = {"model": model, "input": text}
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc
    # The API may return either 'embedding' or 'embeddings'.  Normalise here.
    if isinstance(data, dict):
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
            return data["embeddings"][0]
    raise ValueError(f"Unexpected embedding response format: {data}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="path to DOCX")
    ap.add_argument("--rev", default=None, help="revision identifier (optional)")
    ap.add_argument("--section-hint", default=None)
    ap.add_argument("--push-graph", action="store_true")
    ap.add_argument(
        "--cache-db",
        default=None,
        help=(
            "Path to a SQLite database used to cache Dify extraction results. "
            "If omitted, caching is disabled."
        ),
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output during extraction (cache hits/misses, parse errors).",
    )
    ap.add_argument(
        "--embed-model",
        default="mxbai-embed-large",
        help="Name of the local embedding model served by Ollama.",
    )
    ap.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Base URL of the Ollama service used for embedding.",
    )
    args = ap.parse_args()

    docx_path = Path(args.doc)
    assert docx_path.exists(), f"Not found: {docx_path}"  # noqa: WPS305

    print("[1/5] 解析 DOCX…")
    chunks = load_docx_text_chunks(str(docx_path), rev=args.rev)
    print(f"已生成 {len(chunks)} 個 chunk。")

    # Step 1: compute embeddings for each chunk
    print("[2/5] 生成嵌入向量…")
    for ch in tqdm(chunks):
        ch.embedding = _embed_text(ch.text, model=args.embed_model, host=args.ollama_host)

    # Step 2: call Dify for extraction
    print("[3/5] 呼叫 Dify 進行抽取…")
    dify = DifyClient(
        DIFY.base_url,
        DIFY.api_key,
        DIFY.app_type,
        DIFY.response_mode,
        DIFY.user_id,
    )
    extracted: list = []
    for ch in tqdm(chunks):
        ex = run_extraction(
            [ch],
            dify,
            cache_db_path=args.cache_db,
            debug=args.debug,
        )
        extracted.extend(ex)

    # Step 3: merge and deduplicate extraction results
    print("[4/5] 合併與去重…")
    graph = merge_and_dedup(extracted)
    print(
        f"entities={len(graph['entities'])}, relations={len(graph['relations'])}, attributes={len(graph['attributes'])}"
    )

    # Step 4: push graph and vectors into Neo4j
    if args.push_graph:
        print("[5/5] 寫入 Neo4j…")
        neo = Neo4jClient(NEO4J.uri, (NEO4J.user, NEO4J.password))
        neo.upsert_graph(graph)
        # Use a fixed index name; adjust as needed
        neo.upsert_chunks(chunks, index_name="chunk_vector_index")
        neo.close()
        print("完成！")


if __name__ == "__main__":  # pragma: no cover
    main()