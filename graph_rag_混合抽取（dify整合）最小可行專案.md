# GraphRAG-混合抽取（Dify整合）最小可行專案

> 適用：以**文字為主**、**圖片為輔**的混合抽取，將 OI/SOP 這類含中英段落、表格、示意圖的檔案，轉為**圖譜（Neo4j）+向量檢索（可選）**，並提供與 **Dify** 的 Chat / Workflow API 交互範例。

---

## 專案結構
```
GraphRAG-Dify/
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ src/
│  ├─ main.py
│  ├─ config.py
│  ├─ models/
│  │  └─ schema.py
│  ├─ utils/
│  │  ├─ regexes.py
│  │  ├─ normalization.py
│  │  └─ chunking.py
│  ├─ io/
│  │  ├─ docx_parser.py
│  │  ├─ pdf_renderer.py
│  │  └─ ocr.py
│  ├─ extraction/
│  │  ├─ prompts.py
│  │  └─ pipeline.py
│  ├─ dify/
│  │  └─ client.py
│  └─ graph/
│     └─ neo4j_client.py
└─ tests/
   └─ test_smoke.py
```

---

## README.md
```md
# GraphRAG + Dify Minimal Project

這個專案提供一個可直接運行的最小可行範例：
1. 解析 DOCX（段落、表格），必要時轉 PDF 並對頁面做截圖（圖片為輔）。
2. 依規則切段與表格行，產生「文字 +（可選）對應截圖」的混合輸入。
3. 呼叫多模態抽取（可接 Dify Chat / Workflow），輸出結構化 JSON（三元組）。
4. 正規化與去重，寫入 Neo4j 圖譜。

> 設計原則：**文字為主、圖片為輔**；圖片僅在版面承載語義（複雜表格/幾何公差/流程圖/不良照片）時附上，並作為證據與可追溯性。

## 安裝
```bash
uv venv && source .venv/bin/activate  # 或 python -m venv
uv pip install -e .                   # 若未使用 uv，可 `pip install -e .`
```

或：
```bash
pip install -e .
```

## 需求套件
- python-docx, pydantic, requests, python-dotenv, PyMuPDF (fitz), Pillow, neo4j, tqdm
- 可選：docx2pdf（Windows/含Word環境較穩）、pytesseract + tesseract-ocr（OCR）

## 環境變數（.env）
```
DIFY_BASE_URL=https://api.dify.ai
DIFY_API_KEY=your_api_key_here
DIFY_APP_TYPE=workflow   # 值為 chat 或 workflow
DIFY_RESPONSE_MODE=blocking  # 或 streaming
DIFY_USER_ID=panjit-user-001
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

## 快速開始
```bash
python -m src.main \
  --doc "./samples/EUTECTIC1610-OI01-03.docx" \
  --rev 03 \
  --section-hint "6.2" \
  --push-graph
```

流程：
1. 讀取 DOCX → 段落/表格切分與雙語對齊 → 產生 chunk（含可選圖片證據）
2. 每個 chunk 呼叫 `extraction.pipeline.run_extraction()`：
   - 若 Dify 為 workflow：先 `/files/upload`（必要時）→ `/workflows/run`
   - 若 Dify 為 chat：`/chat-messages`（可附 file ids）
3. 彙整 JSON → 正規化去重 → `graph.neo4j_client.upsert_graph()` 寫入

## 測試
```bash
pytest -q
```
```

---

## pyproject.toml
```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "graphrag-dify"
version = "0.1.0"
description = "GraphRAG minimal project with DOCX parsing and Dify integration"
authors = [{name = "PANJIT AI"}]
requires-python = ">=3.10"
dependencies = [
  "python-docx>=1.0.0",
  "pydantic>=2.6",
  "python-dotenv>=1.0.1",
  "requests>=2.31",
  "tqdm>=4.66",
  "neo4j>=5.21",
  "PyMuPDF>=1.24.9",
  "Pillow>=10.3.0"
]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

---

## src/config.py
```python
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DifyConfig:
    base_url: str = os.getenv("DIFY_BASE_URL", "https://api.dify.ai")
    api_key: str = os.getenv("DIFY_API_KEY", "")
    app_type: str = os.getenv("DIFY_APP_TYPE", "workflow")  # "chat" or "workflow"
    response_mode: str = os.getenv("DIFY_RESPONSE_MODE", "blocking")
    user_id: str = os.getenv("DIFY_USER_ID", "demo-user")

@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "neo4j")

DIFY = DifyConfig()
NEO4J = Neo4jConfig()
```

---

## src/models/schema.py
```python
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
```

---

## src/utils/regexes.py
```python
SECTION_RE = r"^(\d+(?:\.\d+){0,3})\s+"
CODE_RE = r"[A-Z]{1,3}-SP\d{3}|DB-PA\d{3}|DB-OC\d{3}|F-[A-Z]{2}\d{4}[A-Z0-9]?"
NUMUNIT_RE = r"(≤|≥|=|±)?\s*\d+(?:\.\d+)?\s*(um|µm|g|°C|%|mil|hrs?|hours?)"
FREQ_WORDS = [
    "每班","每兩小時","每天","開機","維修","換Type","Every shift","Every two hours","Daily"
]
```

---

## src/utils/normalization.py
```python
from __future__ import annotations
import re
from .regexes import CODE_RE

CODE_PAT = re.compile(CODE_RE)

def canonical(text: str) -> str:
    return re.sub(r"\s+"," ", text or "").strip().lower()

def normalize_codes(text: str) -> list[str]:
    return list({m.group(0) for m in re.finditer(CODE_PAT, text or "")})
```

---

## src/utils/chunking.py
```python
from __future__ import annotations
import re
from typing import Iterable, List
from ..models.schema import TextChunk, EvidenceImage
from .regexes import SECTION_RE

SEC_PAT = re.compile(SECTION_RE)

def pair_bilingual_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if i + 1 < len(lines):
            nxt = lines[i+1].strip()
            # 簡單啟發：英行包含大量 ASCII 與空白，中文含 CJK
            if is_cjk(cur) and not is_cjk(nxt):
                out.append(f"zh:{cur}\nen:{nxt}")
                i += 2
                continue
        out.append(cur)
        i += 1
    return [x for x in out if x]

CJK_RANGE = (0x4E00, 0x9FFF)

def is_cjk(s: str) -> bool:
    return any(CJK_RANGE[0] <= ord(ch) <= CJK_RANGE[1] for ch in s)


def split_sections(paras: Iterable[str], rev: str | None=None) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    buf: list[str] = []
    cur_sec: str | None = None
    def flush():
        nonlocal buf, cur_sec
        if not buf:
            return
        text = "\n".join(buf).strip()
        cid = f"sec-{cur_sec or 'body'}-{len(chunks)}"
        chunks.append(TextChunk(chunk_id=cid, section_id=cur_sec, rev=rev, text=text))
        buf = []
    for p in paras:
        m = re.match(SEC_PAT, p)
        if m:
            flush()
            cur_sec = m.group(1)
        buf.append(p)
    flush()
    return chunks

```

---

## src/io/docx_parser.py
```python
from __future__ import annotations
from typing import List
from docx import Document
from ..utils.chunking import pair_bilingual_lines, split_sections
from ..models.schema import TextChunk


def load_docx_text_chunks(path: str, rev: str | None=None) -> List[TextChunk]:
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]
    lines = pair_bilingual_lines(paras)
    chunks = split_sections(lines, rev=rev)
    # 表格：按行輸出為附加 chunk（行級利於結構化）
    for ti, table in enumerate(doc.tables):
        headers = [c.text.strip() for c in table.rows[0].cells] if table.rows else []
        for ri, row in enumerate(table.rows[1:], start=1):
            cells = [c.text.strip() for c in row.cells]
            text = " | ".join([f"{h}:{v}" for h, v in zip(headers, cells)])
            chunks.append(TextChunk(
                chunk_id=f"table-{ti}-row-{ri}",
                section_id=None,
                rev=rev,
                text=f"TABLE_ROW\nHEADERS: {headers}\nROW: {cells}"
            ))
    return chunks
```

---

## src/io/pdf_renderer.py
```python
from __future__ import annotations
from typing import List
import fitz  # PyMuPDF
from PIL import Image
from ..models.schema import EvidenceImage


def render_page_images(pdf_path: str, pages: list[int] | None=None, zoom: float=2.0) -> List[EvidenceImage]:
    doc = fitz.open(pdf_path)
    if pages is None:
        pages = list(range(len(doc)))
    out: List[EvidenceImage] = []
    for pno in pages:
        page = doc.load_page(pno)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_path = f".cache/page-{pno+1}.png"
        Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(img_path)
        out.append(EvidenceImage(image_id=f"page-{pno+1}", page=pno+1, caption=f"page {pno+1}", purpose="page",))
    return out
```

---

## src/io/ocr.py
```python
from __future__ import annotations
try:
    import pytesseract
    from PIL import Image
except Exception:  # 可選依賴
    pytesseract = None
    Image = None

def ocr_image(path: str) -> str:
    if not pytesseract or not Image:
        return ""
    return pytesseract.image_to_string(Image.open(path), lang="eng+chi_tra")
```

---

## src/extraction/prompts.py
```python
MULTIMODAL_SYSTEM = (
    "你是一個多模態製程文件抽取器。輸入包含一段文字與0~N張對應截圖。"
    "請輸出嚴格JSON：entities[], relations[], attributes[], refs[], "
    "source{section_id,page_hint,rev}, evidence{images[]}, notes{conflicts[]}。"
    "若圖片與文字衝突，以文字為準，並將差異放入 notes.conflicts[]。"
)

TRIPLE_EXTRACTION_USER_TPL = (
    "文字:\n{text}\n\n"
    "圖片IDs(可選): {image_ids}\n"
    "請依Ontology抽取三元組，確保數值包含value/unit/operator，頻率標準化。"
)
```

---

## src/extraction/pipeline.py
```python
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
            "images": [img.image_id for img in ch.images],
            "section_id": ch.section_id,
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
```

---

## src/dify/client.py
```python
from __future__ import annotations
from typing import Dict, Any, Optional
import requests

class DifyClient:
    def __init__(self, base_url: str, api_key: str, app_type: str = "workflow", response_mode: str = "blocking", user_id: str = "demo-user"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.app_type = app_type
        self.response_mode = response_mode
        self.user_id = user_id
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })

    # --- 檔案上傳（可供 Workflow / Chat 使用） ---
    def upload_file(self, filepath: str) -> str:
        url = f"{self.base_url}/v1/files/upload"
        with open(filepath, "rb") as f:
            files = {"file": (filepath, f)}
            data = {"user": self.user_id}
            r = self.session.post(url, files=files, data=data, timeout=120)
        r.raise_for_status()
        return r.json()["id"]

    # --- Workflow 觸發 ---
    def run_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/workflows/run"
        payload = {
            "inputs": inputs,
            "response_mode": self.response_mode,
            "user": self.user_id,
        }
        r = self.session.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    # --- Chat 訊息 ---
    def chat_messages(self, inputs: Dict[str, Any], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat-messages"
        payload = {
            "inputs": inputs,
            "response_mode": self.response_mode,
            "user": self.user_id,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        r = self.session.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    # --- 封裝：抽取 ---
    def extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 視 Dify 應用設計：
        #   - Workflow：將 {text, images, section_id, rev} 當作 workflow inputs
        #   - Chat：將上述內容塞入 inputs/text 或 fields
        if self.app_type == "workflow":
            return self.run_workflow(inputs=payload)
        return self.chat_messages(inputs={"text": payload})
```

---

## src/graph/neo4j_client.py
```python
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
```

---

## src/main.py
```python
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
```

---

## tests/test_smoke.py
```python
from src.utils.normalization import canonical, normalize_codes

def test_canonical():
    assert canonical("  A  B  ") == "a b"

def test_code_extract():
    s = "見 DB-SP015 與 F-RD09M4"
    codes = normalize_codes(s)
    assert "DB-SP015" in codes and "F-RD09M4" in codes
```

---

## .env.example
```env
DIFY_BASE_URL=https://api.dify.ai
DIFY_API_KEY=sk-xxxxx
DIFY_APP_TYPE=workflow
DIFY_RESPONSE_MODE=blocking
DIFY_USER_ID=panjit-user-001
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

---

### 備註與說明
- 若你的 Dify 應用是 **Workflow**：請在工作流中定義對應的 `inputs`（例如 `text`, `images`, `section_id`, `rev`），`client.extract()` 會將這些欄位直接帶入 `/v1/workflows/run`。
- 若你的 Dify 應用是 **Chat**：`client.extract()` 會把 payload 塞進 `inputs.text`，你可在應用 Prompt 中解析 JSON 並回嚴格 JSON。
- 需要檔案上傳時，請先呼叫 `upload_file()`，並在 `inputs.files` 或工作流節點中引用 `upload_file_id`。
- 圖片證據僅作為輔助與可追溯；主要三元組以文字抽取為準。
- Neo4j schema 在此為簡化示例；實務可將實體類型拆成不同 Label（ProcessStep/InspectionItem/...），屬性以關係表示更佳。
```

