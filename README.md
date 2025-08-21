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