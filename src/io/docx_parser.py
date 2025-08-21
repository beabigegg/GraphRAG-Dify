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
                section_id=f"table-{ti}",
                rev=rev,
                text=f"TABLE_ROW\nHEADERS: {headers}\nROW: {cells}"
            ))
    return chunks