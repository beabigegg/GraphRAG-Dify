from __future__ import annotations
import re
from typing import Iterable, List
from ..models.schema import TextChunk, EvidenceImage
from .regexes import SECTION_RE

SEC_PAT = re.compile(SECTION_RE)

def split_with_overlap(
    lines: Iterable[str],
    rev: str | None = None,
    window_size: int = 10,
    overlap: int = 3,
) -> List[TextChunk]:
    """
    將行列表以滑動窗口方式切分，每個區塊有 overlap 行與前一區塊重疊。

    Args:
        lines: 經過 pair_bilingual_lines() 等處理後的行列表。
        rev: 版本號，傳入 TextChunk.rev。
        window_size: 每個區塊包含的行數。
        overlap: 區塊間重疊的行數。

    Returns:
        一組 TextChunk 物件，每個 chunk_id 按序號標記。
    """
    lines = [ln.strip() for ln in lines if ln.strip()]
    chunks = []
    start = 0
    idx = 0
    step = max(1, window_size - overlap)
    while start < len(lines):
        end = start + window_size
        window = "\n".join(lines[start:end])
        chunk_id = f"win-{idx}"
        chunks.append(TextChunk(chunk_id=chunk_id, section_id=None, rev=rev, text=window))
        idx += 1
        # 下一個窗口起點往前移 step 行，留出 overlap 重疊
        start += step
    return chunks

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