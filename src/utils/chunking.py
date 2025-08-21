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