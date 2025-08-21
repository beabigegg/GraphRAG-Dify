from __future__ import annotations
import re
from .regexes import CODE_RE

CODE_PAT = re.compile(CODE_RE)

def canonical(text: str) -> str:
    return re.sub(r"\s+"," ", text or "").strip().lower()

def normalize_codes(text: str) -> list[str]:
    return list({m.group(0) for m in re.finditer(CODE_PAT, text or "")})