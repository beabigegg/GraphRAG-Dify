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