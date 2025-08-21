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