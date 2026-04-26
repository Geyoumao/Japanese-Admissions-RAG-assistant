from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from app.text import normalize_text


@dataclass(slots=True)
class ParsedPage:
    page_number: int
    text: str
    used_ocr: bool = False


class PDFParser:
    def __init__(self, ocr_text_threshold: int = 80, ocr_language: str = "japan") -> None:
        self.ocr_text_threshold = ocr_text_threshold
        self.ocr_language = ocr_language
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.ocr_language)
        return self._ocr

    def _ocr_page(self, page) -> str:
        import fitz
        import numpy as np
        from PIL import Image

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        results = self._get_ocr().ocr(np.array(image))
        lines: list[str] = []
        for block in results or []:
            for item in block or []:
                text = item[1][0] if len(item) > 1 else ""
                if text:
                    lines.append(text)
        return normalize_text("\n".join(lines))

    def extract_pages(self, pdf_path: str | Path) -> list[ParsedPage]:
        import fitz

        parsed: list[ParsedPage] = []
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, start=1):
                text = normalize_text(page.get_text("text"))
                used_ocr = False
                if len(text) < self.ocr_text_threshold:
                    try:
                        ocr_text = self._ocr_page(page)
                    except Exception:
                        ocr_text = ""
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        used_ocr = bool(ocr_text)
                parsed.append(ParsedPage(page_number=index, text=text, used_ocr=used_ocr))
        return parsed
