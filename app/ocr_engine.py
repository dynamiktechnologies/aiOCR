"""Core OCR engine wrapping PaddleOCR 3.x."""

import logging
import tempfile
import os
from typing import Optional

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# Cache OCR instances per language to avoid reloading models
_ocr_cache: dict[str, PaddleOCR] = {}


def get_ocr(lang: str = "en") -> PaddleOCR:
    """Get or create a PaddleOCR instance for the given language."""
    if lang not in _ocr_cache:
        logger.info(f"Initializing PaddleOCR for language: {lang}")
        _ocr_cache[lang] = PaddleOCR(lang=lang)
    return _ocr_cache[lang]


def _save_temp_image(image: Image.Image) -> str:
    """Save PIL Image to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    tmp.close()
    return tmp.name


def extract_text(
    image: Image.Image,
    lang: str = "en",
) -> list[dict]:
    """
    Extract text blocks from an image.

    Args:
        image: PIL Image object.
        lang: PaddleOCR language code.

    Returns:
        List of dicts with keys: text, confidence, bbox.
    """
    ocr = get_ocr(lang)
    tmp_path = _save_temp_image(image)

    try:
        results = ocr.predict(tmp_path)
        blocks = []

        for result in results:
            res_json = result.json
            res = res_json.get("res", res_json)

            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", [])

            for i, text in enumerate(texts):
                confidence = float(scores[i]) if i < len(scores) else 0.0
                bbox = polys[i] if i < len(polys) else []
                blocks.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
                })

        return blocks
    finally:
        os.unlink(tmp_path)


def extract_full_text(
    image: Image.Image,
    lang: str = "en",
) -> str:
    """Extract full text from image as a single string."""
    blocks = extract_text(image, lang)
    return "\n".join(block["text"] for block in blocks)
