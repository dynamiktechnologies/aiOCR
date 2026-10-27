"""Table detection and extraction module using PPStructureV3."""

import logging
import tempfile
import os
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_structure_pipeline = None


def get_structure_pipeline():
    """
    Lazy-load PPStructureV3 pipeline for table detection.
    Returns None if the pipeline cannot be initialized (e.g., missing models).
    """
    global _structure_pipeline
    if _structure_pipeline is not None:
        return _structure_pipeline

    try:
        from paddleocr import PPStructureV3
        logger.info("Loading PPStructureV3 pipeline for table detection...")
        _structure_pipeline = PPStructureV3(
            use_table_recognition=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        return _structure_pipeline
    except Exception as e:
        logger.warning(
            f"PPStructureV3 not available: {e}. "
            "Table detection is disabled. "
            "Install paddlex[ocr] and ensure models are downloaded."
        )
        return None


def _as_quad(box) -> Optional[list[list[float]]]:
    """Normalise a box to the API's polygon form [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].

    PaddleX reports axis-aligned boxes as a flat [x1, y1, x2, y2] list while
    OCR text blocks use 4 corner points; the response schema expects the latter.
    """
    if not box:
        return None
    try:
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = (float(v) for v in box)
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        quad = [[float(p[0]), float(p[1])] for p in box]
        return quad or None
    except (TypeError, ValueError, IndexError):
        return None


def detect_tables(
    image: Image.Image,
    text_blocks: list[dict],
) -> list[dict]:
    """
    Detect tables in the image and extract table cell data.

    Args:
        image: PIL Image object.
        text_blocks: List of text block dicts from OCR engine (unused when
                     PPStructureV3 handles its own recognition).

    Returns:
        List of table dicts, each with cells (row, col, text, bbox).
    """
    pipeline = get_structure_pipeline()
    if pipeline is None:
        return []

    tmp_path = None
    try:
        tmp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp_path.name)
        tmp_path.close()

        # predict() expects a path / ndarray / PIL image, not a file handle.
        results = pipeline.predict(tmp_path.name)
        tables = []

        for result in results:
            res_json = getattr(result, "json", None)
            if not isinstance(res_json, dict):
                continue
            res = res_json.get("res", res_json)
            if not isinstance(res, dict):
                continue

            # PP-StructureV3 reports document blocks under 'parsing_res_list';
            # a table block carries its predicted HTML in 'block_content'.
            page_tables = []
            blocks = res.get("parsing_res_list") or []
            for item in blocks:
                if not isinstance(item, dict):
                    continue
                if str(item.get("block_label", "")).lower() != "table":
                    continue
                cells = _parse_html_table(item.get("block_content") or "")
                if cells:
                    page_tables.append({
                        "cells": cells,
                        "bbox": _as_quad(item.get("block_bbox")),
                    })

            # Fallback: the raw table-recognition results (older/newer layouts).
            if not page_tables:
                for table_res in res.get("table_res_list") or []:
                    if not isinstance(table_res, dict):
                        continue
                    cells = _parse_html_table(
                        table_res.get("pred_html") or "",
                        table_res.get("cell_box_list") or table_res.get("cell_bboxes"),
                    )
                    if cells:
                        page_tables.append({"cells": cells, "bbox": None})

            tables.extend(page_tables)

        return tables

    except Exception as e:
        logger.error(f"Table detection failed: {e}")
        return []
    finally:
        if tmp_path:
            os.unlink(tmp_path.name)


def _parse_html_table(html: str, cell_bboxes: list = None) -> list[dict]:
    """Parse an HTML table string into a list of cell dicts."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed. Cannot parse HTML tables.")
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    cells = []
    cell_idx = 0
    for row_idx, tr in enumerate(table.find_all("tr")):
        for col_idx, td in enumerate(tr.find_all(["td", "th"])):
            text = td.get_text(strip=True)
            bbox = None
            if cell_bboxes and cell_idx < len(cell_bboxes):
                bbox = _as_quad(cell_bboxes[cell_idx])
            cell_idx += 1

            cells.append({
                "row": row_idx,
                "col": col_idx,
                "text": text,
                "bbox": bbox,
            })

    return cells
