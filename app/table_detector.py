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

        results = pipeline.predict(tmp_path)
        tables = []

        for result in results:
            res_json = result.json
            res = res_json.get("res", res_json)

            # Extract table results from the layout parsing
            layout_results = res.get("layout_parsing_res", res.get("pruned_result", []))
            if isinstance(layout_results, dict):
                layout_results = [layout_results]

            for item in layout_results:
                label = item.get("label", "")
                if label != "table":
                    continue

                table_res = item.get("table_res", {})
                if not table_res:
                    continue

                pred_html = table_res.get("pred_html", "")
                cell_bboxes = table_res.get("cell_bboxes", [])

                cells = []
                # Parse HTML table to extract cells
                if pred_html:
                    cells = _parse_html_table(pred_html, cell_bboxes)

                if cells:
                    tables.append({
                        "cells": cells,
                        "bbox": item.get("bbox", None),
                    })

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
    for row_idx, tr in enumerate(table.find_all("tr")):
        for col_idx, td in enumerate(tr.find_all(["td", "th"])):
            text = td.get_text(strip=True)
            bbox = None
            cell_idx = row_idx * len(tr.find_all(["td", "th"])) + col_idx
            if cell_bboxes and cell_idx < len(cell_bboxes):
                bbox = cell_bboxes[cell_idx]

            cells.append({
                "row": row_idx,
                "col": col_idx,
                "text": text,
                "bbox": bbox,
            })

    return cells
