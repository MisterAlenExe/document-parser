"""Configuration classes for document parser."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""

    tesseract_cmd: str = "tesseract"
    tessdata_prefix: Optional[str] = None
    languages: str = "rus+kaz+eng"
    dpi: int = 150
    psm_mode: int = 6
    max_workers: int = 4
    pdf_ocr_min_chars: int = 500
    pdf_ocr_min_chars_per_page: int = 150
    pdf_ocr_min_file_size_bytes: int = 200_000


@dataclass
class ExtractorConfig:
    """Configuration for document extraction."""

    ocr_config: OCRConfig
    table_strategy: str = "lines_strict"
    fontsize_limit: int = 3
    force_text: bool = True
