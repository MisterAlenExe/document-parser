"""Configuration classes for document parser."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""

    tesseract_cmd: str = "tesseract"
    tessdata_prefix: Optional[str] = None
    languages: str = "rus+kaz+eng"
    dpi: int = 200
    psm_mode: int = 3  # Page segmentation mode


@dataclass
class ExtractorConfig:
    """Configuration for document extraction."""

    ocr_config: OCRConfig
    table_strategy: str = "lines_strict"
    fontsize_limit: int = 3
    force_text: bool = True
