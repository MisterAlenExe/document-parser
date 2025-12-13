"""Configuration classes for document parser."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRConfig:
    """Configuration for OCR processing.
    
    All parameters are configurable to optimize for different environments
    (VPS, local machines, cloud servers, etc.) and use cases.
    
    Examples:
        >>> # Default configuration (optimized for 4GB RAM, 4 vCPU)
        >>> config = OCRConfig()
        
        >>> # High quality configuration for powerful systems
        >>> config = OCRConfig(dpi=300, max_workers=7)
        
        >>> # Fast processing for limited resources
        >>> config = OCRConfig(dpi=120, max_workers=2)
    """

    tesseract_cmd: str = "tesseract"
    """Path to tesseract binary. Default: "tesseract" (assumes in PATH)."""
    
    tessdata_prefix: Optional[str] = None
    """Optional path to tessdata directory. If None, uses system default."""
    
    languages: str = "rus+kaz+eng"
    """OCR languages in Tesseract format (e.g., "eng", "eng+fra", "rus+kaz+eng")."""
    
    dpi: int = 150
    """Image DPI for PDF rendering. Higher = better quality but slower and more memory.
    
    Recommended values:
    - 120: Fastest, lowest memory, acceptable quality
    - 150: Default, balanced speed/quality/memory
    - 200: Better quality, ~30% more memory
    - 300: Best quality, ~2x memory (not recommended for 4GB RAM)
    """
    
    psm_mode: int = 6
    """Page segmentation mode (0-13). Default: 6 (uniform block of text).
    
    Common modes:
    - 3: Fully automatic page segmentation (default for most cases)
    - 6: Uniform block of text (good for documents)
    - 11: Sparse text (for documents with few words)
    - 13: Raw line (treat image as single text line)
    """
    
    max_workers: int = 3
    """Number of parallel workers for OCR processing.
    
    Recommended: Leave 1 CPU core for system processes.
    - For 4 vCPU: use 3 (default)
    - For 8 vCPU: use 7
    - For limited RAM (<4GB): use 2 even with more CPUs
    """
    
    pdf_ocr_min_chars: int = 500
    """Minimum characters extracted from native PDF to skip OCR.
    If native extraction yields fewer chars, OCR will be triggered."""
    
    pdf_ocr_min_chars_per_page: int = 150
    """Minimum characters per page from native PDF to skip OCR.
    If average chars per page is lower, OCR will be triggered."""
    
    pdf_ocr_min_file_size_bytes: int = 200_000
    """Minimum file size (bytes) to consider OCR.
    Small files with little text are assumed to be text-based PDFs."""
    
    enable_image_preprocessing: bool = True
    """Enable image preprocessing before OCR.
    
    When enabled:
    - Converts images to grayscale (faster OCR, ~66% less memory)
    - Enhances contrast (better accuracy for scanned documents)
    
    Disable if you need color information or want maximum speed.
    """
    
    contrast_enhancement: float = 1.2
    """Contrast enhancement factor for image preprocessing.
    
    - 1.0: No enhancement
    - 1.2: Default, 20% contrast boost (good for scanned documents)
    - 1.5: Strong enhancement (for poor quality scans)
    - >2.0: May cause artifacts
    """
    
    use_oem_1: bool = True
    """Use Tesseract OEM 1 (LSTM engine only).
    
    - True: Faster, less memory, modern engine (recommended)
    - False: Uses default engine (may use legacy engine, slower)
    """


@dataclass
class ExtractorConfig:
    """Configuration for document extraction."""

    ocr_config: OCRConfig
    table_strategy: str = "lines_strict"
    fontsize_limit: int = 3
    force_text: bool = True
