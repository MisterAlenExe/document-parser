"""PyMuPDF4LLM-based document extractor with Tesseract OCR support."""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pymupdf4llm
import pytesseract
from docx import Document
from PIL import Image

from document_parser.config import OCRConfig
from document_parser.exceptions import ExtractionError
from document_parser.logger import Timer, get_logger

logger = get_logger(__name__)


class PyMuPDFExtractor:
    """Lightweight document extractor using PyMuPDF4LLM.

    No AI/ML dependencies - uses PyMuPDF for PDFs, python-docx for DOCX,
    and Tesseract OCR for images. Outputs markdown format optimized for LLMs.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize extractor with configuration.

        Args:
            config: OCR configuration. If None, uses defaults.
        """
        self.config = config or OCRConfig()

        # Set Tesseract environment variables if provided
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
        if self.config.tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = self.config.tessdata_prefix

        logger.info(
            "Initializing PyMuPDFExtractor",
            extra_data={
                "tesseract_cmd": self.config.tesseract_cmd,
                "languages": self.config.languages,
                "dpi": self.config.dpi,
            },
        )

    def extract(self, file_bytes: bytes, file_name: str) -> str:
        """Extract text from document.

        Args:
            file_bytes: Raw file bytes
            file_name: Original filename (used for format detection)

        Returns:
            Extracted text in Markdown format

        Raises:
            ExtractionError: If extraction fails
        """
        suffix = Path(file_name).suffix.lower()

        logger.debug(
            "Starting document extraction",
            extra_data={
                "file_name": file_name,
                "file_extension": suffix,
                "file_size_bytes": len(file_bytes),
            },
        )

        try:
            if suffix == ".pdf":
                return self._extract_pdf(file_bytes, file_name)
            elif suffix in [".docx", ".doc"]:
                return self._extract_docx(file_bytes, file_name)
            elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                return self._extract_image(file_bytes, file_name)
            else:
                # Fallback: treat as PDF
                logger.warning(
                    "Unknown file extension, attempting PDF extraction",
                    extra_data={
                        "file_name": file_name,
                        "file_extension": suffix,
                    },
                )
                return self._extract_pdf(file_bytes, file_name)

        except Exception as exc:
            logger.error(
                "Document extraction failed",
                extra_data={
                    "file_name": file_name,
                    "file_extension": suffix,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                exc_info=True,
            )
            raise ExtractionError(f"Failed to extract document: {exc}") from exc

    def _extract_pdf(self, file_bytes: bytes, file_name: str = "unknown.pdf") -> str:
        """Extract from PDF using PyMuPDF4LLM."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            # PyMuPDF4LLM handles:
            # - Native text extraction (fast path)
            # - Table detection and markdown formatting
            # - Structured output (headers, lists, etc.)
            # Note: Does NOT include OCR - scanned PDFs will return empty text
            with Timer("pdf_native_extraction") as native_timer:
                md_text = pymupdf4llm.to_markdown(
                    tmp_path,
                    # Table extraction
                    table_strategy="lines_strict",  # Detect tables with visible lines
                    # Text extraction
                    force_text=True,  # Extract text even over images
                    # Image handling
                    write_images=False,  # Don't save images separately
                    ignore_images=True,  # Skip decorative images
                    # Text processing
                    ignore_code=False,  # Preserve code formatting
                    fontsize_limit=3,  # Ignore text smaller than 3pt
                )

            text = md_text.strip()
            char_count = len(text)

            logger.debug(
                "PDF native text extraction completed",
                extra_data={
                    "file_name": file_name,
                    "characters_extracted": char_count,
                    "extraction_time_ms": native_timer.get_elapsed_ms(),
                },
            )

            # If no text extracted, it's likely a scanned/image-only PDF - use OCR
            if not text:
                logger.info(
                    "No native text found in PDF, attempting OCR",
                    extra_data={
                        "file_name": file_name,
                    },
                )

                with Timer("pdf_ocr") as ocr_timer:
                    text = self._ocr_pdf(tmp_path, file_name)

                logger.info(
                    "OCR extraction completed",
                    extra_data={
                        "file_name": file_name,
                        "characters_extracted": len(text),
                        "ocr_time_ms": ocr_timer.get_elapsed_ms(),
                    },
                )

            return text

        finally:
            # Cleanup temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def _ocr_pdf(self, pdf_path: str, file_name: str = "unknown.pdf") -> str:
        """Perform OCR on a PDF using Tesseract.

        Args:
            pdf_path: Path to the PDF file
            file_name: Original file name for logging

        Returns:
            Extracted text from all pages
        """
        try:
            # Open PDF and convert pages to images
            pdf_document = fitz.open(pdf_path)
            page_count = len(pdf_document)
            all_text = []

            logger.debug(
                "Starting OCR on PDF pages",
                extra_data={
                    "file_name": file_name,
                    "page_count": page_count,
                },
            )

            for page_num in range(page_count):
                with Timer(f"ocr_page_{page_num}") as page_timer:
                    # Render page to image at configured DPI
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(dpi=self.config.dpi)

                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))

                    # Perform OCR with configured languages
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=self.config.languages,
                        config=f"--psm {self.config.psm_mode}",
                    )

                    if page_text.strip():
                        all_text.append(page_text.strip())

                    logger.info(
                        f"OCR completed for page {page_num + 1}/{page_count}",
                        extra_data={
                            "file_name": file_name,
                            "page_number": page_num + 1,
                            "characters_extracted": len(page_text.strip()),
                            "ocr_time_ms": page_timer.get_elapsed_ms(),
                        },
                    )

            pdf_document.close()

            total_text = "\n\n".join(all_text)

            logger.info(
                "PDF OCR completed for all pages",
                extra_data={
                    "file_name": file_name,
                    "page_count": page_count,
                    "pages_with_text": len(all_text),
                    "total_characters": len(total_text),
                },
            )

            return total_text

        except Exception as e:
            logger.error(
                "OCR failed for PDF",
                extra_data={
                    "file_name": file_name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
                exc_info=True,
            )
            return ""

    def _extract_docx(self, file_bytes: bytes, file_name: str = "unknown.docx") -> str:
        """Extract from DOCX using python-docx."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            with Timer("docx_extraction") as timer:
                doc = Document(tmp_path)

                # Extract paragraphs
                paragraphs = []
                for para in doc.paragraphs:
                    text = para.text.strip()
                    if text:
                        paragraphs.append(text)

                # Extract tables (markdown format)
                tables = []
                for table in doc.tables:
                    # Build markdown table
                    rows = []
                    for i, row in enumerate(table.rows):
                        cells = [cell.text.strip() for cell in row.cells]
                        rows.append(" | ".join(cells))

                        # Add header separator after first row
                        if i == 0:
                            separator = " | ".join(["---"] * len(cells))
                            rows.append(separator)

                    if rows:
                        tables.append("\n".join(rows))

                # Combine paragraphs and tables
                parts = paragraphs
                if tables:
                    parts.extend(tables)

                result = "\n\n".join(parts)

            logger.debug(
                "DOCX extraction completed",
                extra_data={
                    "file_name": file_name,
                    "paragraph_count": len(paragraphs),
                    "table_count": len(tables),
                    "characters_extracted": len(result),
                    "extraction_time_ms": timer.get_elapsed_ms(),
                },
            )

            return result

        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def _extract_image(self, file_bytes: bytes, file_name: str = "unknown.jpg") -> str:
        """Extract from image using Tesseract OCR directly."""
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(file_bytes))
            image_format = image.format
            image_size = image.size

            logger.debug(
                "Starting OCR on image",
                extra_data={
                    "file_name": file_name,
                    "image_format": image_format,
                    "image_width": image_size[0],
                    "image_height": image_size[1],
                },
            )

            # Perform OCR with configured languages
            with Timer("image_ocr") as timer:
                text = pytesseract.image_to_string(
                    image,
                    lang=self.config.languages,
                    config=f"--psm {self.config.psm_mode}",
                )

            result = text.strip()

            logger.info(
                "Image OCR completed",
                extra_data={
                    "file_name": file_name,
                    "image_format": image_format,
                    "image_dimensions": f"{image_size[0]}x{image_size[1]}",
                    "characters_extracted": len(result),
                    "ocr_time_ms": timer.get_elapsed_ms(),
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Image OCR failed",
                extra_data={
                    "file_name": file_name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ExtractionError(f"Failed to extract text from image: {e}")
