"""Document handler orchestration."""

import base64
import binascii
from pathlib import Path
from typing import Optional

from document_parser.config import OCRConfig
from document_parser.detector import DocumentDescriptor, DocumentDetector
from document_parser.exceptions import (
    DecodingError,
    ExtractionError,
    InvalidBase64Error,
    UnsupportedTypeError,
)
from document_parser.extractor import PyMuPDFExtractor
from document_parser.logger import Timer, get_logger
from document_parser.models import DocumentExtractionResult

logger = get_logger(__name__)


class DocumentHandler:
    def __init__(
        self,
        detector: Optional[DocumentDetector] = None,
        extractor: Optional[PyMuPDFExtractor] = None,
        ocr_config: Optional[OCRConfig] = None,
    ) -> None:
        """Initialize document handler.

        Args:
            detector: Document type detector. If None, creates default.
            extractor: Document extractor. If None, creates default with ocr_config.
            ocr_config: OCR configuration for the extractor. Only used if extractor is None.
        """
        self.detector = detector or DocumentDetector()
        self.extractor = extractor or PyMuPDFExtractor(config=ocr_config)

    def decode_file(self, encoded: str) -> bytes:
        """Decode base64-encoded file.

        Args:
            encoded: Base64-encoded file content

        Returns:
            Decoded bytes

        Raises:
            InvalidBase64Error: If decoding fails
        """
        try:
            with Timer("base64_decode") as timer:
                decoded = base64.b64decode(encoded, validate=True)

            logger.debug(
                "Successfully decoded base64 file",
                extra_data={
                    "decoded_size_bytes": len(decoded),
                    "decode_time_ms": timer.get_elapsed_ms(),
                },
            )
            return decoded
        except (ValueError, binascii.Error) as exc:
            logger.error(
                "Failed to decode base64 string",
                extra_data={
                    "error_type": type(exc).__name__,
                    "encoded_length": len(encoded) if encoded else 0,
                },
            )
            raise InvalidBase64Error(
                "file_base64 must be a valid base64 string"
            ) from exc

    def extract(
        self, encoded: str, mime_type: str, file_name: str
    ) -> DocumentExtractionResult:
        """Extract text from encoded document.

        Args:
            encoded: Base64-encoded file content
            mime_type: MIME type hint
            file_name: Original filename

        Returns:
            DocumentExtractionResult with extracted text and metadata

        Raises:
            InvalidBase64Error: If base64 decoding fails
            UnsupportedTypeError: If document type is not supported
            ExtractionError: If text extraction fails
            DecodingError: If UTF-8 decoding fails (for text files)
        """
        file_bytes = self.decode_file(encoded)

        # Detection phase
        with Timer("detection") as detect_timer:
            try:
                descriptor = self.detector.detect(
                    file_bytes=file_bytes, mime_type=mime_type, file_name=file_name
                )
            except ValueError as exc:
                logger.warning(
                    "Document detection failed - unsupported type",
                    extra_data={
                        "file_name": file_name,
                        "mime_type": mime_type,
                        "error": str(exc),
                    },
                )
                raise UnsupportedTypeError(str(exc)) from exc

        logger.debug(
            "Document detection completed",
            extra_data={
                "file_name": file_name,
                "detected_mime_type": descriptor.mime_type,
                "detection_time_ms": detect_timer.get_elapsed_ms(),
            },
        )

        # Extraction phase
        with Timer("extraction") as extract_timer:
            # Special handling for plain text files
            if descriptor.mime_type == "text/plain":
                try:
                    text = file_bytes.decode("utf-8")
                    descriptor.character_count = len(text)
                    descriptor.ocr_used = False

                    logger.info(
                        "Extracted text from plain text file",
                        extra_data={
                            "file_name": file_name,
                            "character_count": len(text),
                            "extraction_time_ms": extract_timer.get_elapsed_ms(),
                        },
                    )
                except UnicodeDecodeError as exc:
                    logger.error(
                        "Failed to decode plain text file as UTF-8",
                        extra_data={
                            "file_name": file_name,
                            "file_size_bytes": len(file_bytes),
                        },
                    )
                    raise DecodingError(
                        "Unable to decode text file (not valid UTF-8)"
                    ) from exc
            else:
                # Extract with PyMuPDFExtractor (unified for PDF/DOCX/images)
                try:
                    text = self.extractor.extract(file_bytes, file_name)
                    descriptor.character_count = len(text)
                    descriptor.ocr_used = getattr(
                        self.extractor, "last_ocr_used", False
                    ) or self._detect_ocr_usage(file_name, text)

                    logger.info(
                        "Successfully extracted text from document",
                        extra_data={
                            "file_name": file_name,
                            "mime_type": descriptor.mime_type,
                            "character_count": len(text),
                            "ocr_used": descriptor.ocr_used,
                            "extraction_time_ms": extract_timer.get_elapsed_ms(),
                        },
                    )
                except Exception as exc:
                    logger.error(
                        "Document extraction failed",
                        extra_data={
                            "file_name": file_name,
                            "mime_type": descriptor.mime_type,
                            "error": str(exc),
                            "extraction_time_ms": extract_timer.get_elapsed_ms(),
                        },
                    )
                    raise ExtractionError(str(exc)) from exc

        if not text:
            logger.warning(
                "No text content extracted from document",
                extra_data={
                    "file_name": file_name,
                    "mime_type": descriptor.mime_type,
                    "file_size_bytes": len(file_bytes),
                },
            )
            raise ExtractionError(
                "Unable to extract text content from the provided file"
            )

        return DocumentExtractionResult(
            text=text,
            mime_type=descriptor.mime_type,
            file_name=descriptor.file_name,
            character_count=descriptor.character_count,
            ocr_used=descriptor.ocr_used,
        )

    def _detect_ocr_usage(self, file_name: str, text: str) -> bool:
        """Heuristic: images always use OCR, PDFs may use OCR if text is short."""
        suffix = Path(file_name).suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return True
        # For PDFs: if text is very short or empty, OCR was likely used
        # This is a best-effort heuristic
        return suffix == ".pdf" and len(text.strip()) < 100
