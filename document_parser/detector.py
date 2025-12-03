"""Document type detection and validation."""

import mimetypes
from dataclasses import dataclass

from document_parser.logger import get_logger

logger = get_logger(__name__)


PDF_SIGNATURE = b"%PDF"
ZIP_SIGNATURE = b"PK\x03\x04"
PNG_SIGNATURE = b"\x89PNG"
OLE_SIGNATURE = b"\xd0\xcf\x11\xe0"  # Legacy Office container
JPEG_SIGNATURES = (b"\xff\xd8\xff\xdb", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1")


@dataclass
class DocumentDescriptor:
    mime_type: str
    file_name: str
    character_count: int = 0
    ocr_used: bool = False


class DocumentDetector:
    """Detects document type and validates supported formats."""

    def detect(
        self, file_bytes: bytes, mime_type: str, file_name: str
    ) -> DocumentDescriptor:
        original_mime_type = mime_type
        file_size = len(file_bytes)

        logger.debug(
            "Starting document type detection",
            extra_data={
                "file_name": file_name,
                "provided_mime_type": mime_type,
                "file_size_bytes": file_size,
            },
        )

        sniffed_type = self._sniff_mime(file_bytes)
        if sniffed_type:
            mime_type = sniffed_type
            logger.debug(
                "Detected MIME type from file signature",
                extra_data={
                    "file_name": file_name,
                    "sniffed_mime_type": sniffed_type,
                    "original_mime_type": original_mime_type,
                },
            )
        else:
            guessed, _ = mimetypes.guess_type(file_name)
            if guessed:
                mime_type = guessed
                logger.debug(
                    "Guessed MIME type from file extension",
                    extra_data={
                        "file_name": file_name,
                        "guessed_mime_type": guessed,
                        "original_mime_type": original_mime_type,
                    },
                )

        supported_types = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "application/vnd.ms-excel",
            "image/png",
            "image/jpeg",
            "image/jpg",
            "text/plain",
        }

        if mime_type not in supported_types:
            logger.warning(
                "Unsupported MIME type detected",
                extra_data={
                    "file_name": file_name,
                    "detected_mime_type": mime_type,
                    "original_mime_type": original_mime_type,
                },
            )
            raise ValueError(f"Unsupported mime type: {mime_type}")

        logger.info(
            "Document type detected successfully",
            extra_data={
                "file_name": file_name,
                "final_mime_type": mime_type,
                "original_mime_type": original_mime_type,
                "mime_type_changed": mime_type != original_mime_type,
                "file_size_bytes": file_size,
            },
        )

        return DocumentDescriptor(
            mime_type=mime_type,
            file_name=file_name,
        )

    @staticmethod
    def _sniff_mime(file_bytes: bytes) -> str | None:
        """Detect MIME type from file signature/magic bytes."""
        head = file_bytes[:4]
        if head.startswith(PDF_SIGNATURE):
            return "application/pdf"
        if head.startswith(ZIP_SIGNATURE):
            # DOCX files are ZIP archives
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if head.startswith(OLE_SIGNATURE):
            # Legacy Office container (.doc/.xls). Disambiguate via extension later.
            return None
        if head.startswith(PNG_SIGNATURE):
            return "image/png"
        for jpeg_sig in JPEG_SIGNATURES:
            if file_bytes.startswith(jpeg_sig):
                return "image/jpeg"
        return None
