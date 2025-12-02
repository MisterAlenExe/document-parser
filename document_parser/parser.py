"""High-level API for document parsing."""

import base64
import mimetypes
from pathlib import Path
from typing import Optional

from document_parser.config import OCRConfig
from document_parser.handler import DocumentHandler
from document_parser.models import DocumentExtractionResult


def parse_document(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
    ocr_config: Optional[OCRConfig] = None,
) -> DocumentExtractionResult:
    """Parse a document and extract text content.

    High-level convenience function that accepts either a file path or raw bytes.

    Args:
        file_path: Path to document file (alternative to file_bytes)
        file_bytes: Raw document bytes (alternative to file_path)
        file_name: Original filename (required if using file_bytes)
        mime_type: MIME type hint (optional, will be detected if not provided)
        ocr_config: OCR configuration (optional, uses defaults if not provided)

    Returns:
        DocumentExtractionResult with extracted text and metadata

    Raises:
        ValueError: If neither file_path nor file_bytes provided, or if file_bytes
            provided without file_name
        InvalidBase64Error: If base64 encoding fails
        UnsupportedTypeError: If document type is not supported
        ExtractionError: If text extraction fails
        DecodingError: If UTF-8 decoding fails (for text files)

    Examples:
        >>> # Parse from file path
        >>> result = parse_document(file_path="document.pdf")
        >>> print(result.text)

        >>> # Parse from bytes with custom OCR config
        >>> config = OCRConfig(languages="eng+fra", dpi=300)
        >>> with open("document.pdf", "rb") as f:
        ...     result = parse_document(
        ...         file_bytes=f.read(),
        ...         file_name="document.pdf",
        ...         ocr_config=config
        ...     )
    """
    # Validate inputs
    if file_path and file_bytes:
        raise ValueError("Provide either file_path or file_bytes, not both")

    if not file_path and not file_bytes:
        raise ValueError("Must provide either file_path or file_bytes")

    # Load from file path if provided
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        with open(path, "rb") as f:
            file_bytes = f.read()
        file_name = path.name

        # Guess mime_type from extension if not provided
        if not mime_type:
            guessed_type, _ = mimetypes.guess_type(str(path))
            if guessed_type:
                mime_type = guessed_type

    # Validate file_name is provided when using file_bytes
    if not file_name:
        raise ValueError("file_name is required when using file_bytes")

    # Convert bytes to base64
    encoded = base64.b64encode(file_bytes).decode("ascii")

    # Initialize handler with config
    handler = DocumentHandler(ocr_config=ocr_config)

    # Extract and return
    return handler.extract(
        encoded=encoded, mime_type=mime_type or "", file_name=file_name
    )
