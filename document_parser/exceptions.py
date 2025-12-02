"""Custom exceptions for document parser."""


class DocumentParserError(Exception):
    """Base exception for document parser errors."""

    pass


class InvalidBase64Error(DocumentParserError):
    """Raised when base64 decoding fails."""

    pass


class UnsupportedTypeError(DocumentParserError):
    """Raised when document type is not supported."""

    pass


class ExtractionError(DocumentParserError):
    """Raised when text extraction fails."""

    pass


class DecodingError(DocumentParserError):
    """Raised when document decoding fails (e.g., UTF-8)."""

    pass
