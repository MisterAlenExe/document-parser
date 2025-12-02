"""Data models for document parser."""

from dataclasses import dataclass


@dataclass
class DocumentExtractionResult:
    """Result of document extraction."""

    text: str  # Markdown format from extractor
    mime_type: str
    file_name: str
    character_count: int
    ocr_used: bool
