# Document Parser

A Python library for parsing documents with OCR support. Extracts text from PDF, DOCX, and image files using PyMuPDF4LLM and Tesseract OCR.

## Features

- 📄 **PDF Support**: Native text extraction with OCR fallback for scanned documents
- 📝 **DOCX Support**: Extract text and tables from Word documents
- 🖼️ **Image Support**: OCR for PNG, JPEG, TIFF, and BMP images
- 🌍 **Multi-language OCR**: Supports Russian, Kazakh, English, and more
- ⚡ **Fast**: Uses PyMuPDF for native PDF text extraction
- 🎯 **Simple API**: High-level `parse_document()` function and low-level classes

## Installation

### From Git Repository (SSH)

```bash
pip install git+ssh://git@github.com/your-org/document-parser.git
```

### For Development

```bash
git clone git@github.com:your-org/document-parser.git
cd document-parser
pip install -e .
```

## System Requirements

- Python >= 3.10
- Tesseract OCR binary (required for OCR functionality)

### Install Tesseract

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-kaz
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

## Quick Start

### Basic Usage

```python
from document_parser import parse_document

# Parse a document from file path
result = parse_document(file_path="document.pdf")

print(result.text)              # Extracted text in Markdown format
print(result.mime_type)         # Detected MIME type
print(result.character_count)   # Number of characters extracted
print(result.ocr_used)          # Whether OCR was used
```

### Parse from Bytes

```python
from document_parser import parse_document

with open("document.pdf", "rb") as f:
    file_bytes = f.read()

result = parse_document(
    file_bytes=file_bytes,
    file_name="document.pdf"
)

print(result.text)
```

### Custom OCR Configuration

```python
from document_parser import parse_document, OCRConfig

# Configure OCR settings
config = OCRConfig(
    languages="eng+fra",        # English + French
    dpi=300,                    # Higher resolution for better quality
    tesseract_cmd="tesseract"   # Path to tesseract binary
)

result = parse_document(
    file_path="document.pdf",
    ocr_config=config
)
```

### Low-Level API

```python
import base64
from document_parser import DocumentHandler, OCRConfig

# Configure OCR
config = OCRConfig(languages="rus+kaz+eng", dpi=200)

# Create handler
handler = DocumentHandler(ocr_config=config)

# Read and encode file
with open("document.pdf", "rb") as f:
    file_bytes = f.read()
encoded = base64.b64encode(file_bytes).decode("ascii")

# Extract text
result = handler.extract(
    encoded=encoded,
    mime_type="application/pdf",
    file_name="document.pdf"
)

print(result.text)
```

## Supported File Formats

| Format | MIME Type | Detection | Extraction |
|--------|-----------|-----------|------------|
| PDF (text) | `application/pdf` | Magic bytes | PyMuPDF native |
| PDF (scanned) | `application/pdf` | Magic bytes | Tesseract OCR |
| DOCX | `application/vnd...wordprocessingml.document` | ZIP signature | python-docx |
| PNG | `image/png` | Magic bytes | Tesseract OCR |
| JPEG | `image/jpeg` | Magic bytes | Tesseract OCR |
| Text | `text/plain` | Extension | UTF-8 decode |

## Configuration

### OCRConfig

Configure Tesseract OCR settings:

```python
from document_parser import OCRConfig

config = OCRConfig(
    tesseract_cmd="tesseract",       # Path to tesseract binary
    tessdata_prefix="/path/to/data",  # Optional: path to tessdata
    languages="rus+kaz+eng",          # OCR languages (Tesseract format)
    dpi=200,                          # Image DPI for PDF rendering
    psm_mode=3                        # Page segmentation mode
)
```

### Logging

The library includes structured logging. To enable:

```python
from document_parser.logger import setup_logging

# Configure logging level
setup_logging(log_level="INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## API Reference

### High-Level API

#### `parse_document()`

```python
def parse_document(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
    ocr_config: Optional[OCRConfig] = None,
) -> DocumentExtractionResult:
    """Parse a document and extract text."""
```

**Returns:** `DocumentExtractionResult` with:
- `text` (str): Extracted text in Markdown format
- `mime_type` (str): Detected MIME type
- `file_name` (str): Original filename
- `character_count` (int): Number of characters
- `ocr_used` (bool): Whether OCR was used

### Low-Level Classes

#### `DocumentHandler`

Main orchestration class for document processing.

```python
handler = DocumentHandler(ocr_config=OCRConfig())
result = handler.extract(encoded, mime_type, file_name)
```

#### `DocumentDetector`

Detects and validates document types.

```python
detector = DocumentDetector()
descriptor = detector.detect(file_bytes, mime_type, file_name)
```

#### `PyMuPDFExtractor`

Handles text extraction from various document formats.

```python
extractor = PyMuPDFExtractor(config=OCRConfig())
text = extractor.extract(file_bytes, file_name)
```

## Exceptions

All exceptions inherit from `DocumentParserError`:

- `InvalidBase64Error`: Base64 decoding failed
- `UnsupportedTypeError`: Document type not supported
- `ExtractionError`: Text extraction failed
- `DecodingError`: UTF-8 decoding failed (text files)

```python
from document_parser import parse_document
from document_parser.exceptions import DocumentParserError, UnsupportedTypeError

try:
    result = parse_document(file_path="document.pdf")
except UnsupportedTypeError as e:
    print(f"File type not supported: {e}")
except DocumentParserError as e:
    print(f"Parsing failed: {e}")
```

## Development

### Setup

```bash
# Clone repository
git clone git@github.com:your-org/document-parser.git
cd document-parser

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black document_parser/

# Lint
ruff check document_parser/

# Type check
mypy document_parser/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please open an issue on GitHub.
