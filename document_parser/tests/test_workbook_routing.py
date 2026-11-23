import io
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from openpyxl import Workbook

from document_parser import parse_document
from document_parser.detector import XLSM_MIME_TYPE
from document_parser.exceptions import UnsupportedTypeError
from document_parser.extractor import PyMuPDFExtractor


def test_xlsm_is_read_as_excel(tmp_path):
    workbook = Workbook()
    workbook.active.title = "Finance"
    workbook.active.append(["Revenue", 1234])
    xlsx = io.BytesIO()
    workbook.save(xlsx)

    file_path = tmp_path / "model.XLSM"
    with ZipFile(xlsx) as original, ZipFile(file_path, "w", ZIP_DEFLATED) as xlsm:
        for name in original.namelist():
            data = original.read(name)
            if name == "[Content_Types].xml":
                data = data.replace(
                    b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
                    b"application/vnd.ms-excel.sheet.macroEnabled.main+xml",
                )
            xlsm.writestr(name, data)

    result = parse_document(file_path=str(file_path))

    assert result.mime_type == XLSM_MIME_TYPE
    assert "## Sheet: Finance" in result.text
    assert "Revenue | 1234" in result.text
    assert not result.ocr_used


@pytest.mark.parametrize(
    "contents", [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", b"PK\x03\x04invalid"]
)
def test_unsupported_xlsm_fails_detection(contents):
    with pytest.raises(UnsupportedTypeError, match="expected an Excel OOXML workbook"):
        parse_document(file_bytes=contents, file_name="model.xlsm")


def test_unknown_zip_does_not_fall_back_to_pdf(monkeypatch):
    def unexpected_pdf_call(*args):
        pytest.fail("ZIP was passed to the PDF extractor")

    monkeypatch.setattr(PyMuPDFExtractor, "_extract_pdf", unexpected_pdf_call)

    with pytest.raises(UnsupportedTypeError, match="Unsupported file extension"):
        parse_document(file_bytes=b"PK\x03\x04fake", file_name="archive.unknown")


def test_pdf_without_extension_still_uses_pdf_extractor(monkeypatch):
    def extract_pdf(*args):
        return "Extracted PDF text"

    monkeypatch.setattr(PyMuPDFExtractor, "_extract_pdf", extract_pdf)

    result = parse_document(file_bytes=b"%PDF-1.7", file_name="document")

    assert result.mime_type == "application/pdf"
    assert result.text == "Extracted PDF text"
