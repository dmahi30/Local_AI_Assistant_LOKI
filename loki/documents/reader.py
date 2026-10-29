"""Safe local document text and metadata extraction for LOKI.

Supports:
- Plain-text files (.txt, .md, .csv, .json, .log, .yaml, .yml)
- PDF files via local pdfplumber with page limits and scanned-PDF detection
- Image files (.png, .jpg, .jpeg, .webp, .bmp) with OCR and vision checks
- Explicit detection and clear reporting of missing dependencies (.docx, .xlsx)
- Bounded file sizes, page limits, and extracted text limits
- Corrupt, encrypted, empty, and unsupported file resilience
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
from enum import Enum
import io
import os
from pathlib import Path
import shutil
from typing import Any

from ..core.logging import get_logger

logger = get_logger("documents.reader")

# Safety & resource limits
DEFAULT_MAX_FILE_BYTES: int = 20 * 1024 * 1024       # 20 MB file size limit
DEFAULT_MAX_PAGES: int = 30                           # 30 pages limit for PDFs
DEFAULT_MAX_EXTRACTED_CHARS: int = 50_000             # 50k characters extracted text limit

TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".markdown", ".log", ".json", ".yaml", ".yml",
}
CSV_EXTENSIONS: set[str] = {".csv"}
PDF_EXTENSIONS: set[str] = {".pdf"}
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

KNOWN_VISION_MODELS: set[str] = {
    "llava", "llama3.2-vision", "qwen2-vl", "minicpm-v", "bakllava", "moondream",
}


class DocumentType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    CSV = "csv"
    UNSUPPORTED = "unsupported"


@dataclass
class DocumentContent:
    """Structured representation of extracted document contents and metadata."""
    file_path: Path
    file_type: DocumentType
    extension: str
    size_bytes: int
    text: str
    pages: list[tuple[int, str]] = field(default_factory=list)  # (page_number, text)
    page_count: int = 1
    is_scanned: bool = False
    needs_ocr: bool = False
    is_truncated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentReader:
    """Safe, bounded local document reader with robust error and edge-case handling."""

    def __init__(
        self,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_extracted_chars: int = DEFAULT_MAX_EXTRACTED_CHARS,
    ) -> None:
        self.max_file_bytes = max_file_bytes
        self.max_pages = max_pages
        self.max_extracted_chars = max_extracted_chars

    def read(
        self,
        file_path: str | Path,
        installed_models: list[str] | None = None,
    ) -> DocumentContent:
        """Extract text and metadata from a local document, enforcing size and safety bounds."""
        path = Path(file_path).resolve()

        if not path.exists():
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.UNSUPPORTED,
                extension=path.suffix.lower(),
                size_bytes=0,
                text="",
                error=f"Document does not exist: '{path}'.",
            )

        if not path.is_file():
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.UNSUPPORTED,
                extension=path.suffix.lower(),
                size_bytes=0,
                text="",
                error=f"Specified path is a directory, not a file: '{path}'.",
            )

        try:
            stat = path.stat()
            size_bytes = stat.st_size
        except OSError as e:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.UNSUPPORTED,
                extension=path.suffix.lower(),
                size_bytes=0,
                text="",
                error=f"Could not read document attributes: {e}",
            )

        # Enforce maximum file size limit
        if size_bytes > self.max_file_bytes:
            size_mb = size_bytes / (1024 * 1024)
            limit_mb = self.max_file_bytes / (1024 * 1024)
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.UNSUPPORTED,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"Document size ({size_mb:.2f} MB) exceeds the maximum allowed limit of {limit_mb:.0f} MB.",
            )

        # Detect empty files
        if size_bytes == 0:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.TEXT,
                extension=path.suffix.lower(),
                size_bytes=0,
                text="",
                error=f"Document is empty (0 bytes): '{path.name}'.",
            )

        ext = path.suffix.lower()

        # Handle explicit unsupported dependency cases
        if ext in (".docx", ".doc"):
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.UNSUPPORTED,
                extension=ext,
                size_bytes=size_bytes,
                text="",
                error=(
                    "Word documents (.docx/.doc) require 'python-docx' which is not installed. "
                    "Supported formats: .txt, .md, .csv, .pdf, .png, .jpg, .jpeg, .webp."
                ),
            )

        if ext in (".xlsx", ".xls"):
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.UNSUPPORTED,
                extension=ext,
                size_bytes=size_bytes,
                text="",
                error=(
                    "Excel spreadsheets (.xlsx/.xls) require 'openpyxl' which is not installed. "
                    "Supported formats: .txt, .md, .csv, .pdf, .png, .jpg, .jpeg, .webp."
                ),
            )

        if ext in TEXT_EXTENSIONS:
            return self._read_text_file(path, size_bytes)
        elif ext in CSV_EXTENSIONS:
            return self._read_csv_file(path, size_bytes)
        elif ext in PDF_EXTENSIONS:
            return self._read_pdf_file(path, size_bytes)
        elif ext in IMAGE_EXTENSIONS:
            return self._read_image_file(path, size_bytes, installed_models)
        else:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.UNSUPPORTED,
                extension=ext,
                size_bytes=size_bytes,
                text="",
                error=(
                    f"Unsupported file format '{ext}'. "
                    "Supported formats: .txt, .md, .csv, .pdf, .png, .jpg, .jpeg, .webp."
                ),
            )

    def _read_text_file(self, path: Path, size_bytes: int) -> DocumentContent:
        """Read plain text files with robust multi-encoding fallback."""
        raw_bytes: bytes
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.TEXT,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"Failed to read file from disk: {e}",
            )

        decoded_text = ""
        for encoding in ("utf-8", "latin-1", "utf-16"):
            try:
                decoded_text = raw_bytes.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if not decoded_text.strip():
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.TEXT,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"Document contains no readable text: '{path.name}'.",
            )

        is_truncated = False
        if len(decoded_text) > self.max_extracted_chars:
            decoded_text = (
                decoded_text[: self.max_extracted_chars]
                + f"\n\n[Truncated: Document content exceeded {self.max_extracted_chars} characters.]"
            )
            is_truncated = True

        return DocumentContent(
            file_path=path,
            file_type=DocumentType.TEXT,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
            text=decoded_text,
            pages=[(1, decoded_text)],
            page_count=1,
            is_truncated=is_truncated,
        )

    def _read_csv_file(self, path: Path, size_bytes: int) -> DocumentContent:
        """Read tabular CSV data into a structured plain-text representation."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.CSV,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"Failed to read CSV file: {e}",
            )

        reader = csv.reader(io.StringIO(content))
        rows: list[list[str]] = []
        try:
            for r in reader:
                if r:
                    rows.append(r)
        except Exception as e:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.CSV,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"Malformed CSV file '{path.name}': {e}",
            )

        if not rows:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.CSV,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"CSV file is empty: '{path.name}'.",
            )

        header = rows[0]
        data_rows = rows[1:]
        max_rows = min(len(data_rows), 200)

        lines = [
            f"CSV Document: {path.name}",
            f"Columns ({len(header)}): {', '.join(header)}",
            f"Total Data Rows: {len(data_rows)}",
            "\n--- Preview of Rows ---",
        ]

        for i, row in enumerate(data_rows[:max_rows], 1):
            row_repr = " | ".join(f"{h}: {val}" for h, val in zip(header, row))
            lines.append(f"Row {i}: {row_repr}")

        is_truncated = len(data_rows) > max_rows
        if is_truncated:
            lines.append(f"\n[Preview truncated to first {max_rows} rows of {len(data_rows)} total rows.]")

        formatted_text = "\n".join(lines)
        if len(formatted_text) > self.max_extracted_chars:
            formatted_text = (
                formatted_text[: self.max_extracted_chars]
                + f"\n\n[Truncated: Extracted CSV content exceeded {self.max_extracted_chars} characters.]"
            )
            is_truncated = True

        return DocumentContent(
            file_path=path,
            file_type=DocumentType.CSV,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
            text=formatted_text,
            pages=[(1, formatted_text)],
            page_count=1,
            is_truncated=is_truncated,
            metadata={"columns": header, "row_count": len(data_rows)},
        )

    def _read_pdf_file(self, path: Path, size_bytes: int) -> DocumentContent:
        """Read PDF text using local pdfplumber, with scanned-PDF and encryption detection."""
        try:
            import pdfplumber
        except ImportError:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.PDF,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error="PDF parsing library 'pdfplumber' is not installed.",
            )

        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    return DocumentContent(
                        file_path=path,
                        file_type=DocumentType.PDF,
                        extension=path.suffix.lower(),
                        size_bytes=size_bytes,
                        text="",
                        page_count=0,
                        error=f"PDF document contains no pages: '{path.name}'.",
                    )

                pages_to_process = min(total_pages, self.max_pages)
                pages_data: list[tuple[int, str]] = []
                total_alphanumeric_chars = 0
                has_images = False

                for idx in range(pages_to_process):
                    page = pdf.pages[idx]
                    page_num = idx + 1
                    try:
                        extracted = page.extract_text() or ""
                    except Exception as pe:
                        logger.warning("Error extracting text on page %d: %s", page_num, pe)
                        extracted = ""

                    clean_text = extracted.strip()
                    pages_data.append((page_num, clean_text))
                    total_alphanumeric_chars += sum(1 for c in clean_text if c.isalnum())

                    if hasattr(page, "images") and len(page.images) > 0:
                        has_images = True

                # Scanned PDF Detection:
                # If total extracted alphanumeric characters across all read pages is less than 20
                if total_alphanumeric_chars < 20:
                    return DocumentContent(
                        file_path=path,
                        file_type=DocumentType.PDF,
                        extension=path.suffix.lower(),
                        size_bytes=size_bytes,
                        text=(
                            "This PDF appears to be a scanned document containing no selectable text. "
                            "Optical Character Recognition (OCR) is required to extract text from scanned pages."
                        ),
                        pages=pages_data,
                        page_count=total_pages,
                        is_scanned=True,
                        needs_ocr=True,
                        metadata={"has_images": has_images, "total_pages": total_pages},
                    )

                # Format page texts with clear headers
                formatted_pages: list[str] = []
                for p_num, p_text in pages_data:
                    if p_text:
                        formatted_pages.append(f"--- [Page {p_num}] ---\n{p_text}")
                    else:
                        formatted_pages.append(f"--- [Page {p_num}] ---\n(No selectable text on this page)")

                full_text = "\n\n".join(formatted_pages)
                is_truncated = False

                if total_pages > self.max_pages:
                    full_text += (
                        f"\n\n[Truncated: Document contains {total_pages} pages. "
                        f"Only the first {self.max_pages} pages were processed.]"
                    )
                    is_truncated = True

                if len(full_text) > self.max_extracted_chars:
                    full_text = (
                        full_text[: self.max_extracted_chars]
                        + f"\n\n[Truncated: Extracted text exceeded {self.max_extracted_chars} characters.]"
                    )
                    is_truncated = True

                return DocumentContent(
                    file_path=path,
                    file_type=DocumentType.PDF,
                    extension=path.suffix.lower(),
                    size_bytes=size_bytes,
                    text=full_text,
                    pages=pages_data,
                    page_count=total_pages,
                    is_truncated=is_truncated,
                    metadata={"processed_pages": pages_to_process, "total_pages": total_pages},
                )
        except Exception as e:
            err_str = str(e).lower()
            if "password" in err_str or "encrypted" in err_str:
                return DocumentContent(
                    file_path=path,
                    file_type=DocumentType.PDF,
                    extension=path.suffix.lower(),
                    size_bytes=size_bytes,
                    text="",
                    error=f"The PDF is password-protected or encrypted: '{path.name}'. Please provide an unencrypted document.",
                )
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.PDF,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"Corrupt or unreadable PDF file '{path.name}': {e}",
            )

    def _read_image_file(
        self,
        path: Path,
        size_bytes: int,
        installed_models: list[str] | None = None,
    ) -> DocumentContent:
        """Inspect image and extract text via local OCR or prepare for local vision model."""
        try:
            from PIL import Image
        except ImportError:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.IMAGE,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error="Image processing library 'Pillow' is not installed.",
            )

        try:
            with Image.open(path) as img:
                width, height = img.size
                img_format = img.format or path.suffix.replace(".", "").upper()
                mode = img.mode
        except Exception as e:
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.IMAGE,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text="",
                error=f"Corrupt or unreadable image file '{path.name}': {e}",
            )

        # 1. Check if local OCR engine (Tesseract) is available
        ocr_text: str | None = None
        has_tesseract = shutil.which("tesseract") is not None
        try:
            import pytesseract  # type: ignore

            if has_tesseract or getattr(pytesseract.pytesseract, "tesseract_cmd", None):
                with Image.open(path) as img:
                    ocr_text = pytesseract.image_to_string(img).strip()
        except Exception as oe:
            logger.debug("pytesseract execution failed: %s", oe)
            ocr_text = None

        if ocr_text is not None and len(ocr_text) > 0:
            formatted = f"[Image OCR Text ({path.name}, {width}x{height})]:\n{ocr_text}"
            return DocumentContent(
                file_path=path,
                file_type=DocumentType.IMAGE,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                text=formatted,
                pages=[(1, formatted)],
                page_count=1,
                metadata={"width": width, "height": height, "format": img_format, "ocr_engine": "tesseract"},
            )

        # 2. Check if local vision model is available in Ollama
        has_vision_model = False
        vision_model_name: str | None = None
        if installed_models:
            for m in installed_models:
                m_clean = m.lower().split(":")[0]
                if any(v in m_clean for v in KNOWN_VISION_MODELS):
                    has_vision_model = True
                    vision_model_name = m
                    break

        if has_vision_model and vision_model_name:
            # Prepare base64 image data for local Ollama vision call
            try:
                b64_data = base64.b64encode(path.read_bytes()).decode("utf-8")
                return DocumentContent(
                    file_path=path,
                    file_type=DocumentType.IMAGE,
                    extension=path.suffix.lower(),
                    size_bytes=size_bytes,
                    text=f"[Image File: {path.name} ({width}x{height}, {img_format}) ready for vision processing]",
                    pages=[(1, f"Image: {path.name}")],
                    page_count=1,
                    metadata={
                        "width": width,
                        "height": height,
                        "format": img_format,
                        "has_vision": True,
                        "vision_model": vision_model_name,
                        "image_b64": b64_data,
                    },
                )
            except Exception as e:
                logger.warning("Could not encode image for vision: %s", e)

        # 3. Neither OCR nor local vision model available
        return DocumentContent(
            file_path=path,
            file_type=DocumentType.IMAGE,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
            text="",
            needs_ocr=True,
            metadata={"width": width, "height": height, "format": img_format},
            error=(
                f"No compatible local OCR engine (such as Tesseract) or local vision model (e.g. llava) is installed. "
                f"Image '{path.name}' ({width}x{height} {img_format}) could not be read. "
                "Please install Tesseract or configure a local vision model in Ollama to process images."
            ),
        )
