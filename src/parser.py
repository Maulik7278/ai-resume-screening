"""
PDF resume parsing.

Extracts raw text from a PDF file. Kept deliberately small and isolated so
parser failures (corrupt PDFs, scanned images with no text layer, etc.) are
easy to catch and report as a single, well-understood failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ParseError(Exception):
    """Raised when a PDF cannot be parsed into usable text."""


@dataclass
class ParsedResume:
    source_file: str
    text: str
    page_count: int


def parse_resume(path: Path) -> ParsedResume:
    """
    Extract text from a single PDF resume.

    Raises ParseError on any failure (corrupt file, encrypted file that
    can't be opened, no extractable text, etc.) so callers can isolate the
    failure to this one resume without affecting the rest of the batch.
    """
    if not path.exists():
        raise ParseError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ParseError(f"Unsupported file type (not a PDF): {path.name}")

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise ParseError(f"Corrupt or unreadable PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - external library, wrap broadly on purpose
        raise ParseError(f"Failed to open PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            # Try an empty password; many "encrypted" resumes just have
            # owner-password restrictions with no user password required.
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"Encrypted PDF could not be decrypted: {exc}") from exc

    text_chunks: list[str] = []
    try:
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_chunks.append(page_text)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Failed extracting text from PDF: {exc}") from exc

    full_text = "\n".join(text_chunks).strip()
    if not full_text:
        raise ParseError("No extractable text found (likely a scanned/image-only PDF)")

    return ParsedResume(source_file=path.name, text=full_text, page_count=len(reader.pages))


def discover_resumes(input_dir: Path) -> list[Path]:
    """Return every PDF found directly inside input_dir, sorted for determinism."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
