"""convert_zotero_pdf_to_md.py
--------------------------------
Traverse a Zotero `storage` directory, find every PDF file, extract
its text and write it to a Markdown file with the same base‐name in a
single output directory (default: ``/Users/bingran_you/Zotero/deeptutor_storage``).

Usage (CLI):
    python convert_zotero_pdf_to_md.py \
        --input /Users/bingran_you/Zotero/storage \
        --output /Users/bingran_you/Zotero/deeptutor_storage

Dependencies:
    - Python ≥ 3.8
    - ``pypdf`` (``pip install pypdf``)  *or* ``PyPDF2`` 2.x.  The script
      first tries to import ``pypdf`` and falls back to ``PyPDF2``.

The extractor is intentionally minimal: it walks the directory tree,
skips non‑PDF files, extracts page text, joins pages with two newlines
between them, then writes the result to ``<basename>.md``.  If multiple
PDFs share the same base‑name the script appends a numeric suffix to
avoid overwriting files (e.g. ``paper.md``, ``paper_1.md``).
"""

#
# Extended version: incremental conversion & OCR handling
# ------------------------------------------------------

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Set

# External dependency: langchain‑community (PyMuPDFLoader)
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ModuleNotFoundError as exc:  # pragma: no cover – runtime error path
    raise SystemExit(
        "Missing dependency 'langchain_community'. Install via 'pip install langchain-community==0.0.29' "
        "or compatible version that provides PyMuPDFLoader."
    ) from exc

# Optional dependency: ocrmypdf for image‑only PDFs
# We won't import it here; we'll call via subprocess when needed.


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF text extraction helpers
# ---------------------------------------------------------------------------


def _get_pdf_reader() -> Callable[[str], list[str]]:
    """Return a function `(pdf_path) -> list[str]` that extracts page texts.

    The function tries to use ``pypdf`` first (more actively maintained)
    and falls back to ``PyPDF2`` if ``pypdf`` is not available.  We lazy
    import the libraries so the script still works even if neither is
    installed – an informative error is raised in that case.
    """

    # Try pypdf (preferred)
    try:
        from pypdf import PdfReader  # type: ignore

        def _extract_with_pypdf(path: str | Path) -> list[str]:
            reader = PdfReader(str(path))
            return [page.extract_text() or "" for page in reader.pages]

        return _extract_with_pypdf
    except ModuleNotFoundError:
        pass

    # Fallback to PyPDF2 ≤ 3.x API
    try:
        from PyPDF2 import PdfReader  # type: ignore

        def _extract_with_pypdf2(path: str | Path) -> list[str]:
            reader = PdfReader(str(path))
            return [page.extract_text() or "" for page in reader.pages]

        return _extract_with_pypdf2
    except ModuleNotFoundError:
        pass

    def _missing(_: str | Path) -> list[str]:  # pragma: no cover – runtime error path
        raise RuntimeError(
            "Neither 'pypdf' nor 'PyPDF2' is installed.  Please install one, "
            "e.g. `pip install pypdf`."
        )

    return _missing


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def generate_file_id(path: Path, chunk_size: int = 8192) -> str:
    """Return a hex SHA‑256 hash of *path*'s binary contents."""

    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def _is_pdf_image_only(pdf_path: str | Path) -> bool:  # noqa: D401 – function follows spec above
    """Detect image‑only PDFs using PyMuPDFLoader (no extractable text)."""

    try:
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        total_text = "".join(
            doc.page_content.strip() for doc in documents if getattr(doc, "page_content", None)
        )
        if not total_text or len(total_text.strip()) < 10:
            logger.info(f"PDF appears to be image‑only: {pdf_path}")
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Error analysing PDF {pdf_path}: {exc}. Assuming image‑only.")
        return True


def _ocr_with_ocrmypdf(input_pdf: Path, tmp_dir: Path, language: str = "eng") -> Path:
    """Run OCR on *input_pdf* using `ocrmypdf` and return path to OCR‑ed PDF."""

    output_pdf = tmp_dir / (input_pdf.stem + "_ocr.pdf")
    sidecar_txt = tmp_dir / (input_pdf.stem + "_ocr.txt")

    cmd = [
        sys.executable,
        "-m",
        "ocrmypdf",
        "--force-ocr",
        "--sidecar",
        str(sidecar_txt),
        "-l",
        language,
        str(input_pdf),
        str(output_pdf),
    ]

    logger.info(f"Running OCR on {input_pdf} …")
    try:
        subprocess.run(cmd, check=True)
        return output_pdf
    except subprocess.CalledProcessError as exc:  # pragma: no cover – OCR failure
        logger.error(f"OCR failed for {input_pdf}: {exc}")
        raise


def _extract_text_via_loader(pdf_path: Path) -> list[str]:
    """Extract text from *pdf_path* using PyMuPDFLoader, page by page."""

    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    return [doc.page_content or "" for doc in documents]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def convert_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    existing_names: Set[str],
    mapping: Dict[str, str],
    tmp_dir: Path,
) -> Optional[str]:
    """Convert *pdf_path* to Markdown, update *mapping* and return file ID."""

    file_id = generate_file_id(pdf_path)

    # Skip if already converted and file still exists
    if file_id in mapping:
        return file_id

    # Extract or OCR if needed
    if _is_pdf_image_only(pdf_path):
        try:
            ocr_pdf = _ocr_with_ocrmypdf(pdf_path, tmp_dir)
            pages = _extract_text_via_loader(ocr_pdf)
        except Exception:
            logger.warning(f"Skipping {pdf_path} – OCR failed.")
            return None
    else:
        pages = _extract_text_via_loader(pdf_path)

    # Determine unique stem
    stem = pdf_path.stem
    unique_stem = stem
    counter = 1
    while unique_stem in existing_names:
        unique_stem = f"{stem}_{counter}"
        counter += 1
    existing_names.add(unique_stem)

    md_path = output_dir / f"{unique_stem}.md"
    md_path.write_text("\n\n".join(pages), encoding="utf-8")

    mapping[file_id] = md_path.name
    logger.info(f"[new] {pdf_path} -> {md_path}")
    return file_id


def traverse_and_convert(input_root: Path, output_dir: Path) -> None:
    """Incrementally convert PDFs, syncing deletions and additions."""

    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "conversion_mapping.json"
    if mapping_path.exists():
        mapping: Dict[str, str] = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        mapping = {}

    existing_names: Set[str] = {Path(name).stem for name in mapping.values()}

    # Discover current PDFs and build id set
    current_ids: Set[str] = set()
    tmp_dir = output_dir / ".tmp_ocr"
    tmp_dir.mkdir(exist_ok=True)

    for root, _dirs, files in os.walk(input_root):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdf_path = Path(root) / name
                fid = convert_single_pdf(pdf_path, output_dir, existing_names, mapping, tmp_dir)
                if fid:
                    current_ids.add(fid)

    # Handle deletions: remove md files whose source PDF disappeared
    deleted_ids = set(mapping.keys()) - current_ids
    for fid in deleted_ids:
        md_name = mapping.pop(fid)
        md_path = output_dir / md_name
        if md_path.exists():
            md_path.unlink()
            logger.info(f"[del] Removed orphaned markdown {md_path}")

    # Persist mapping
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    logger.info(f"Sync complete. Total PDFs tracked: {len(mapping)}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:  # noqa: D401 – simple function
    parser = argparse.ArgumentParser(
        description="Convert all PDFs in Zotero storage to Markdown files.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("/Users/bingran_you/Zotero/storage"),
        help="Path to Zotero 'storage' directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/Users/bingran_you/Zotero/deeptutor_storage"),
        help="Destination directory for Markdown files.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401 – simple function
    args = _parse_args()
    traverse_and_convert(args.input.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
