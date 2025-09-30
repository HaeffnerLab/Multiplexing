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

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional


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


extract_pages: Callable[[str | Path], list[str]] = _get_pdf_reader()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def convert_single_pdf(pdf_path: Path, output_dir: Path, existing_names: set[str]) -> Optional[Path]:
    """Convert *pdf_path* to Markdown and save it under *output_dir*.

    ``existing_names`` is mutated to include the output file stem to help
    avoid creating duplicate file names.
    """

    try:
        pages = extract_pages(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Skipping {pdf_path} (failed to extract text: {exc})", file=sys.stderr)
        return None

    # Derive a unique stem
    stem = pdf_path.stem
    unique_stem = stem
    counter = 1
    while unique_stem in existing_names:
        unique_stem = f"{stem}_{counter}"
        counter += 1

    existing_names.add(unique_stem)

    md_path = output_dir / f"{unique_stem}.md"
    md_content = "\n\n".join(pages)

    try:
        md_path.write_text(md_content, encoding="utf-8")
        print(f"[ok] {pdf_path} -> {md_path}")
        return md_path
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to write {md_path}: {exc}", file=sys.stderr)
        return None


def traverse_and_convert(input_root: Path, output_dir: Path) -> None:
    """Walk *input_root* recursively, convert every PDF into *output_dir*."""

    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_names: set[str] = {p.stem for p in output_dir.glob("*.md")}

    pdf_count = 0
    for root, _dirs, files in os.walk(input_root):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdf_path = Path(root) / name
                convert_single_pdf(pdf_path, output_dir, existing_names)
                pdf_count += 1

    print(f"\nConverted {pdf_count} PDF file(s).")


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

