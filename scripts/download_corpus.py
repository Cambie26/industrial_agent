"""
Download the FAA Powerplant Handbook chapters into `data/raw/`, one PDF each.

Run from the repo root in Colab: `!cd repo && python scripts/download_corpus.py`.

`CHAPTER_URLS` is empty until the chapter links (and the FAA licence) have
been checked by hand. Until then, upload the chapter PDFs to `data/raw/`
directly, named with their chapter number (`ch10.pdf`).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")

# Chapter number -> PDF URL on faa.gov. Fill in once checked.
CHAPTER_URLS: dict[int, str] = {}


def download(raw_dir: Path, overwrite: bool = False) -> list[Path]:
    """Fetch every chapter in CHAPTER_URLS to `raw_dir`. Skips files already there."""
    if not CHAPTER_URLS:
        raise RuntimeError(
            "CHAPTER_URLS is empty. Add the chapter links to "
            "scripts/download_corpus.py, or upload the PDFs to "
            f"{raw_dir}/ by hand as ch<NN>.pdf."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Loop through each chapter and save it as ch<NN>.pdf.
    saved = []
    for chapter, url in sorted(CHAPTER_URLS.items()):
        pdf_path = raw_dir / f"ch{chapter:02d}.pdf"
        if pdf_path.exists() and not overwrite:
            print(f"have  {pdf_path}")
        else:
            urllib.request.urlretrieve(url, pdf_path)
            print(f"saved {pdf_path}")
        saved.append(pdf_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    download(arguments.raw_dir, arguments.overwrite)


if __name__ == "__main__":
    main()
