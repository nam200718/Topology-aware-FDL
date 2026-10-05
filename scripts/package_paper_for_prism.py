#!/usr/bin/env python3
"""
Package the paper folder into a pristine .zip ready to import into OpenAI Prism (prism.openai.com) or Overleaf.
Maintains main.tex at the root of the archive with all dependencies and RGB assets.
"""

import os
import zipfile
from pathlib import Path

def create_prism_package():
    repo_root = Path(__file__).resolve().parent.parent
    paper_dir = repo_root / "paper"
    output_zip = paper_dir / "paper_prism_bundle.zip"

    excluded_extensions = {".zip", ".aux", ".bbl", ".blg", ".log", ".out", ".toc", ".synctex.gz", ".fdb_latexmk", ".fls"}
    excluded_names = {".DS_Store", "Thumbs.db"}

    file_count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(paper_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for file in sorted(files):
                if file.startswith(".") or file in excluded_names:
                    continue
                ext = Path(file).suffix.lower()
                if ext in excluded_extensions:
                    continue
                file_path = Path(root) / file
                arcname = file_path.relpath(paper_dir) if hasattr(file_path, "relpath") else os.path.relpath(file_path, paper_dir)
                zf.write(file_path, arcname)
                file_count += 1

    size_kb = output_zip.stat().st_size / 1024.0
    print(f"Prism zip package created: {output_zip} ({size_kb:.1f} KB, {file_count} files)")
    return output_zip

if __name__ == "__main__":
    create_prism_package()
