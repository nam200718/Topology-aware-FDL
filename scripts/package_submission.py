"""
Packaging script for AAMAS 2027 Anonymous Supplementary Submission.

This script:
1. Runs an automated audit for double-blind anonymity violations.
2. Excludes raw datasets (data/), git history (.git/), virtualenvs, and compiler artifacts.
3. Compresses the clean codebase, documentation, and audited experimental artifacts.
4. Asserts that the resulting zip archive is strictly under the 25 MB conference upload limit.
"""

import os
import sys
import zipfile
import re
from pathlib import Path

# Directories to exclude from the anonymous supplementary archive
EXCLUDE_DIRS = {
    '.git',
    'data',             # Standard datasets (CIFAR-10, MNIST) are ~230MB and downloaded via code
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.venv',
    'venv',
    '.vscode',
    '.idea',
    'scratch',
}

# File extensions to exclude (temporary, compiled, or OS artifacts)
EXCLUDE_EXTS = {
    '.pyc',
    '.pyo',
    '.pyd',
    '.log',
    '.aux',
    '.fls',
    '.fdb_latexmk',
    '.out',
    '.blg',
    '.DS_Store',
    'Thumbs.db',
}

# Patterns to audit for double-blind compliance
DISALLOWED_PATTERNS = [
    (re.compile(r'\b(Nghiem|Duc Khanh|Khanh Nam|Hung Anh|Leandro|Marcolino)\b', re.IGNORECASE), "Author name"),
    (re.compile(r'\b(VinUni|VinUniversity|Lancaster)\b', re.IGNORECASE), "Affiliation/Institution"),
    (re.compile(r'\bnam200718\b', re.IGNORECASE), "Personal GitHub handle"),
    (re.compile(r'[a-zA-Z0-9_.+-]+@(vinuni\.edu\.vn|lancaster\.ac\.uk)', re.IGNORECASE), "Author email"),
]


def audit_file_content(file_path: Path) -> list:
    """Check text files for any accidental leaks of author or institution identities."""
    leaks = []
    # Skip binary files and known external libraries/classes/self
    if file_path.name in {'package_submission.py', 'aamas.cls', 'ACM-Reference-Format.bst'}:
        return leaks
    if file_path.suffix.lower() in {'.pdf', '.png', '.jpg', '.jpeg', '.eps', '.zip', '.pt', '.pth'}:
        return leaks

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(f, 1):
                # Allow references to other published papers in .bib unless it mentions authors
                if file_path.suffix == '.bib' and 'author' in line.lower():
                    if any(name in line.lower() for name in ['nam.ndk', 'marcolino', 'vinuni', 'lancaster']):
                        leaks.append((file_path, line_no, "Leaked author info in bibliography", line.strip()))
                    continue

                for pattern, desc in DISALLOWED_PATTERNS:
                    m = pattern.search(line)
                    if m:
                        leaks.append((file_path, line_no, f"{desc}: '{m.group(0)}'", line.strip()[:80]))
    except Exception as e:
        pass
    return leaks


def create_submission_zip(output_zip: str = "fedhep_supplementary_material.zip", max_size_mb: float = 25.0):
    root_dir = Path(".").resolve()
    print("=" * 70)
    print("  FEDHEP ANONYMOUS SUBMISSION PACKAGING & COMPLIANCE VERIFIER")
    print("=" * 70)

    # 1. Audit files for anonymity
    print("\n[Step 1/3] Auditing workspace for double-blind anonymity violations...")
    all_leaks = []
    files_to_pack = []

    for path in root_dir.rglob('*'):
        if path.is_dir():
            continue

        # Check if file resides in an excluded directory
        parts = path.relative_to(root_dir).parts
        if any(part in EXCLUDE_DIRS for part in parts[:-1]):
            continue

        # Check if file has an excluded extension or is the zip itself
        if path.suffix in EXCLUDE_EXTS or path.name == output_zip or path.name.endswith('.zip'):
            continue

        files_to_pack.append(path)
        leaks = audit_file_content(path)
        if leaks:
            all_leaks.extend(leaks)

    if all_leaks:
        print("\n[ERROR] Anonymity violations detected:")
        for fp, lno, desc, snippet in all_leaks:
            rel = fp.relative_to(root_dir)
            print(f"  - {rel}:{lno} [{desc}] -> {snippet}")
        print("\nPlease fix these issues before packaging!")
        sys.exit(1)
    else:
        print("  -> Passed! Zero anonymity violations detected.")

    # 2. Package into zip archive
    print(f"\n[Step 2/3] Compressing {len(files_to_pack)} clean files into '{output_zip}'...")
    zip_out = root_dir / output_zip
    if zip_out.exists():
        zip_out.unlink()

    with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files_to_pack:
            rel_path = p.relative_to(root_dir)
            zf.write(p, arcname=str(rel_path).replace('\\', '/'))

    # 3. Check archive size
    size_bytes = zip_out.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"\n[Step 3/3] Checking file size constraint...")
    print(f"  - Archive Name: {output_zip}")
    print(f"  - Packed Files: {len(files_to_pack)}")
    print(f"  - Final Archive Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"  - Size Ceiling: {max_size_mb:.2f} MB")

    if size_mb > max_size_mb:
        print(f"\n[ERROR] Archive size ({size_mb:.2f} MB) exceeds the {max_size_mb:.2f} MB ceiling!")
        sys.exit(1)

    print(f"  -> Compliance SUCCESS: {size_mb:.2f} MB is well within the {max_size_mb:.2f} MB limit! ({size_mb / max_size_mb * 100:.1f}% of quota)")
    print("=" * 70)
    print(f"Archive ready for submission: {zip_out.name}")
    print("=" * 70)


if __name__ == '__main__':
    create_submission_zip()
