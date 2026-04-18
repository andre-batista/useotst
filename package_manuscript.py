"""
Creates a .zip file with all files needed to compile manuscript/main.tex.
"""

import re
import zipfile
from pathlib import Path

# Paths relative to this script (project root)
PROJECT_ROOT = Path(__file__).parent
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
MAIN_TEX = MANUSCRIPT_DIR / "main.tex"
OUTPUT_ZIP = PROJECT_ROOT / "manuscript_submission.zip"

# Files inside the manuscript/ folder that are always required
MANUSCRIPT_FILES = [
    "main.tex",
    "main.bbl",          # pre-built bibliography (for submission without BibTeX)
    "wiley-article.cls",
    "bibliography/refs.bib",
]


def find_graphics(tex_path: Path) -> list[Path]:
    """Return absolute paths of all images referenced via \\includegraphics."""
    tex_dir = tex_path.parent
    content = tex_path.read_text(encoding="utf-8")
    # Match \includegraphics[...]{path} — path may have no extension
    pattern = re.compile(r"\\includegraphics(?:\[.*?\])?\{([^}]+)\}")
    paths = []
    for match in pattern.finditer(content):
        raw = match.group(1).strip()
        # Resolve relative to the .tex file location
        candidate = (tex_dir / raw).resolve()
        # If no extension given, try common ones
        if not candidate.suffix:
            for ext in (".png", ".pdf", ".eps", ".jpg"):
                if candidate.with_suffix(ext).exists():
                    candidate = candidate.with_suffix(ext)
                    break
        paths.append(candidate)
    return paths


def main():
    files_to_add: list[tuple[Path, str]] = []  # (absolute_path, archive_name)

    # --- Manuscript static files ---
    for rel in MANUSCRIPT_FILES:
        abs_path = MANUSCRIPT_DIR / rel
        if abs_path.exists():
            files_to_add.append((abs_path, f"manuscript/{rel}"))
        else:
            print(f"  [WARNING] Not found: {abs_path}")

    # --- Images referenced in main.tex ---
    for img_path in find_graphics(MAIN_TEX):
        if img_path.exists():
            # Preserve path relative to project root inside the archive
            try:
                arc_name = img_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                arc_name = img_path.name
            files_to_add.append((img_path, arc_name))
        else:
            print(f"  [WARNING] Image not found: {img_path}")

    # --- Write zip ---
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for abs_path, arc_name in files_to_add:
            if arc_name in seen:
                continue
            seen.add(arc_name)
            zf.write(abs_path, arc_name)
            print(f"  Added: {arc_name}")

    print(f"\nCreated: {OUTPUT_ZIP}")
    print(f"Total files: {len(seen)}")


if __name__ == "__main__":
    main()
