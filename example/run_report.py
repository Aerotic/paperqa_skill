"""
Example: full pipeline analysis with paperqa_skill.

Usage:
    python example/run_report.py                              # uses ../2312.pdf
    python example/run_report.py paper.pdf                    # custom PDF/URL
    python example/run_report.py paper.pdf -t "Paper Title"   # with title

Output goes to example/demo_output/{slug}/, where {slug} is derived from
the paper title (simplified) or PDF filename stem as fallback.
Report files inside are named {slug}_report_cn.html, {slug}_report_en.html, etc.

Requirements:
    pip install paper-qa paper-qa[local]
    Set DEEPSEEK_API_KEY and BAILIAN_API_KEY env vars.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from paperqa_skill import check_config, full_pipeline

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = SCRIPT_DIR.parent / "2312.pdf"
DEMO_OUTPUT = SCRIPT_DIR / "demo_output"


def _make_slug(title: str, pdf_source: str) -> str:
    """Derive a clean slug from the paper title, falling back to PDF stem."""
    if title:
        # Take first few words, strip punctuation, lowercase, join
        words = re.findall(r"[a-zA-Z0-9]+", title)
        slug = "_".join(w.lower() for w in words[:5])
        if slug:
            return slug[:60]
    # Fallback: PDF filename stem
    if pdf_source.startswith(("http://", "https://")):
        stem = Path(urlparse(pdf_source).path).stem
    else:
        stem = Path(pdf_source).stem
    return re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:60] or "paper"


def _rename_reports(output_dir: Path, new_stem: str) -> dict[str, str]:
    """Rename all *_report_* files to {new_stem}_report_*.  Also _zh → _cn.
    Returns mapping of old path → new path.
    Uses replace() to overwrite existing files on re-runs."""
    renamed = {}
    for old_path in sorted(output_dir.glob("*_report_*")):
        suffix = old_path.suffix  # .html or .txt
        # Determine lang: zh → cn, en stays en
        if "_report_zh" in old_path.stem:
            lang = "cn"
        elif "_report_en" in old_path.stem:
            lang = "en"
        else:
            continue
        new_path = output_dir / f"{new_stem}_report_{lang}{suffix}"
        old_path.replace(new_path)
        renamed[str(old_path)] = str(new_path)
    return renamed


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paperqa_skill full pipeline on a PDF.",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=str(DEFAULT_PDF),
        help=f"PDF path or URL (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "-t", "--title",
        default="",
        help="Paper title (used for report header and output folder naming)",
    )
    parser.add_argument(
        "--no-multimodal",
        action="store_true",
        help="Disable multimodal figure extraction (no Bailian key needed)",
    )
    args = parser.parse_args()

    # ── Pre-flight config check ───────────────────────────────────────
    missing = check_config()
    if "deepseek_api_key" in str(missing):
        print("ERROR: DEEPSEEK_API_KEY not configured.")
        print("  Set it via env var or run: python -m paperqa_skill --configure")
        sys.exit(1)

    if not args.no_multimodal and "bailian_api_key" in str(missing):
        print("WARNING: BAILIAN_API_KEY not configured — disabling multimodal.")
        args.no_multimodal = True

    # ── Derive slug and output directory ──────────────────────────────
    slug = _make_slug(args.title, args.pdf)
    output_dir = DEMO_OUTPUT / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run pipeline ──────────────────────────────────────────────────
    print(f"Analyzing: {args.pdf}")
    print(f"Title:     {args.title or '(derived from PDF)'}")
    print(f"Output:    {output_dir}")

    zh_html, en_html = await full_pipeline(
        pdf_source=args.pdf,
        paper_title=args.title,
        output_dir=str(output_dir),
        multimodal=not args.no_multimodal,
    )

    # ── Rename reports to use slug ────────────────────────────────────
    renamed = _rename_reports(output_dir, slug)
    for old, new in renamed.items():
        print(f"  {Path(old).name}  →  {Path(new).name}")

    # Update returned paths
    zh_html = renamed.get(zh_html, zh_html)
    en_html = renamed.get(en_html, en_html)

    print(f"\nChinese report: {zh_html}")
    print(f"English report: {en_html}")


if __name__ == "__main__":
    asyncio.run(main())
