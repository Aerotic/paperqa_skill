"""
Example: full pipeline analysis with paperqa_skill.

Usage:
    python example/run_report.py                                   # uses ../2312.pdf
    python example/run_report.py paper.pdf                         # custom PDF/URL
    python example/run_report.py paper.pdf -t "Paper Title"        # with title
    python example/run_report.py demo_output/2312 --regenerate     # rebuild from cache

Output goes to example/demo_output/{slug}/, where {slug} is derived from
the paper title (simplified) or PDF filename stem as fallback.
Report files: {slug}_report_zh.html, {slug}_report_en.html,
              {slug}_report_zh.txt,  {slug}_report_en.txt

Requirements:
    pip install paper-qa paper-qa[local]
    Set DEEPSEEK_API_KEY and (optionally) BAILIAN_API_KEY env vars.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from paperqa_skill import check_config, full_pipeline, regenerate_reports

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = SCRIPT_DIR.parent / "2312.pdf"
DEMO_OUTPUT = SCRIPT_DIR / "demo_output"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_slug(title: str, pdf_source: str) -> str:
    """Derive a clean slug from title (preferred) or PDF filename stem."""
    if title:
        words = re.findall(r"[a-zA-Z0-9]+", title)
        slug = "_".join(w.lower() for w in words[:5])
        if slug:
            return slug[:60]
    # Fallback: PDF filename stem
    stem = (Path(urlparse(pdf_source).path).stem
            if pdf_source.startswith(("http://", "https://"))
            else Path(pdf_source).stem)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:60] or "paper"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paperqa_skill full pipeline / regenerate from cache.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_PDF),
        help=f"PDF path/URL, or cache directory with --regenerate (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "-t", "--title",
        default="",
        help="Paper title (used for report header and output folder naming)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: demo_output/{slug}/)",
    )
    parser.add_argument(
        "--no-multimodal",
        action="store_true",
        help="Disable multimodal figure extraction (no Bailian key needed)",
    )
    parser.add_argument(
        "-r", "--regenerate",
        action="store_true",
        help="Regenerate reports from cached PaperQA2 results (source is cache dir)",
    )
    args = parser.parse_args()

    # ── Config check ────────────────────────────────────────────────────
    missing = check_config()
    if missing:
        if any("deepseek_api_key" in m for m in missing):
            print("ERROR: DEEPSEEK_API_KEY not configured.")
            print("  Set it via env var or run: python -m paperqa_skill --configure")
            sys.exit(1)

    if not args.no_multimodal and any("bailian_api_key" in m for m in missing):
        print("WARNING: BAILIAN_API_KEY not configured — disabling multimodal.")
        args.no_multimodal = True

    # ── Derive slug and output directory ────────────────────────────────
    slug = _make_slug(args.title, args.source)
    output_dir = Path(args.output_dir) if args.output_dir else (DEMO_OUTPUT / slug)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run pipeline ────────────────────────────────────────────────────
    if args.regenerate:
        # Rebuild from cached intermediate results
        cache_dir = Path(args.source)
        if not cache_dir.is_dir():
            print(f"ERROR: Cache directory not found: {cache_dir}")
            sys.exit(1)

        # Try to find the PDF for figure extraction
        pdf_path = None
        if not args.no_multimodal:
            candidates = list(cache_dir.glob("*.pdf")) + list(cache_dir.parent.glob("*.pdf"))
            if candidates:
                pdf_path = str(candidates[0])
                print(f"Found PDF for figures: {pdf_path}")
            else:
                print("No PDF found — figures will be skipped")

        print(f"Regenerating reports from: {cache_dir}")
        print(f"Title:  {args.title or '(derived from dir name)'}")
        print(f"Output: {output_dir}")

        zh_html, en_html = await regenerate_reports(
            cache_dir=str(cache_dir),
            paper_title=args.title,
            pdf_path=pdf_path,
            output_dir=str(output_dir),
        )
    else:
        # Full pipeline: analyze PDF from scratch
        print(f"Analyzing: {args.source}")
        print(f"Title:     {args.title or '(derived from PDF)'}")
        print(f"Output:    {output_dir}")

        zh_html, en_html = await full_pipeline(
            pdf_source=args.source,
            paper_title=args.title,
            output_dir=str(output_dir),
            multimodal=not args.no_multimodal,
        )

    print(f"\nChinese report: {zh_html}")
    print(f"English report: {en_html}")


if __name__ == "__main__":
    asyncio.run(main())
