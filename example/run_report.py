"""
Example: full pipeline analysis with paperqa_skill.

Usage:
    python example/run_report2.py                          # uses ../2312.pdf
    python example/run_report2.py path/to/paper.pdf        # custom PDF/URL
    python example/run_report2.py paper.pdf -t "Title"     # with title

Requirements:
    pip install paper-qa paper-qa[local]
    Set DEEPSEEK_API_KEY and BAILIAN_API_KEY env vars.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from paperqa_skill import check_config, full_pipeline

# ── Resolve default PDF relative to this script ───────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = SCRIPT_DIR.parent / "2312.pdf"


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
        help="Paper title (optional, defaults to PDF filename stem)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: same dir as PDF)",
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

    # ── Run pipeline ──────────────────────────────────────────────────
    pdf_source = args.pdf
    print(f"Analyzing: {pdf_source}")

    zh_html, en_html = await full_pipeline(
        pdf_source=pdf_source,
        paper_title=args.title,
        output_dir=args.output,
        multimodal=not args.no_multimodal,
    )

    print(f"\nChinese report: {zh_html}")
    print(f"English report: {en_html}")


if __name__ == "__main__":
    asyncio.run(main())
