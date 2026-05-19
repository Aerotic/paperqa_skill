"""
Full end-to-end test for paperqa_skill report generation pipeline.

Modes:
    --quick         Uses local mock data (no API keys, no network)
    --render-only   Skips LLM call, uses cached report from .cache/
    (default)       Full pipeline: reads cached QA results → LLM → HTML → validate

Usage:
    python example/test_full_pipeline.py                    # full pipeline
    python example/test_full_pipeline.py --quick            # mock data, no API
    python example/test_full_pipeline.py --render-only      # re-render cached report
    python example/test_full_pipeline.py -o demo_output/test_out  # custom output
"""

import argparse
import asyncio
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"
DEFAULT_QA_DIR = SCRIPT_DIR / "demo_output" / "2312"


# ═══════════════════════════════════════════════════════════════════════════════
# Mock data — simulates PaperQA2 query results without real API calls
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_ANSWERS = [
    # Query 1: Comprehensive overview (English, as PaperQA2 outputs)
    """
The paper introduces MobileGPT, a system that augments LLMs with human-like app memory for mobile task automation. The key problem is enabling LLMs to efficiently learn and recall sub-tasks across different app pages, as existing datasets do not capture repeated tasks.

The proposed methodology uses a hierarchical memory architecture where app pages are grouped into nodes based on shared functionalities rather than visual appearance. MobileGPT employs a multi-phase approach: cold-start uses Explore, Select, and Derive phases to learn new tasks; warm-start uses sub-task slot-filling and in-context action adaptation to recall learned tasks. Screen classification checks for key UI elements rather than extracting full subtask lists each time. The system uses GPT-4-turbo for the Explore/Select/Derive phases and GPT-3.5-turbo for slot-filling.

Evaluation used a custom dataset of 80 tasks across 8 apps. MobileGPT achieved 82.5% task success rate in warm-start, outperforming baselines AutoDroid and AppAgent. Warm-start reduced latency by 87%. An ablation study showed near-perfect accuracy (98.75%) during warm-start due to HITL task repair and memory. Offline exploration of 50 unique app pages per app covered 89.65% of needed pages at $10.78 cost. Step accuracy reached 96.4% for Explore phase and 99.1% for Derive phase. Slot-filling accuracy was 99.5% and in-context adaptation accuracy was 100%. Screen classification achieved only 3 false positives and 0 false negatives across 269 screens.
""",
    # Query 2: Innovation deep dive (English)
    """
1. Explore-Select-Derive-Recall Workflow with Hierarchical Memory
- Core idea: MobileGPT structures task automation into four phases with three-level hierarchical memory (tasks, sub-tasks, actions).
- Difference from prior work: Prior systems lacked structured sub-task decomposition and memory reuse.
- Implementation: Uses GPT-4-turbo for Explore/Select/Derive and GPT-3.5-turbo for slot-filling.
- Evidence: Phase accuracies: Explore 96.4%, Select 96.2%, Derive 99.1%; step accuracy 99.8%. Warm-start latency reduced 87% vs AutoDroid.

2. Sub-task-based Screen Classification
- Core idea: Classifies screens by verifying key UI elements using cosine similarity.
- Difference from prior work: Traditional methods had 38/57 false positives/negatives; MobileGPT achieves 3 false positives and 0 false negatives on 269 app screens.
- Evidence: False positives reduced 92.1%, false negatives reduced 100%.

3. Dual-Strategy Correction Mechanism
- Core idea: Combines LLM self-correction with Human-in-the-loop (HITL) repair.
- Implementation: Rule-based feedback for UI errors and loop detection; HITL for Explore, Select, Derive phases.
- Evidence: Warm-start accuracy 98.75% (with HITL) vs 72.3% (without HITL), a 36.6% improvement.

4. Random Exploration for Offline App Screen Collection
- Core idea: Uses random explorer and user trace monitor to pre-collect 50 app pages per app.
- Evidence: 89.65% coverage of required pages, one-time cost $10.78.

5. Attribute-based and In-context Action Adaptation
- Core idea: Adjusts recalled actions dynamically for changed parameters using LLM few-shot learning.
- Evidence: Slot filling accuracy 99.5%, in-context adaptation accuracy 100%, step accuracy 99.8%.
""",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HTML validation — parses output and checks for truncation / structure issues
# ═══════════════════════════════════════════════════════════════════════════════

class SectionInfo:
    """Info about one HTML section card."""
    def __init__(self):
        self.title = ""
        self.body_parts: list[str] = []


class ReportValidator(HTMLParser):
    """Parse paperqa_skill HTML report and extract section structure."""

    def __init__(self):
        super().__init__()
        self.sections: list[SectionInfo] = []
        self._current: SectionInfo | None = None
        self._in_h3 = False
        self._in_body = False
        self._buffer = ""

    def handle_starttag(self, tag, attrs):
        if tag == "h3":
            self._in_h3 = True
            self._buffer = ""
        elif tag == "div":
            for k, v in attrs:
                if k == "class" and v == "section-body":
                    self._in_body = True
                    if self._current is None:
                        self._current = SectionInfo()

    def handle_endtag(self, tag):
        if tag == "h3":
            if self._current:
                self._current.title = self._buffer.strip()
            self._buffer = ""
            self._in_h3 = False
        elif tag == "div":
            if self._in_body:
                self._in_body = False
                if self._current:
                    self.sections.append(self._current)
                    self._current = None

    def handle_data(self, data):
        if self._in_h3:
            self._buffer += data
        elif self._in_body and self._current:
            d = data.strip()
            if d:
                self._current.body_parts.append(d)


def _validate_html(html_path: str, expected_sections: list[str]) -> dict:
    """Parse generated HTML and return validation results.

    Args:
        html_path: Path to the HTML file
        expected_sections: List of expected section title keywords (e.g. ["简述", "问题与挑战", ...])

    Returns:
        dict with keys: "passed", "failures", "warnings", "sections_found"
    """
    failures: list[str] = []
    warnings: list[str] = []
    results = {"passed": True, "failures": failures, "warnings": warnings,
               "sections_found": []}

    if not os.path.isfile(html_path):
        failures.append(f"HTML file not found: {html_path}")
        results["passed"] = False
        return results

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # ── Parse sections ──────────────────────────────────────────────────
    validator = ReportValidator()
    validator.feed(html)
    validator.close()

    results["sections_found"] = [s.title for s in validator.sections]

    # ── Check 1: Expected sections present ──────────────────────────────
    found_keywords = set()
    for sec in validator.sections:
        for kw in expected_sections:
            if kw in sec.title:
                found_keywords.add(kw)

    missing = [kw for kw in expected_sections if kw not in found_keywords]
    if missing:
        failures.append(f"Missing expected sections: {missing}")

    # ── Check 2: No truncated titles near body content ──────────────────
    for sec in validator.sections:
        if "…" in sec.title and len(sec.title) < 20:
            failures.append(f"Section title appears truncated with short length: '{sec.title[:60]}'")
        # Titles ending with … should have body content
        if "…" in sec.title and not sec.body_parts:
            failures.append(f"Truncated title with EMPTY body: '{sec.title[:60]}'")

    # ── Check 3: Sections with legitimate titles should have body ───────
    for sec in validator.sections:
        title_clean = re.sub(r'<[^>]+>', '', sec.title).strip()
        # Architecture flow sections may have SVG instead of body text
        if "架构流程" in title_clean or "Architecture" in title_clean:
            continue
        if len(title_clean) > 5 and not sec.body_parts:
            failures.append(f"Section with title has EMPTY body: '{title_clean[:50]}'")

    # ── Check 4: First section (overview) should have substantial body ──
    if validator.sections:
        first = validator.sections[0]
        body_text = " ".join(first.body_parts)
        if len(body_text) < 30:
            failures.append(
                f"First section body too short ({len(body_text)} chars): '{body_text[:80]}'")

    # ── Check 5: Metrics table present ──────────────────────────────────
    if "<table>" not in html:
        warnings.append("No <table> found in HTML (metrics table missing?)")

    # ── Check 6: SVG flowchart present (if architecture section exists) ─
    if any("架构流程" in s.title or "Architecture Flow" in s.title
           for s in validator.sections):
        if "<svg" not in html:
            warnings.append("Architecture section exists but no <svg> found")

    # ── Check 7: No raw markdown artifacts ──────────────────────────────
    if re.search(r'^#{1,4}\s', html, re.MULTILINE):
        warnings.append("Raw markdown heading (#) found in HTML output")

    results["passed"] = len(failures) == 0
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════════

async def run_pipeline(
    combined: str,
    paper_title: str,
    output_dir: Path,
    lang: str = "zh",
    skip_llm: bool = False,
) -> str:
    """Run the full report generation pipeline (LLM prompt → HTML → TXT)."""
    from litellm import acompletion
    from paperqa_skill import (
        DEEPSEEK_API_KEY, DEEPSEEK_API_BASE,
        _generate_html_report, _format_metrics_for_txt,
    )

    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", paper_title or output_dir.name)[:60] or "paper"
    cache_file = CACHE_DIR / f"{slug}_{lang}_report.txt"

    # ── Step 1: LLM report generation (or cache) ───────────────────────
    if skip_llm and cache_file.exists():
        print(f"  [CACHE] Using: {cache_file}")
        report_text = cache_file.read_text(encoding="utf-8")
    else:
        print(f"  [LLM] Generating {lang} report...")
        report_text = await _llm_generate(combined, paper_title, lang)
        print(f"  [LLM] Done ({len(report_text)} chars)")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(report_text, encoding="utf-8")

    # ── Step 2: Parse parts ────────────────────────────────────────────
    parts = [p.strip() for p in re.split(r'\n(?=# [^#])', report_text.strip()) if p.strip()]
    print(f"  Parts: {len(parts)} ({[p.split(chr(10))[0][:40] for p in parts]})")

    # ── Step 3: Generate flow SVG ──────────────────────────────────────
    from paperqa_skill.__init__ import _build_flow_svg
    flow_svg = await _build_flow_svg(paper_title, combined, lang=lang)
    print(f"  Flow SVG: {'ok' if flow_svg else 'empty'}")

    # ── Step 4: Generate HTML ──────────────────────────────────────────
    html_path = str(output_dir / f"{slug}_report_{lang}.html")
    _generate_html_report(
        html_path=html_path,
        title=paper_title or slug,
        summary=report_text,
        all_answers=[],
        flow_svg=flow_svg,
        figures=[],
        parts=parts,
        lang=lang,
    )
    print(f"  HTML: {html_path}")

    # ── Step 5: Generate TXT ───────────────────────────────────────────
    txt_path = str(output_dir / f"{slug}_report_{lang}.txt")
    text = "\n\n".join(parts)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'###\s+', '', text)
    text = re.sub(r'# [^\n]*\n', '', text)
    text = _format_metrics_for_txt(text)
    Path(txt_path).write_text(text.strip(), encoding="utf-8")
    print(f"  TXT: {txt_path}")

    return html_path


async def _llm_generate(combined: str, title: str, lang: str) -> str:
    """Call DeepSeek to generate the structured report."""
    from litellm import acompletion
    from paperqa_skill import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE

    if lang == "zh":
        prompt = (
            f"你是一位顶级学术论文分析专家。请根据以下对论文「{title}」的分析数据，"
            f"生成一份完整的**中文技术分析报告**。\n\n"
            f"## 输出结构 (严格按顺序, 2个大块)\n\n"
            f"# 第一部分：总览 (含 5 个子部分，每个用 ### 标记)\n\n"
            f"### 简述\n"
            f"用一段话简洁概括论文全貌，覆盖核心问题、方法、关键效果，关键数字用**加粗**。\n\n"
            f"### 问题与挑战\n\n"
            f"### 方法概览\n\n"
            f"### 架构流程\n"
            f"用文字描述系统架构的完整链路，按步骤编号列出各阶段组件。\n\n"
            f"### 关键效果\n\n"
            f"**指标表格**（放在第一部分末尾，不属于单独小节）：\n"
            f"表格示例：\n"
            f"```html\n"
            f"<table>\n"
            f"    <tr>\n"
            f"        <td>指标名称</td>\n"
            f"        <td>值</td>\n"
            f"        <td>含义解释</td>\n"
            f"    </tr>\n"
            f"    <tr>\n"
            f"        <td>准确率提升</td>\n"
            f"        <td>39%</td>\n"
            f"        <td>Top-1准确率相较于baseline的提升幅度，准确率是与ground truth一致的预测数量和总数量的比值</td>\n"
            f"    </tr>\n"
            f"</table>\n"
            f"```\n"
            f"重要：含义解释列不可省略，必须用完整句子描述指标意义，不可仅重复指标名。\n\n"
            f"# 第二部分：创新点深度剖析\n"
            f"每个主要创新点一个 ### 小节：创新点命名 → 核心思想 → 与已有工作的区别 → 技术实现 → 有效性证据。\n\n"
            f"## 格式要求\n"
            f"- 用 # 标记 2 大块, ### 标记小节，### 标题独占一行\n"
            f"- 关键数字用 **加粗**，技术术语中英文对照\n"
            f"- 创新点每小节 5-8 句有实质内容\n"
            f"- 仅写数据能确认的信息，不空洞评价\n"
            f"- 指标表格放在第一部分末尾\n\n"
            f"## 分析数据\n"
            f"{combined[:12000]}"
        )
    else:
        prompt = (
            f"You are a professional academic analyst. Based on the analysis data "
            f"for paper '{title}', write a complete **technical analysis report**.\n\n"
            f"## Output Structure (strict order, 2 major parts)\n\n"
            f"# Part 1: Overview (5 subsections, each marked with ###)\n\n"
            f"### Summary\n"
            f"Briefly summarize the paper covering problem, method, key results. Bold numbers with **.\n\n"
            f"### Problem & Challenges\n\n"
            f"### Method Overview\n\n"
            f"### Architecture Flow\n"
            f"Describe the complete system pipeline, numbering each stage.\n\n"
            f"### Key Results\n\n"
            f"**Metrics Table** (at end of Part 1):\n"
            f"```html\n"
            f"<table><tr><td>Metric Name</td><td>Value</td><td>Explanation</td></tr>"
            f"<tr><td>Accuracy Gain</td><td>39%</td><td>Top-1 accuracy improvement over baseline</td></tr>"
            f"</table>\n```\n\n"
            f"# Part 2: Innovation Deep Dive\n"
            f"Each innovation as ###: name → core idea → difference → implementation → evidence.\n\n"
            f"## Format\n"
            f"- # for 2 parts, ### for subsections, ### title on its own line\n"
            f"- Bold key numbers with **\n"
            f"- 5-8 substantive sentences per innovation\n\n"
            f"## Analysis Data\n"
            f"{combined[:12000]}"
        )

    response = await acompletion(
        model="openai/deepseek-v4-pro",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_API_BASE,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _print_validation(results: dict) -> None:
    """Pretty-print validation results."""
    print(f"\n{'─' * 60}")
    print("VALIDATION RESULTS")
    print(f"{'─' * 60}")

    print(f"\nSections found ({len(results['sections_found'])}):")
    for s in results["sections_found"]:
        clean = re.sub(r'<[^>]+>', '', s)[:80]
        print(f"  • {clean}")

    if results["failures"]:
        print(f"\n❌ FAILURES ({len(results['failures'])}):")
        for f in results["failures"]:
            print(f"  - {f}")
    else:
        print("\n✅ All checks passed!")

    if results.get("warnings"):
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for w in results["warnings"]:
            print(f"  - {w}")

    if results["passed"]:
        print("\n🏁 RESULT: PASSED")
    else:
        print("\n🏁 RESULT: FAILED")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline test for paperqa_skill report generation"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use mock data (no API keys required)")
    parser.add_argument("--render-only", action="store_true",
                        help="Skip LLM, re-render from cached report text")
    parser.add_argument("--cache-dir", default=str(DEFAULT_QA_DIR),
                        help="Directory with cached PaperQA2 results (default: demo_output/2312)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory")
    parser.add_argument("--title", "-t", default="2312",
                        help="Paper title")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect answers ────────────────────────────────────────────────
    if args.quick:
        print("[QUICK MODE] Using mock data (no API calls)")
        combined = "\n\n".join(MOCK_ANSWERS)
    else:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_dir():
            print(f"ERROR: Directory not found: {cache_dir}")
            sys.exit(1)

        print(f"Loading QA results from: {cache_dir}")
        answers = []
        for fpath in sorted(cache_dir.glob("*_*.txt")):
            if "_report_" in fpath.name:
                continue
            content = fpath.read_text(encoding="utf-8")
            a_start = content.find("=== ANSWER ===")
            f_start = content.find("=== FORMATTED ANSWER ===")
            if a_start >= 0 and f_start > a_start:
                ans = content[a_start + 14:f_start].strip()
                if ans:
                    answers.append(ans)
                    print(f"  Loaded: {fpath.name} ({len(ans)} chars)")

        if not answers:
            print("ERROR: No QA results found. Use --quick or provide valid cache_dir.")
            sys.exit(1)
        combined = "\n\n".join(answers)

    print(f"\nCombined analysis: {len(combined)} chars")
    print(f"Title: {args.title}")

    # ── Run pipeline ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RUNNING PIPELINE")
    print(f"{'=' * 60}")

    html_path = await run_pipeline(
        combined=combined,
        paper_title=args.title,
        output_dir=output_dir,
        lang="zh",
        skip_llm=args.render_only,
    )

    # ── Validate ───────────────────────────────────────────────────────
    expected = ["简述", "问题与挑战", "方法概览", "架构流程", "关键效果"]
    results = _validate_html(html_path, expected)
    _print_validation(results)

    # ── Show snippet of first section for manual verification ──────────
    if results["sections_found"]:
        print(f"\n{'─' * 60}")
        print("FIRST SECTION PREVIEW")
        print(f"{'─' * 60}")
        validator = ReportValidator()
        with open(html_path, "r", encoding="utf-8") as f:
            validator.feed(f.read())
        validator.close()
        if validator.sections:
            first = validator.sections[0]
            print(f"  Title: {first.title[:100]}")
            print(f"  Body:  {(' '.join(first.body_parts))[:200]}...")
            print(f"  Body length: {len(' '.join(first.body_parts))} chars")

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
