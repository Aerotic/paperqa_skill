---
name: paperqa-skill
description: >
  Academic paper analysis pipeline using PaperQA2. Generates bilingual
  (Chinese/English) HTML + plain-text technical reports with metrics,
  architecture flow diagrams, and extracted figure descriptions.
  Use when the user provides a PDF URL or local path and wants in-depth
  paper analysis, report generation, or metric extraction.
---

# PaperQA2 Skill

Analyze academic papers and generate bilingual technical reports.

**Stack**: DeepSeek (text Q&A + report generation) + Qwen3-VL-Plus (multimodal
figure descriptions) + st-all-mpnet-base-v2 (local embeddings).

## Trigger

Load this skill when the user:
- Provides a PDF URL or local path and asks for analysis / a report
- Uses phrases like "analyze this paper", "论文分析", "paper analysis",
  "generate paper report", "论文报告"
- Wants metrics, innovations, or architecture extracted from a paper
- Needs to download a paper and then analyze it

## Pre-flight: Configuration Check (MANDATORY)

**Before any analysis, you MUST verify API keys are configured.** The skill will
fail silently or produce degraded output if keys are missing.

### Step 1: Run `check_config()`

This checks for required keys across three sources (priority: env var > config file > default):

| Key | Env Var | Required | Purpose |
|-----|---------|----------|---------|
| DeepSeek | `DEEPSEEK_API_KEY` | **Yes** | Text Q&A, report generation, translation |
| Bailian | `BAILIAN_API_KEY` | No (only for multimodal) | Figure descriptions via Qwen3-VL-Plus |

Configuration is loaded from `~/.config/paperqa_skill/paperqa_skill_config.json`,
with environment variables taking priority.

### Step 2: If keys are missing

**Tell the user immediately** — do NOT proceed with analysis. The user has two options:

**Option A — Quick (env vars, one-time):**

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-key"
$env:BAILIAN_API_KEY = "sk-your-key"   # optional, skip for text-only mode

# Linux / macOS
export DEEPSEEK_API_KEY="sk-your-key"
export BAILIAN_API_KEY="sk-your-key"   # optional
```

**Option B — Persistent (config file, survives restarts):**

```bash
python -m paperqa_skill --configure
```

This launches an interactive wizard that saves to `~/.config/paperqa_skill/paperqa_skill_config.json`.

### Step 3: Re-verify

After the user sets their keys, call `check_config()` again. Only proceed when
it returns an empty list.

## Primary Entry Point

### `full_pipeline(pdf_source, paper_title="", queries=None, output_dir=None, multimodal=True) -> tuple[str, str]`

This is the **main function** — the only one you usually need.

**Inputs**:
- `pdf_source`: URL (auto-downloaded) or local file path
- `paper_title`: paper title for the report header
- `queries`: custom query list (default: built-in overview + innovation queries)
- `output_dir`: where reports go (default: same dir as PDF)
- `multimodal`: enable figure extraction + Qwen3-VL-Plus descriptions

**Returns**: `(zh_html_path, en_html_path)` — paths to the Chinese and English HTML reports.

**Pipeline steps** (handled automatically):
1. Download PDF if URL, resolve to local path
2. Run PaperQA2 with the configured queries against the PDF
3. Generate a **Chinese** technical report via DeepSeek (focused, high-quality)
4. **Translate** Chinese report to English via a separate LLM call (structural fidelity)
5. Extract original figures from the PDF, describe them with Qwen3-VL-Plus
6. Generate architecture flow diagrams as inline SVG (zh + en)
7. Produce HTML reports (stat cards, bar charts, flow SVGs, embedded figures)
8. Produce plain-text reports (aligned metrics table, clean prose)

**Output files** per run:
| File | Content |
|------|---------|
| `{stem}_report_zh.html` | Chinese HTML — stat cards, bar charts, flow SVG, embedded figures |
| `{stem}_report_en.html` | English HTML — translated, same structure |
| `{stem}_report_zh.txt` | Chinese plain text with formatted metrics table |
| `{stem}_report_en.txt` | English plain text with formatted metrics table |

## Secondary Functions (use only when full_pipeline is insufficient)

### `analyze_paper(pdf_source, queries, output_dir, multimodal) -> dict[str, str]`
Run custom queries against a single paper. Returns `{query_key: output_file_path}`.
Use when the user wants specific questions answered rather than a full report.

### `download_pdf(url, output_dir, filename) -> str`
Download PDF from arXiv, ScienceDirect, DOI, handle.net, etc. Returns local path.

### `resolve_pdf_source(source, output_dir, filename) -> str`
Smart resolver: downloads if URL, returns path if local file.

### `configure(config_dir) -> dict`
Interactive wizard to set API keys.

### `check_config(config) -> list[str]`
Returns list of missing required config keys.

## Usage Examples

### Python (synchronous wrapper available via asyncio.run)

```python
import asyncio
from paperqa_skill import full_pipeline

zh_html, en_html = asyncio.run(full_pipeline(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",
    paper_title="My Paper Title",
    multimodal=True,
))
```

### CLI

```bash
python -m paperqa_skill paper.pdf --full --title "Title" --output ./reports
python -m paperqa_skill https://arxiv.org/pdf/2401.12345.pdf --full -t "Title"
python -m paperqa_skill --configure   # set API keys
```

## Important Notes

- PDF sources support URLs (arxiv, ScienceDirect, DOI, handle.net) and local paths
- The bilingual strategy: **Chinese first, then translate to English** — ensures
  the English version faithfully mirrors the Chinese structure and data
- Multimodal mode requires `BAILIAN_API_KEY` — if unavailable, pass `multimodal=False`
  and figure descriptions will be skipped
- The module is at `paperqa_skill/__init__.py` — all imports come from the package
- HTML reports embed figures as base64 — they are fully self-contained
