# PaperQA2 Paper Analysis Skill

> Reusable deep paper analysis pipeline — DeepSeek (text Q&A) + Qwen3-VL-Plus (multimodal figure descriptions) + all-mpnet-base-v2 (local embeddings)
>
> Input a PDF, output bilingual (Chinese/English) HTML + plain-text technical reports with metric cards, architecture flow diagrams, and extracted figure descriptions.
> 
> Based on [Paper-QA2](https://github.com/future-house/paper-qa)
>
> [中文版](README.md)

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Requirements](#2-requirements)
3. [Installation & Configuration](#3-installation--configuration)
4. [Quick Start](#4-quick-start)
5. [Python API](#5-python-api)
6. [CLI Usage](#6-cli-usage)
7. [Supported PDF Sources](#7-supported-pdf-sources)
8. [Output Files](#8-output-files)
9. [Known Issues](#9-known-issues)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    paperqa_skill                      │
├──────────────────────────────────────────────────────┤
│  Step 1: PDF → text chunking + figure extraction     │
│  Step 2: local embedding (all-mpnet-base-v2)         │
│  Step 3: multimodal figure enrichment (Qwen3-VL-Plus)│
│  Step 4: text Q&A (DeepSeek Chat)                    │
│  Step 5: HTML + plain-text report generation          │
└──────────────────────────────────────────────────────┘
```

### Dual LLM Architecture

| Component | Model | Purpose | API |
|-----------|-------|---------|-----|
| Text LLM | DeepSeek Chat | Text Q&A, report generation, translation | `api.deepseek.com` |
| Multimodal LLM | Qwen3-VL-Plus | Figure/chart descriptions | `dashscope-intl.aliyuncs.com` |
| Embedding | all-mpnet-base-v2 | Local semantic search | Local (sentence-transformers) |

---

## 2. Requirements

- **OS:** Windows (verified), Linux/macOS should work
- **Python:** ≥ 3.10 (tested on 3.14)
- **Dependencies:** `paper-qa>=2026.3.18`, `paper-qa[local]`, `litellm`, `pymupdf`

- **API Keys:**

| Key | Required | Get from |
|-----|----------|----------|
| DeepSeek | **Yes** | https://platform.deepseek.com |
| Bailian | No (skip with `--no-multimodal`) | https://bailian.console.aliyun.com |

---

## 3. Installation & Configuration

```bash
pip install paper-qa
pip install paper-qa[local]
```

### Configure API Keys

**Option A: Environment Variables (recommended)**

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-key"
$env:BAILIAN_API_KEY = "sk-your-key"        # optional

# Linux / macOS
export DEEPSEEK_API_KEY="sk-your-key"
export BAILIAN_API_KEY="sk-your-key"        # optional

# Optional: custom API Base URL (defaults shown below, usually not needed)
$env:DEEPSEEK_API_BASE = "https://api.deepseek.com"
$env:BAILIAN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

**Option B: Interactive Setup (persistent)**

```bash
python -m paperqa_skill --configure
```

Saves to `~/.config/paperqa_skill/paperqa_skill_config.json` and auto-loads on next start.

---

## 4. Quick Start

### 4.1 Input Methods

All functions accept both **PDF URLs** (auto-download) and **local paths**:

```python
from paperqa_skill import resolve_pdf_source

# URL → auto-download
local = resolve_pdf_source("https://arxiv.org/pdf/2401.12345.pdf")

# Local path → use directly
local = resolve_pdf_source(r"C:\papers\paper.pdf")
```

Supported URL types: direct PDF links, arXiv, DOI, ScienceDirect, handle.net, etc. (see [Supported PDF Sources](#7-supported-pdf-sources)).

### 4.2 Single Paper Analysis

```python
import asyncio
from paperqa_skill import analyze_paper

results = asyncio.run(analyze_paper(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",
    queries=[
        "What is the key problem? Describe the methodology and key results.",
        "Explain the experimental setup, benchmarks, and evaluation metrics.",
    ],
    multimodal=True,
))
# returns {"query_key": "output_file_path", ...}
```

### 4.3 Full Pipeline (with HTML Reports)

```python
import asyncio
from paperqa_skill import full_pipeline

zh_html, en_html = asyncio.run(full_pipeline(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",
    paper_title="My Paper Title",
    multimodal=True,
))
```

### 4.4 Command Line

```bash
# Basic analysis
python -m paperqa_skill paper.pdf --title "Paper Title"

# Full pipeline
python -m paperqa_skill paper.pdf --full --title "Paper Title" --output ./reports

# Disable multimodal (no Bailian key needed)
python -m paperqa_skill paper.pdf --full --no-multimodal
```

---

## 5. Python API

### `full_pipeline(pdf_source, paper_title, queries, output_dir, multimodal)`

**Core function** — end-to-end: analysis → report generation → translation → figure extraction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pdf_source` | `str` | PDF URL or local path |
| `paper_title` | `str` | Paper title (for report header) |
| `queries` | `list[str]` | Custom queries (default: overview + innovation deep-dive) |
| `output_dir` | `str` | Output directory (default: same dir as PDF) |
| `multimodal` | `bool` | Enable figure extraction + Qwen3-VL-Plus descriptions |

**Returns:** `(zh_html_path, en_html_path)` — tuple of Chinese and English HTML report paths.

### `analyze_paper(pdf_source, queries, output_dir, multimodal)`

Run custom queries against a PDF. Returns `{query_key: output_file_path}`.

### `resolve_pdf_source(source, output_dir, filename)`

Resolve PDF source: URL → auto-download; local path → return directly.

### `make_settings(paper_directory, multimodal, agent_type)`

Create PaperQA2 Settings instance. `agent_type="fake"` disables multi-step agent loop (recommended for single paper analysis).

---

## 6. CLI Usage

```
python -m paperqa_skill [-h] [--title TITLE] [--output DIR] [--no-multimodal] [--full] source

Positional:
  source                PDF URL or local file path

Options:
  -h, --help            Show help
  -t, --title TITLE     Paper title
  -o, --output DIR      Output directory
  --no-multimodal       Disable multimodal (no Bailian key needed)
  -f, --full            Run full pipeline (with HTML reports)
```

---

## 7. Supported PDF Sources

| Source Type | Example | Strategy |
|------------|---------|----------|
| Direct PDF link | `https://arxiv.org/pdf/2401.12345.pdf` | Direct download |
| arXiv abstract | `https://arxiv.org/abs/2401.12345` | Convert to `/pdf/` link |
| DOI page | `https://doi.org/10.1145/3636534.3690682` | Extract `citation_pdf_url` |
| ScienceDirect | `https://www.sciencedirect.com/...` | Scan HTML for PDF links |
| handle.net | `https://hdl.handle.net/2031/...` | Follow redirect → extract PDF |
| Generic HTML | Any page with PDF links | `href` scan + OA pattern matching |

> **Note:** Commercial publishers like ScienceDirect often return 403. If auto-download fails, download manually and pass the local path.

---

## 8. Output Files

Each `full_pipeline()` run produces 4 files:

| File | Content |
|------|---------|
| `{stem}_report_zh.html` | Chinese HTML — metric cards, flow diagram, figures (base64 embedded) |
| `{stem}_report_en.html` | English HTML — translated, same structure |
| `{stem}_report_zh.txt` | Chinese plain text — with aligned metrics table |
| `{stem}_report_en.txt` | English plain text — with aligned metrics table |

---

## 9. Known Issues

- **DeepSeek cost warning**: `Failed to calculate cost for deepseek-v4-flash` — harmless, LiteLLM price mapping gap, does not affect functionality
- **image_url context error**: `unknown variant 'image_url', expected 'text'` — DeepSeek is text-only and cannot process image messages; the system auto-retries without media, no impact on final results
- **ScienceDirect 403**: Commercial publisher PDFs require OA repository access or manual download
