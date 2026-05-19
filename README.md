# PaperQA2 论文分析 Skill

> 可复用的论文深度分析流水线 — DeepSeek (文本 Q&A) + Qwen3-VL-Plus (多模态图片描述) + all-mpnet-base-v2 (本地嵌入)
>
> 输入 PDF，输出中英双语 HTML + 纯文本技术报告，含指标卡片、架构流程图、论文原图描述。
>
> 基于[Paper-QA2](https://github.com/future-house/paper-qa)
>
> [English version](README.en.md)

## 目录

1. [架构概览](#1-架构概览)
2. [环境要求](#2-环境要求)
3. [安装与配置](#3-安装与配置)
4. [快速开始](#4-快速开始)
5. [Python API](#5-python-api)
6. [命令行使用](#6-命令行使用)
7. [PDF 来源支持](#7-pdf-来源支持)
8. [输出文件](#8-输出文件)
9. [已知问题](#9-已知问题)

---

## 1. 架构概览

```
┌──────────────────────────────────────────────────────┐
│                    paperqa_skill                      │
├──────────────────────────────────────────────────────┤
│  Step 1: PDF → 文本分块 + 图片提取 (PyMuPDF)          │
│  Step 2: 本地嵌入向量化 (all-mpnet-base-v2)           │
│  Step 3: 多模态图片增强 (Qwen3-VL-Plus @ Bailian)     │
│  Step 4: 文本 Q&A (DeepSeek Chat)                     │
│  Step 5: HTML 图文报告 + 纯文本报告生成                │
└──────────────────────────────────────────────────────┘
```

### 双 LLM 架构

| 组件 | 模型 | 用途 | API |
|------|------|------|-----|
| 文本 LLM | DeepSeek Chat | 文本 Q&A、报告生成、翻译 | `api.deepseek.com` |
| 多模态 LLM | Qwen3-VL-Plus | 图片/图表描述 | `dashscope-intl.aliyuncs.com` |
| 嵌入模型 | all-mpnet-base-v2 | 本地语义搜索 | 本地 (sentence-transformers) |

---

## 2. 环境要求

- **OS:** Windows (已验证), Linux/macOS 理论可运行
- **Python:** ≥ 3.10 (实测 3.14)
- **依赖:** `paper-qa>=2026.3.18`, `paper-qa[local]`, `litellm`, `pymupdf`

- **API Keys:**

| Key | 必需 | 获取地址 |
|-----|------|---------|
| DeepSeek | **是** | https://platform.deepseek.com |
| Bailian (百炼) | 否 (关闭多模态时) | https://bailian.console.aliyun.com |

---

## 3. 安装与配置

```bash
pip install paper-qa
pip install paper-qa[local]
```

### 配置 API Key

**方式一：环境变量 (推荐)**

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-key"
$env:BAILIAN_API_KEY = "sk-your-key"        # 可选

# Linux / macOS
export DEEPSEEK_API_KEY="sk-your-key"
export BAILIAN_API_KEY="sk-your-key"        # 可选

# 可选：自定义 API Base URL（默认值如下，一般无需修改）
$env:DEEPSEEK_API_BASE = "https://api.deepseek.com"
$env:BAILIAN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

**方式二：交互式配置 (持久化)**

```bash
python -m paperqa_skill --configure
```

配置保存至 `~/.config/paperqa_skill/paperqa_skill_config.json`，下次启动自动加载。

---

## 4. 快速开始

### 4.1 输入方式

所有函数同时支持 **PDF URL**（自动下载）和 **本地路径**：

```python
from paperqa_skill import resolve_pdf_source

# URL → 自动下载
local = resolve_pdf_source("https://arxiv.org/pdf/2401.12345.pdf")

# 本地路径 → 直接使用
local = resolve_pdf_source(r"C:\papers\paper.pdf")
```

支持的 URL 类型：直接 PDF 链接、arXiv、DOI、ScienceDirect、handle.net 等（详见 [PDF 来源支持](#7-pdf-来源支持)）。

### 4.2 单篇分析

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

### 4.3 完整流水线 (含 HTML 报告)

```python
import asyncio
from paperqa_skill import full_pipeline

zh_html, en_html = asyncio.run(full_pipeline(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",
    paper_title="My Paper Title",
    multimodal=True,
))
```

### 4.4 命令行

```bash
# 基础分析
python -m paperqa_skill paper.pdf --title "论文标题"

# 完整流水线
python -m paperqa_skill paper.pdf --full --title "论文标题" --output ./reports

# 禁用多模态 (无需 Bailian Key)
python -m paperqa_skill paper.pdf --full --no-multimodal

# 从缓存中间结果重新生成报告（无需 PDF）
python -m paperqa_skill ./reports/2312 --regenerate --title "MobileGPT"
```

### 4.5 示例脚本

`example/run_report.py` 是开箱即用的完整示例，自动管理输出目录和文件命名：

```bash
# 分析默认 PDF (../2312.pdf)，输出到 demo_output/2312/
python example/run_report.py

# 分析指定 PDF，指定标题
python example/run_report.py paper.pdf -t "论文标题"

# 禁用多模态（跳过图片提取）
python example/run_report.py paper.pdf --no-multimodal

# 从缓存中间结果重生成报告
python example/run_report.py demo_output/2312 --regenerate -t "MobileGPT"

# 自定义输出目录
python example/run_report.py paper.pdf -o ./my_reports
```

### 4.6 批量分析 + 选择性生成图文报告

典型工作流：先批量分析大量论文（非多模态，快速低成本），再对重点论文生成图文报告。

```python
import asyncio
from paperqa_skill import analyze_paper, regenerate_reports

# Step 1: 批量分析（非多模态模式，快且便宜）
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
for pdf in papers:
    asyncio.run(analyze_paper(
        pdf_source=pdf,
        queries=["What is the key problem and methodology?", 
                 "What are the major innovations and quantitative results?"],
        multimodal=False,  # 跳过图片提取
    ))

# Step 2: 对重点论文生成图文报告（从缓存中间结果 + PDF 图片）
zh_html, en_html = asyncio.run(regenerate_reports(
    cache_dir="./output/paper1",     # 包含 analyze_paper 输出的目录
    paper_title="My Key Paper",
    pdf_path="./papers/paper1.pdf",  # 用于提取论文原图
))

---

## 5. Python API

### `full_pipeline(pdf_source, paper_title, queries, output_dir, multimodal)`

**核心函数**，端到端执行：分析 → 报告生成 → 翻译 → 图片提取。

| 参数 | 类型 | 说明 |
|------|------|------|
| `pdf_source` | `str` | PDF URL 或本地路径 |
| `paper_title` | `str` | 论文标题 (用于报告) |
| `queries` | `list[str]` | 自定义问题列表 (默认: 概览 + 创新点深度分析) |
| `output_dir` | `str` | 输出目录 (默认: PDF 所在目录) |
| `multimodal` | `bool` | 启用图提取与 Qwen3-VL-Plus 描述 |

**返回:** `(zh_html_path, en_html_path)` 中英文 HTML 报告路径元组。

### `analyze_paper(pdf_source, queries, output_dir, multimodal)`

对单篇 PDF 执行自定义查询，返回 `{query_key: output_file_path}`。

### `regenerate_reports(cache_dir, paper_title, pdf_path, output_dir)`

从已缓存的 PaperQA2 中间结果重新生成 HTML/TXT 报告，**无需重新分析 PDF**。

| 参数 | 类型 | 说明 |
|------|------|------|
| `cache_dir` | `str` | 包含 `analyze_paper` 输出文件 (`*_*.txt`) 的目录 |
| `paper_title` | `str` | 论文标题 (用于报告) |
| `pdf_path` | `str` | PDF 文件路径。提供时提取论文原图嵌入报告；`None` 则跳过图片 |
| `output_dir` | `str` | 输出目录 (默认同 cache_dir) |

**返回:** `(zh_html_path, en_html_path)` 中英文 HTML 报告路径元组。

**使用场景：**
- `--no-multimodal` 批量分析多篇论文后，挑选重点论文补充图片生成图文报告
- 修改报告生成逻辑后，快速重生成报告而不重新跑 PaperQA2 查询

### `resolve_pdf_source(source, output_dir, filename)`

解析 PDF 源：URL → 自动下载；本地路径 → 直接返回。

### `make_settings(paper_directory, multimodal, agent_type)`

创建 PaperQA2 Settings 实例。`agent_type="fake"` 禁用多步 agent 循环，推荐用于单篇分析。

---

## 6. 命令行使用

```
python -m paperqa_skill [-h] [--title TITLE] [--output DIR] [--no-multimodal] [--full] [--regenerate] source

位置参数:
  source                PDF URL/本地路径，或 --regenerate 时的缓存目录

可选参数:
  -h, --help            显示帮助
  -t, --title TITLE     论文标题
  -o, --output DIR      输出目录
  --no-multimodal       禁用多模态 (无需 Bailian Key)
  -f, --full            运行完整流水线 (含 HTML 报告)
  -r, --regenerate      从缓存中间结果重新生成报告 (source 为缓存目录)
```

**示例：**

```bash
# 批量分析 + 重生成图文报告
python -m paperqa_skill paper.pdf --full --no-multimodal    # 第一步：无图分析
python -m paperqa_skill ./output/paper --regenerate -t "标题"  # 第二步：生成图文报告
```

---

## 7. PDF 来源支持

| 来源类型 | 示例 | 策略 |
|---------|------|------|
| 直接 PDF 链接 | `https://arxiv.org/pdf/2401.12345.pdf` | 直接下载 |
| arXiv 摘要页 | `https://arxiv.org/abs/2401.12345` | 转 `/pdf/` 链接下载 |
| DOI 页面 | `https://doi.org/10.1145/3636534.3690682` | 提取 `citation_pdf_url` |
| ScienceDirect | `https://www.sciencedirect.com/...` | HTML 扫描 PDF 链接 |
| handle.net | `https://hdl.handle.net/2031/...` | 跟随重定向 → 提取 PDF |
| 通用 HTML | 含 PDF 链接的页面 | `href` 扫描 + OA 匹配 |

> **注意:** ScienceDirect 等商业出版社常返回 403。若自动下载失败，请手动下载后传入本地路径。

---

## 8. 输出文件

每次运行 `full_pipeline()` 生成 4 个文件：

| 文件 | 内容 |
|------|------|
| `{stem}_report_zh.html` | 中文 HTML — 指标卡片、架构流程图、论文原图 (base64 嵌入) |
| `{stem}_report_en.html` | 英文 HTML — 翻译版本，结构一致 |
| `{stem}_report_zh.txt` | 中文纯文本 — 含对齐指标表格 |
| `{stem}_report_en.txt` | 英文纯文本 — 含对齐指标表格 |

---

## 9. 已知问题

- **DeepSeek 成本警告**: `Failed to calculate cost for deepseek-v4-flash` — 无害，LiteLLM 价格映射缺失，不影响功能
- **image_url 上下文错误**: `unknown variant 'image_url', expected 'text'` — DeepSeek 纯文本模型不支持图像消息，系统自动回退无媒体模式重试，不影响最终结果
- **ScienceDirect 403**: 商业出版社 PDF 需通过 OA 仓库获取，或手动下载
