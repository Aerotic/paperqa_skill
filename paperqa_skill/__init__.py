"""
PaperQA2 论文分析 Skill — 可复用的论文深度分析流水线

架构: DeepSeek (文本Q&A) + Qwen3-VL-Plus (多模态图片描述) + st-all-mpnet-base-v2 (本地嵌入)
输入: PDF URL (自动下载) 或本地 PDF 路径
"""

__version__ = "2.0.0"

import os, asyncio, json, re, ssl, uuid, sys, tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import urllib.request

# ---------------------------------------------------------------------------
# 配置管理 (JSON 配置文件 + 环境变量后备 + 交互式提示)
# ---------------------------------------------------------------------------

CONFIG_FILENAME = "paperqa_skill_config.json"


def _default_config_dir() -> str:
    """跨平台配置文件目录: ~/.config/paperqa_skill/"""
    base = os.environ.get(
        "XDG_CONFIG_HOME",
        os.path.join(str(Path.home()), ".config"),
    )
    return os.path.join(base, "paperqa_skill")


def get_config_path(config_dir: Optional[str] = None) -> str:
    """返回配置文件完整路径"""
    return os.path.join(config_dir or _default_config_dir(), CONFIG_FILENAME)


_CONFIG_SCHEMA = {
    "deepseek_api_key": {
        "env": "DEEPSEEK_API_KEY",
        "prompt": "请输入 DeepSeek API Key",
        "default": "",
    },
    "deepseek_api_base": {
        "env": "DEEPSEEK_API_BASE",
        "prompt": "请输入 DeepSeek API Base URL",
        "default": "https://api.deepseek.com",
    },
    "bailian_api_key": {
        "env": "BAILIAN_API_KEY",
        "prompt": "请输入百炼 (DashScope/Bailian) API Key (用于 Qwen3-VL-Plus 多模态)",
        "default": "",
    },
    "bailian_api_base": {
        "env": "BAILIAN_API_BASE",
        "prompt": "请输入百炼 API Base URL",
        "default": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
    "paper_dir": {
        "env": "PAPER_DIR",
        "prompt": "请输入 PDF 工作目录",
        "default": str(Path(tempfile.gettempdir()) / "pqa_test"),
    },
}


def load_config(config_dir: Optional[str] = None) -> dict:
    """加载配置: 配置文件 > 环境变量 > 内置默认值。

    - 首次加载时不会交互式提示 (提示由 configure() 处理)
    - 调用方可通过 check_config() 验证关键项是否齐全
    """
    cfg_path = get_config_path(config_dir)
    config = {}

    # 1) 从文件加载已有配置
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[paperqa_skill] Warning: failed to read config {cfg_path}: {e}")

    # 2) 环境变量覆盖 (最高优先级)
    for key, meta in _CONFIG_SCHEMA.items():
        env_val = os.environ.get(meta["env"])
        if env_val:
            config[key] = env_val

    # 3) 用内置默认值填充完全缺失的项
    for key, meta in _CONFIG_SCHEMA.items():
        if key not in config or not config.get(key):
            config[key] = meta["default"]

    return config


def check_config(config: Optional[dict] = None) -> list[str]:
    """检查配置中缺失的关键项，返回缺失项名称列表

    Args:
        config: 配置字典 (默认使用全局配置)

    Returns:
        缺失的 key 列表 (空列表表示一切正常)
    """
    if config is None:
        config = _CFG
    missing = []
    if not config.get("deepseek_api_key"):
        missing.append("deepseek_api_key (DEEPSEEK_API_KEY)")
    if not config.get("bailian_api_key"):
        missing.append("bailian_api_key (BAILIAN_API_KEY)")
    return missing


def save_config(config: dict, config_dir: Optional[str] = None) -> str:
    """保存配置到 JSON 文件"""
    cfg_dir = config_dir or _default_config_dir()
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, CONFIG_FILENAME)

    # 只保留 schema 中定义的字段
    filtered = {k: config.get(k, "") for k in _CONFIG_SCHEMA}
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        print(f"[paperqa_skill] Config saved: {cfg_path}")
    except OSError as e:
        print(f"[paperqa_skill] Warning: failed to save config {cfg_path}: {e}")
    return cfg_path


# ---------------------------------------------------------------------------
# 全局配置实例 (模块启动时加载一次)
# ---------------------------------------------------------------------------
_CFG = load_config()

DEEPSEEK_API_KEY = _CFG["deepseek_api_key"]
DEEPSEEK_API_BASE = _CFG["deepseek_api_base"]
BAILIAN_API_KEY = _CFG["bailian_api_key"]
BAILIAN_API_BASE = _CFG["bailian_api_base"]
PAPER_DIR = _CFG["paper_dir"]

# ---------------------------------------------------------------------------
# PDF 下载 (URL → 本地文件)
# ---------------------------------------------------------------------------

def _is_url(source: str) -> bool:
    """判断输入是 URL 还是本地路径"""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def download_pdf(
    url: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """从 URL 下载 PDF，支持多种来源。

    - 直接 PDF 链接 (.pdf) → 直接下载
    - 学术网站 (ScienceDirect, doi.org, handle.net, arXiv) → 提取 PDF 链接后下载
    - 通用 HTML 页面 → 扫描页面寻找 PDF 链接

    Args:
        url: PDF 或论文页面 URL
        output_dir: 下载目录 (默认 $PAPER_DIR / downloads)
        filename: 自定义文件名 (不含 .pdf)

    Returns:
        本地 PDF 文件路径

    Raises:
        ValueError: 无法从 URL 获取有效的 PDF
    """
    output_dir = output_dir or os.path.join(PAPER_DIR, "downloads")
    os.makedirs(output_dir, exist_ok=True)

    # 确定文件名
    if not filename:
        # 从 URL 提取一个合理的文件名
        parsed = urlparse(url)
        basename = Path(parsed.path).stem or "paper"
        # 去除多余的 hash/query 字符
        basename = re.sub(r"[^a-zA-Z0-9_-]", "_", basename)[:80]
        if not basename or basename == "_":
            basename = f"paper_{uuid.uuid4().hex[:8]}"
        filename = basename

    out_path = os.path.join(output_dir, f"{filename}.pdf")

    # 如果是直接 PDF 链接
    if url.lower().endswith(".pdf") or "/pdf" in url.lower() or "pdf" in urlparse(url).path.lower():
        pdf_path = await _download_direct(url, out_path)
        if pdf_path:
            return pdf_path

    # arXiv HTML 页面 (/html/XXXX.XXXXX) → 替换为 /pdf/ 链接
    parsed = urlparse(url)
    if "arxiv.org" in parsed.netloc and parsed.path.startswith("/html/"):
        pdf_url = f"https://arxiv.org/pdf/{parsed.path.split('/html/')[1]}"
        print(f"[paperqa_skill] Trying arXiv PDF: {pdf_url}")
        pdf_path = await _download_direct(pdf_url, out_path)
        if pdf_path:
            return pdf_path

    # ScienceDirect / 学术页面 → 尝试 HTML 提取
    html = await _fetch_page(url)
    if html:
        # 尝试从 HTML 提取 PDF 链接
        pdf_urls = _extract_pdf_urls(html, url)
        for pu in pdf_urls:
            pdf_path = await _download_direct(pu, out_path)
            if pdf_path:
                return pdf_path

        # OpenAlex / handle.net 重定向处理
        oa_match = re.search(r'href=["\']([^"\']*oa[^"\']*\.pdf[^"\']*)["\']', html, re.I)
        if oa_match:
            pdf_url = urllib.parse.urljoin(url, oa_match.group(1))
            pdf_path = await _download_direct(pdf_url, out_path)
            if pdf_path:
                return pdf_path

    raise ValueError(
        f"Could not obtain a valid PDF from: {url}\n"
        "Try providing a direct PDF URL, or download the PDF manually and use the local path."
    )


async def _fetch_page(url: str, timeout: int = 30) -> Optional[str]:
    """获取 HTML 页面内容 (异步 wrapper)"""
    ctx = _ssl_ctx()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[paperqa_skill] Warning: failed to fetch page {url}: {e}")
        return None


def _extract_pdf_urls(html: str, base_url: str) -> list[str]:
    """从 HTML 中提取所有 PDF 相关的 URL"""
    import urllib.parse
    urls = []
    patterns = [
        r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
        r'citation_pdf_url[^>]+content=["\']([^"\']+)["\']',
        r'pdfUrl[\s:=]+["\']([^"\']+)["\']',
        r'data-pdf-url=["\']([^"\']+)["\']',
        r'rel=["\']download["\'][^>]+href=["\']([^"\']+)["\']',
    ]
    for pat in patterns:
        for m in re.finditer(pat, html, re.I):
            raw = m.group(1)
            absolute = urllib.parse.urljoin(base_url, raw)
            if absolute not in urls:
                urls.append(absolute)
    return urls


async def _download_direct(url: str, out_path: str) -> Optional[str]:
    """直接下载 PDF 文件"""
    ctx = _ssl_ctx()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
        "Referer": url,
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            data = resp.read()
        if data[:4] == b"%PDF":
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"[paperqa_skill] Downloaded PDF ({len(data)} bytes) -> {out_path}")
            return out_path
        else:
            print(f"[paperqa_skill] Response is not a PDF (header={data[:20]})")
            return None
    except Exception as e:
        print(f"[paperqa_skill] Download failed: {e}")
        return None


async def resolve_pdf_source(
    source: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """解析 PDF 输入源: URL 或本地路径 (异步)

    - 传入 URL (http/https) → 自动下载 PDF
    - 传入本地路径 → 直接使用

    Args:
        source: PDF URL 或本地文件路径
        output_dir: 下载/输出目录
        filename: 自定义文件名 (不含扩展名)

    Returns:
        本地 PDF 文件路径
    """
    if _is_url(source):
        return await download_pdf(source, output_dir, filename)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {source}")
        return str(path.absolute())


def resolve_pdf_source_sync(
    source: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """解析 PDF 输入源: URL 或本地路径 (同步 wrapper)"""
    return asyncio.run(resolve_pdf_source(source, output_dir, filename))


# ---------------------------------------------------------------------------
# PaperQA2 Settings 工厂
# ---------------------------------------------------------------------------
def make_settings(
    paper_directory: Optional[str] = None,
    multimodal: bool = True,
    agent_type: str = "fake",
):
    """创建 PaperQA2 Settings 实例 (DeepSeek + Qwen3-VL-Plus + 本地嵌入)"""
    from paperqa import Settings

    deepseek_config = {
        "model_list": [
            {
                "model_name": "openai/deepseek-chat",
                "litellm_params": {
                    "model": "openai/deepseek-chat",
                    "api_key": DEEPSEEK_API_KEY,
                    "api_base": DEEPSEEK_API_BASE,
                },
            }
        ]
    }

    parsing = {"multimodal": False}
    if multimodal:
        parsing = {
            "multimodal": True,
            "enrichment_llm": "openai/qwen3-vl-plus",
            "enrichment_llm_config": {
                "model_list": [
                    {
                        "model_name": "openai/qwen3-vl-plus",
                        "litellm_params": {
                            "model": "openai/qwen3-vl-plus",
                            "api_key": BAILIAN_API_KEY,
                            "api_base": BAILIAN_API_BASE,
                        },
                    }
                ]
            },
        }

    return Settings(
        llm="openai/deepseek-chat",
        llm_config=deepseek_config,
        summary_llm="openai/deepseek-chat",
        summary_llm_config=deepseek_config,
        embedding="st-all-mpnet-base-v2",
        parsing=parsing,
        agent={
            "agent_type": agent_type,
            "index": {"paper_directory": paper_directory or PAPER_DIR},
        },
        answer={
            "evidence_text_only_fallback": True,  # 避免 text-only LLM 收到 image_url
        },
    )


# ---------------------------------------------------------------------------
# 核心分析函数
# ---------------------------------------------------------------------------
async def analyze_paper(
    pdf_source: str,
    queries: list[str],
    output_dir: Optional[str] = None,
    multimodal: bool = True,
) -> dict[str, str]:
    """对一篇 PDF 执行 PaperQA2 分析

    Args:
        pdf_source: PDF URL 或本地文件路径
        queries: 问题列表 (每个问题独立查询)
        output_dir: 输出目录 (默认 $PAPER_DIR / output)
        multimodal: 是否启用多模态

    Returns:
        {query_key: output_file_path} 字典
    """
    import tempfile, shutil
    from paperqa import Docs

    # 使用临时目录作为 PaperQA2 的工作目录，避免索引污染
    work_dir = tempfile.mkdtemp(prefix="pqa_work_")

    try:
        # 解析输入 (URL → 本地下载) 到临时目录
        pdf_path = await resolve_pdf_source(pdf_source, output_dir=work_dir)

        settings = make_settings(
            paper_directory=work_dir,
            multimodal=multimodal,
        )
        output_dir = output_dir or str(Path(pdf_path).parent)
        os.makedirs(output_dir, exist_ok=True)

        docs = Docs()
        docname = await docs.aadd(pdf_path, settings=settings)
        print(f"[paperqa_skill] Added doc: {docname}")

        results = {}
        for i, query in enumerate(queries):
            key = query.strip()[:60].replace(" ", "_").replace("?", "").replace(",", "")
            key = "".join(c for c in key if c.isalnum() or c in "_")
            key = key[:50] or f"output_q{i+1}"

            print(f"\n[paperqa_skill] Query {i+1}/{len(queries)}: {query[:80]}...")
            session = await docs.aquery(
                query,
                settings=settings,
            )
            output_path = os.path.join(output_dir, f"{key}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=== ANSWER ===\n")
                f.write(session.answer)
                f.write("\n\n=== FORMATTED ANSWER ===\n")
                f.write(session.formatted_answer)
                f.write("\n\n=== CONTEXTS ===\n")
                for j, c in enumerate(session.contexts):
                    f.write(f"\n--- Context {j+1} (doc={c.text.name}, score={c.score}) ---\n")
                    f.write(c.context[:600] if c.context else "(no text)")
            print(f"[paperqa_skill] -> {output_path} ({len(session.answer)} chars)")
            results[key] = output_path

        return results
    finally:
        # 清理临时工作目录
        shutil.rmtree(work_dir, ignore_errors=True)


def analyze_paper_sync(
    pdf_source: str,
    queries: list[str],
    output_dir: Optional[str] = None,
    multimodal: bool = True,
) -> dict[str, str]:
    """同步接口包装"""
    return asyncio.run(analyze_paper(pdf_source, queries, output_dir, multimodal))


# ---------------------------------------------------------------------------
# 一键分析 + HTML 报告生成
# ---------------------------------------------------------------------------
async def full_pipeline(
    pdf_source: str,
    paper_title: str = "",
    queries: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    multimodal: bool = True,
) -> tuple[str, str]:
    """完整流水线: 多角度查询 + HTML 图文报告生成

    Args:
        pdf_source: PDF URL 或本地文件路径
        paper_title: 论文标题 (用于报告)
        queries: 问题列表 (默认使用内置标准问题集)
        output_dir: 输出目录
        multimodal: 启用多模态

    Returns:
        (zh_html_path, en_html_path) 中英文 HTML 报告路径元组
    """
    # 先解析 PDF 源，确保文件存在
    pdf_path = await resolve_pdf_source(pdf_source, output_dir=output_dir)
    pdf_stem = Path(pdf_path).stem

    if queries is None:
        queries = [
            # 查询 1: 总览
            f"Provide a comprehensive overview of the paper '{paper_title or pdf_stem}'. "
            "Describe the key problem, proposed approach/methodology, "
            "algorithms or architectural changes, experimental setup, benchmarks, and key results. "
            "Include details from any diagrams, figures, or architectural illustrations if relevant.",
            # 查询 2: 创新点深度剖析
            f"Identify and analyze every major innovation and novel contribution in the paper '{paper_title or pdf_stem}'. "
            "For EACH innovation: (1) Name the innovation, (2) Explain the core technical idea in detail, "
            "(3) Describe how it differs from prior work, (4) Explain the specific implementation/architecture, "
            "(5) Cite quantitative evidence of its effectiveness from the paper's experiments. "
            "Be extremely specific — cite exact numbers, algorithm names, architectural components, and design choices. "
            "List ALL innovations, not just the main one.",
        ]

    output_dir = output_dir or str(Path(pdf_path).parent)

    # Step 1: 分析
    results = await analyze_paper(pdf_path, queries, output_dir, multimodal)

    # Step 2: 收集所有分析结果
    all_answers = []
    for key, outpath in results.items():
        with open(outpath, "r", encoding="utf-8") as f:
            content = f.read()
            answer_start = content.find("=== ANSWER ===")
            formatted_start = content.find("=== FORMATTED ANSWER ===")
            if answer_start >= 0 and formatted_start > answer_start:
                answer_text = content[answer_start + 14:formatted_start].strip()
                all_answers.append(answer_text)

    combined = "\n\n".join(all_answers)

    # Step 3: 生成中文版完整报告 (专注、高质量)
    zh_prompt = (
        f"你是一位顶级学术论文分析专家。请根据以下对论文「{paper_title or pdf_stem}」的分析数据，"
        f"生成一份完整的**中文技术分析报告**。\n\n"
        f"## 输出结构 (严格按顺序, 2个大块)\n\n"
        f"# 第一部分：总览\n"
        f"简洁概括论文全貌，用加粗数字，覆盖问题/挑战/方法/效果。\n"
        f"末尾必须附中文指标表（| 分隔各列，HTML 和纯文本版均以此格式渲染），其应该包含如下列：\n"
        # f"[METRICS]\n"
        f"值、指标名称、含义解释(一句话说清这个指标衡量什么，这个指标的定义)、类别(accuracy/reduction/delta/other)\n"
        f"表格中一行内容的示例（“|”用来分隔不同列的内容）: 39% | 准确率提升 | Top-1准确率相较于baseline的提升幅度，准确率是与ground truth一致的预测数量和总数量的比值 | accuracy\n"
        # f"[/METRICS]\n"
        f"重要：含义解释列不可省略，必须用完整句子描述指标意义，不可仅重复指标名。\n"
        f"### 问题与挑战\n"
        f"### 方法概览\n"
        f"### 架构流程\n"
        f"用文字描述系统架构的完整链路，按步骤编号列出各阶段组件。\n"
        f"### 关键效果\n\n"
        f"# 第二部分：创新点深度剖析\n"
        f"每个主要创新点一个 ### 小节：创新点命名 → 核心思想 → 与已有工作的区别 → 技术实现 → 有效性证据。\n\n"
        f"## 格式要求\n"
        f"- 用 # 标记 2 大块, ### 标记小节\n"
        f"- 关键数字用 **加粗**，技术术语中英文对照\n"
        f"- 创新点每小节 5-8 句有实质内容\n"
        f"- 仅写数据能确认的信息，不空洞评价\n"
        f"- 含有中文指标名的 [METRICS]...[METRICS] 表放在第一部分末尾\n\n"
        f"## 分析数据\n"
        f"{combined[:12000]}"
    )

    from litellm import acompletion
    response_zh = await acompletion(
        model="openai/deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_API_BASE,
        messages=[{"role": "user", "content": zh_prompt}],
        temperature=0.2,
    )
    zh_report = response_zh.choices[0].message.content or ""
    print(f"[paperqa_skill] Chinese report generated ({len(zh_report)} chars)")

    # Step 3b: 翻译中文报告为英文 (忠实翻译，保留结构和数据)
    translate_prompt = (
        f"You are a professional academic translator. Translate the following Chinese technical "
        f"analysis report about the paper '{paper_title or pdf_stem}' into fluent, idiomatic English.\n\n"
        f"## Requirements\n"
        f"- Preserve ALL structure: sections (#), subsections (###), bullet points, paragraphs\n"
        f"- Keep bold numbers (**value**) exactly as they are — do not convert units\n"
        f"- Translate the [METRICS] table: convert Chinese metric names and descriptions to "
        f"standard English equivalents, keep the values unchanged. Preserve the 4-column format:\n"
        f"  value | metric name | explanation (one sentence describing what this metric measures and how this metric is defined) | category(accuracy/reduction/delta/other)\n"
        f"  Example: 39% | Accuracy Gain | Top-1 accuracy improvement over baseline | accuracy\n"
        f"- The explanation column must describe what the metric means — do NOT leave it empty or copy the metric name\n"
        f"- Use standard academic/technical English terminology\n"
        f"- Maintain technical accuracy — do not add, remove, or alter facts\n"
        f"- Output ONLY the translated report, no preamble or commentary\n\n"
        f"## Chinese Report to Translate\n\n"
        f"{zh_report[:18000]}"
    )

    response_en = await acompletion(
        model="openai/deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_API_BASE,
        messages=[{"role": "user", "content": translate_prompt}],
        temperature=0.1,  # 低温确保翻译忠实度
    )
    en_report = response_en.choices[0].message.content or ""
    print(f"[paperqa_skill] English report translated ({len(en_report)} chars)")

    # Step 3.5: 生成架构流程图 (中英各一份)
    flow_svg_zh = await _build_flow_svg(paper_title or pdf_stem, combined, lang="zh")
    flow_svg_en = await _build_flow_svg(paper_title or pdf_stem, combined, lang="en")

    # Step 3.6: 提取论文原图并用 Qwen3-VL-Plus 描述
    figures = _extract_figures(pdf_path, max_images=6)
    if figures:
        print(f"[paperqa_skill] Extracted {len(figures)} figures from PDF, describing with Qwen3-VL-Plus...")
    described_figures = await _describe_figures(figures, paper_title or pdf_stem, combined[:500])
    if described_figures:
        print(f"[paperqa_skill] Described {len(described_figures)} figures")
    else:
        print(f"[paperqa_skill] No figures extracted or described")

    # Step 4: 解析中文和英文报告 (各自独立拆分为2大部分)
    zh_parts = re.split(r'\n(?=# [^#])', zh_report.strip())
    zh_parts = [p.strip() for p in zh_parts if p.strip()]

    en_parts = re.split(r'\n(?=# [^#])', en_report.strip())
    en_parts = [p.strip() for p in en_parts if p.strip()]

    def _gen_one(path, lang_parts, flow, lang, report_text):
        _generate_html_report(
            html_path=path,
            title=paper_title or pdf_stem,
            summary=report_text,
            all_answers=all_answers,
            flow_svg=flow,
            figures=described_figures,
            parts=lang_parts,
            lang=lang,
        )
        print(f"[paperqa_skill] HTML report: {path}")

    zh_path = os.path.join(output_dir, f"{pdf_stem}_report_zh.html")
    en_path = os.path.join(output_dir, f"{pdf_stem}_report_en.html")
    _gen_one(zh_path, zh_parts, flow_svg_zh, "zh", zh_report)
    _gen_one(en_path, en_parts, flow_svg_en, "en", en_report)

    # Step 5: 生成纯文本版报告 (中英各一份)
    def _save_txt(path, lang_parts):
        text = "\n\n".join(lang_parts)
        # 清理 markdown: 去掉 ** 标记和 ### 前缀，保留可读文本
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'###\s+', '', text)
        text = re.sub(r'# [^\n]*\n', '', text)
        # 将 [METRICS] 块格式化为可读文本表格
        text = _format_metrics_for_txt(text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"[paperqa_skill] Text report: {path}")

    zh_txt = os.path.join(output_dir, f"{pdf_stem}_report_zh.txt")
    en_txt = os.path.join(output_dir, f"{pdf_stem}_report_en.txt")
    _save_txt(zh_txt, zh_parts)
    _save_txt(en_txt, en_parts)

    return zh_path, en_path


async def _build_flow_svg(title: str, combined_analysis: str, lang: str = "zh") -> str:
    """单独调用 LLM 生成架构流程描述, 并渲染为 SVG"""

    if lang == "en":
        flow_prompt = (
            f"Based on the paper '{title}', describe the system's end-to-end processing pipeline in one line.\n"
            f"Format: strictly use 'Step1 → Step2 → Step3 → ...' with English short names (8 chars max), "
            f"joined by ' → '. Example: 'User Query → Intent Detection → Planning → Execution → Response'.\n"
            f"Output ONLY this line, no extra text."
        )
    else:
        flow_prompt = (
            f"根据论文「{title}」的技术方案，用一句话描述系统从输入到输出的完整处理链路。\n"
            f"格式：必须严格用 '第一步 → 第二步 → 第三步 → ...' 的格式，每步用中文简短命名(5字内)，"
            f"用' → '连接。如：'用户查询 → 意图识别 → 任务规划 → 执行动作 → 结果返回'。\n"
            f"只输出这一行，不要任何额外文字。"
        )
    from litellm import acompletion
    try:
        resp = await acompletion(
            model="openai/deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            api_base=DEEPSEEK_API_BASE,
            messages=[{"role": "user", "content": flow_prompt}],
            temperature=0.1,  # 低温, 提高格式遵循率
        )
        flow_line = resp.choices[0].message.content or ""
    except Exception:
        return ""  # 流程图生成失败时静默跳过

    # 解析 "A → B → C" 格式
    import re
    steps = re.split(r'\s*→\s*', flow_line.strip())
    steps = [s.strip() for s in steps if s.strip() and len(s.strip()) <= 20]
    if len(steps) < 2:
        return ""

    # 渲染为水平 SVG 流程图
    n = len(steps)
    box_w, box_h = 90, 44
    gap = 36
    arrow_w = 24
    total_w = n * box_w + (n - 1) * (gap + arrow_w) + 40
    svg_h = 100

    boxes = ""
    arrows = ""
    for i, step in enumerate(steps):
        x = 20 + i * (box_w + gap + arrow_w)
        y = 28
        boxes += (
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" fill="#0f3460" opacity="0.92"/>\n'
            f'<text x="{x + box_w/2}" y="{y + box_h/2 + 5}" text-anchor="middle" fill="#fff" font-size="11" font-weight="bold">{step}</text>\n'
        )
        if i < n - 1:
            ax1 = x + box_w
            ax2 = ax1 + arrow_w
            ay = y + box_h // 2
            arrows += (
                f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="#e94560" stroke-width="2.5" marker-end="url(#arrowhead)"/>\n'
            )

    return f"""<div class="chart-card">
  <h3>🔀 系统架构流程</h3>
  <svg viewBox="0 0 {total_w} {svg_h}" width="100%" style="max-width:{total_w}px; display:block; margin:0 auto;">
    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="#e94560"/>
      </marker>
    </defs>
    {boxes}
    {arrows}
  </svg>
</div>"""


def _extract_figures(pdf_path: str, max_images: int = 6) -> list[dict]:
    """从 PDF 中提取图片 (PyMuPDF)，跳过过小的图标。

    Returns:
        [{"bytes": bytes, "ext": str, "page": int}, ...]
    """
    import fitz
    figures = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(min(len(doc), 24)):
            page = doc[page_num]
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue
                img_bytes = base_image["image"]
                if len(img_bytes) < 5000:  # 跳过小图标 / 装饰元素
                    continue
                figures.append({
                    "bytes": img_bytes,
                    "ext": base_image["ext"],
                    "page": page_num + 1,
                })
                if len(figures) >= max_images:
                    break
            if len(figures) >= max_images:
                break
        doc.close()
    except Exception as e:
        print(f"[paperqa_skill] Figure extraction failed: {e}")
    return figures


async def _describe_figures(
    figures: list[dict],
    paper_title: str,
    context_snippet: str = "",
) -> list[dict]:
    """用 Qwen3-VL-Plus 为每张图片生成中英双语描述。

    Returns:
        [{"b64": str, "ext": str, "desc_zh": str, "desc_en": str, "page": int}, ...]
    """
    import base64
    from litellm import acompletion

    described = []
    for fig in figures:
        b64 = base64.b64encode(fig["bytes"]).decode()
        ext = fig["ext"]
        mime = f"image/{ext}" if ext != "jpeg" else "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"

        prompt = (
            f"This figure is from the paper '{paper_title}'. "
            f"Brief context: {context_snippet[:300]}\n\n"
            f"Describe this figure in detail: what it shows, key components/architecture, "
            f"and any quantitative data visible. "
            f"Output format (STRICT):\n"
            f"【中文】... (2-3 sentences)\n"
            f"【English】... (2-3 sentences)"
        )
        try:
            response = await acompletion(
                model="openai/qwen3-vl-plus",
                api_key=BAILIAN_API_KEY,
                api_base=BAILIAN_API_BASE,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                temperature=0.2,
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[paperqa_skill] Figure description failed: {e}")
            text = ""

        # 解析中英
        desc_zh = desc_en = ""
        zh_m = re.search(r'【中文】\s*(.*?)(?=【English】|\Z)', text, re.DOTALL)
        en_m = re.search(r'【English】\s*(.*?)\Z', text, re.DOTALL)
        if zh_m:
            desc_zh = zh_m.group(1).strip()
        if en_m:
            desc_en = en_m.group(1).strip()
        if not desc_en and not desc_zh:
            desc_en = text.strip()[:200]

        described.append({
            "b64": b64, "ext": ext,
            "desc_zh": desc_zh, "desc_en": desc_en,
            "page": fig["page"],
        })

    return described


def _format_metrics_for_txt(text: str) -> str:
    """将 [METRICS]...[/METRICS] 块格式化为对齐的纯文本表格。

    跳过示例行 (以 "示例:" / "Example:" 开头)，按列对齐输出。
    """
    import re

    def _render_table(match: re.Match) -> str:
        block = match.group(1).strip()
        rows = []
        for line in block.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("示例:", "Example:")):
                continue
            cols = [c.strip() for c in line.split("|", 3)]
            if len(cols) >= 2:
                rows.append(cols)
        if not rows:
            return ""

        # 计算各列最大宽度
        n_cols = max(len(r) for r in rows)
        col_widths = [0] * n_cols
        for r in rows:
            for i, c in enumerate(r):
                # 中文字符宽度按2计算，ASCII按1
                width = sum(2 if ord(ch) > 127 else 1 for ch in c)
                col_widths[i] = max(col_widths[i], width, len(c))

        # 渲染表头 (根据指标名列判断语言)
        sep = "═" * (sum(col_widths) + (n_cols - 1) * 3 + 4)
        # 检测指标名称列 (index 1) 是否含中文
        label_col = rows[0][1] if len(rows[0]) > 1 else rows[0][0]
        header = "关键指标" if any(ord(c) > 127 for c in label_col) else "Key Metrics"
        out = [f"\n{sep}", f"  {header}", f"{sep}"]

        for r in rows:
            padded = []
            for i in range(n_cols):
                val = r[i] if i < len(r) else ""
                # 填充到显示宽度
                display_w = sum(2 if ord(ch) > 127 else 1 for ch in val)
                pad = max(0, col_widths[i] - display_w)
                padded.append(val + " " * pad)
            out.append("  " + " | ".join(padded))
        out.append(sep)
        return "\n".join(out)

    return re.sub(
        r'\[METRICS\]\s*\n(.*?)\[/METRICS\]',
        _render_table,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _generate_html_report(
    html_path: str,
    title: str,
    summary: str,
    all_answers: list[str],
    flow_svg: str = "",
    figures: list[dict] = None,
    parts: list[str] = None,
    lang: str = "zh",
) -> None:
    from datetime import datetime
    import re

    if figures is None:
        figures = []
    if parts is None:
        summary_clean = re.sub(r'\n?\[METRICS\].*?\[/METRICS\]', '', summary, flags=re.DOTALL | re.IGNORECASE)
        parts = [p.strip() for p in re.split(r'\n(?=# [^#])', summary_clean.strip()) if p.strip()]

    def _sanitize(text: str) -> str:
        """消除残留的 markdown 语法，确保在 HTML 中正确显示。"""
        # 先保留有用的转换
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', text)
        text = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', text)
        # 清除全部未转换的 ** 和孤立 * 和 ``
        text = text.replace('**', '')
        text = text.replace('*', '')   # 残余单星号
        text = text.replace('`', '')   # 残余反引号
        # 清除行首 # (残留标题标记)
        text = re.sub(r'(?<!\w)#{1,4}\s', '', text)
        # 转换 --- 为分隔线
        text = re.sub(r'(?<=\n)---(?=\n)', '<hr>', text)
        return text

    # ---- 解析 summary 为 2 大部分 (按 # 开头, 非 ## 非 ### 拆分) ----
    # (parts 已由调用方传入，跳过解析)

    part_colors = [
        "#0f3460",  # Overview (深蓝)
        "#e94560",  # Innovation (红)
    ]

    def _render_markdown_line(line: str) -> str:
        """将单行简易 md 转为 HTML，并消除残留 markdown"""
        line = _sanitize(line)
        return line

    def _render_subsection_card(part_title: str, accent: str, subsections: list[str],
                                flow: str = "", figures_html: str = "") -> str:
        """渲染一个大部分的 HTML: 标题行 + 子章节卡片组"""
        part_label = part_title.strip()
        is_overview = "总览" in part_label or "Overview" in part_label
        icon = "📋" if is_overview else "💡"
        # 语言标签 (由外层 lang 参数决定)
        if lang == "zh":
            lang_tag = '<span style="font-size:0.65em;background:#0f3460;color:#fff;padding:2px 10px;border-radius:10px;margin-left:8px;">中文</span>'
        else:
            lang_tag = '<span style="font-size:0.65em;background:#e94560;color:#fff;padding:2px 10px;border-radius:10px;margin-left:8px;">English</span>'

        cards = ""
        for sub in subsections:
            if not sub.strip():
                continue
            lines = sub.strip().split("\n")
            if not lines:
                continue
            sec_title = lines[0].replace("###", "").strip()
            sec_title = _sanitize(sec_title)
            # 标题过长时截断，避免正文内容混入标题
            if len(sec_title) > 80:
                sec_title = sec_title[:80] + "…"
            # 跳过架构流程小节 (已用 SVG 替代)
            if "架构流程" in sec_title or "Architecture Flow" in sec_title:
                if flow:
                    cards += flow
                continue
            body_parts = []
            for line in lines[1:]:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("- ") or s.startswith("* "):
                    body_parts.append(f"<li>{_render_markdown_line(s[2:])}</li>")
                elif s.startswith(("1. ", "2. ", "3. ", "4. ", "5. ")):
                    body_parts.append(f"<li>{_render_markdown_line(re.sub(r'^\d+\.\s', '', s))}</li>")
                elif s.startswith("---"):
                    continue
                else:
                    body_parts.append(f"<p>{_render_markdown_line(s)}</p>")
            body_html = "\n".join(body_parts)
            if "<li>" in body_html:
                body_html = f"<ul>{body_html}</ul>"
            cards += f"""<div class="section-card">
  <div class="section-header" style="border-left: 3px solid {accent};">
    <span class="section-icon">{icon}</span>
    <h3 style="color:{accent};">{sec_title}</h3>
  </div>
  <div class="section-body">{body_html}</div>
</div>\n"""

        return f"""<div class="part-wrap" style="margin-bottom: 32px;">
  <div class="part-title" style="border-left: 4px solid {accent}; padding: 8px 20px; font-size: 1.35em; font-weight: 800; color: {accent}; margin-bottom: 16px;">
    {part_label}{lang_tag}
  </div>
  {cards}
  {figures_html}
</div>"""

    # 生成论文原图 HTML
    figures_html_parts = ["", ""]  # overview figures, innovation figures
    if figures:
        for fig in figures[:6]:
            mime = f"image/{fig['ext']}" if fig['ext'] != 'jpeg' else 'image/jpeg'
            # 根据语言选择描述: 中文报告用 desc_zh, 英文报告用 desc_en
            if lang == "zh":
                desc = fig.get('desc_zh', '') or fig.get('desc_en', '')
            else:
                desc = fig.get('desc_en', '') or fig.get('desc_zh', '')
            desc = _sanitize(desc)
            fig_html = f"""<div class="figure-card">
  <img src="data:{mime};base64,{fig['b64']}" alt="Figure from page {fig['page']}" style="max-width:100%; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);"/>
  <p class="fig-desc"><strong>Figure (p.{fig['page']}):</strong> {desc[:300]}</p>
</div>"""
            # 前一半图片放总览，后一半放创新点
            idx = 0 if figures_html_parts[0].count('figure-card') < 3 else 1
            figures_html_parts[idx] += fig_html

    # 遍历各部分: 每部分拆出 ### 子章节
    parts_html = ""
    for idx, part_text in enumerate(parts):
        clean = part_text.strip()
        if not clean:
            continue
        part_lines = clean.split("\n", 1)
        part_title_line = part_lines[0].replace("#", "").strip()
        part_body = part_lines[1] if len(part_lines) > 1 else ""
        subsections = re.split(r'\n(?=###\s)', part_body.strip())
        accent = part_colors[idx % len(part_colors)]

        # 判断是否为 overview 部分 (包含"总览"或"Overview")
        is_overview = "总览" in part_title_line or "Overview" in part_title_line
        # 判断是否包含架构流程 (需要插入 flow SVG)
        has_architecture = "架构流程" in part_body or "Architecture Flow" in part_body
        # 选择对应的 figures
        fig_idx = 0 if is_overview else 1
        figs_html = figures_html_parts[fig_idx] if figures_html_parts[fig_idx] else ""
        # flow SVG 只插在有架构流程的部分
        flow_html = flow_svg if has_architecture else ""

        parts_html += _render_subsection_card(
            part_title_line, accent, subsections,
            flow=flow_html, figures_html=figs_html,
        )

    # ---- 从 parts 的 [METRICS] 块提取该语言的指标 ----
    metrics = []
    parts_text = "\n\n".join(parts) if parts else summary  # 从本语言 parts 中提取
    mm = re.search(r'\[METRICS\]\s*\n(.*?)\[/METRICS\]', parts_text, re.DOTALL | re.IGNORECASE)
    if not mm:
        # 回退: 从 full summary 中提取
        mm = re.search(r'\[METRICS\]\s*\n(.*?)\[/METRICS\]', summary, re.DOTALL | re.IGNORECASE)
    if mm:
        for line in mm.group(1).strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 跳过示例行 (以 "示例:" 或 "Example:" 开头)
            if line.startswith(("示例:", "Example:")):
                continue
            cols = [p.strip() for p in line.split("|", 3)]
            if len(cols) >= 3:
                value_str = cols[0]
                label_str = cols[1]
                description_str = cols[2] if len(cols) > 2 else ""
                category = cols[3] if len(cols) > 3 else "other"
                # 清理数字
                clean_val = value_str.replace('%', '').replace('×', '').replace('x', '')
                try:
                    num_val = float(clean_val)
                except ValueError:
                    continue
                metrics.append({
                    "label": label_str[:24],
                    "description": description_str[:80],
                    "value": value_str,
                    "num": num_val,
                    "category": category.strip(),
                })

    # 生成指标卡片 (所有指标)
    stats_html = ""
    if metrics:
        cards = ""
        for m in metrics[:6]:
            bg_color = "#0f3460" if m.get("category") in ("accuracy",) else "#e94560"
            desc_html = f'<div class="desc">{m["description"][:60]}</div>' if m.get("description") else ""
            cards += f"""<div class="stat-card" style="border-top-color: {bg_color};">
  <div class="num">{m["value"]}</div>
  <div class="label">{m["label"][:18]}</div>
  {desc_html}
</div>"""
        if cards:
            stats_html = f'<div class="stats-row">{cards}</div>'

    # 生成 SVG 柱状图: 按类别分两组
    bar_charts_html = ""
    for cat, cat_title in [("accuracy", "📊 准确率指标"), ("reduction", "📉 提升/降低指标")]:
        cat_metrics = [m for m in metrics if m.get("category") in (cat, "delta") and m["num"] > 0][:6]
        if len(cat_metrics) < 2:
            continue
        bars = ""
        bar_height = 28
        chart_height = len(cat_metrics) * (bar_height + 16) + 40
        max_val = max(m["num"] for m in cat_metrics)
        for i, m in enumerate(cat_metrics):
            y = 40 + i * (bar_height + 16)
            width_pct = (m["num"] / max_val) * 100 if max_val > 0 else 0
            bars += f"""<text x="0" y="{y + 18}" font-size="12" fill="#1a1a2e">{m["label"][:16]}</text>
  <rect x="130" y="{y}" width="{max(width_pct * 3.2, 4)}" height="{bar_height}" rx="4" fill="url(#barGrad)" opacity="0.85"/>
  <text x="{130 + max(width_pct * 3.2 + 6, 8)}" y="{y + 18}" font-size="12" font-weight="bold" fill="#e94560">{m["value"]}</text>"""
        bar_charts_html += f"""<div class="chart-card">
  <h3>{cat_title}</h3>
  <svg viewBox="0 0 480 {chart_height}" width="100%" style="max-width:480px">
    <defs>
      <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#e94560"/>
        <stop offset="100%" stop-color="#0f3460"/>
      </linearGradient>
    </defs>
    {bars}
  </svg>
</div>"""
    # 如果没有按类别分，回退到单图
    if not bar_charts_html:
        bar_charts_html = ""
        all_metrics = [m for m in metrics if m["num"] > 0][:6]
        if len(all_metrics) >= 2:
            bars = ""
            bar_height = 28
            chart_height = len(all_metrics) * (bar_height + 16) + 40
            max_val = max(m["num"] for m in all_metrics)
            for i, m in enumerate(all_metrics):
                y = 40 + i * (bar_height + 16)
                width_pct = (m["num"] / max_val) * 100 if max_val > 0 else 0
                bars += f"""<text x="0" y="{y + 18}" font-size="12" fill="#1a1a2e">{m["label"][:16]}</text>
  <rect x="130" y="{y}" width="{max(width_pct * 3.2, 4)}" height="{bar_height}" rx="4" fill="url(#barGrad)" opacity="0.85"/>
  <text x="{130 + max(width_pct * 3.2 + 6, 8)}" y="{y + 18}" font-size="12" font-weight="bold" fill="#e94560">{m["value"]}</text>"""
            bar_charts_html = f"""<div class="chart-card">
  <h3>📊 关键指标</h3>
  <svg viewBox="0 0 480 {chart_height}" width="100%" style="max-width:480px">
    <defs>
      <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#e94560"/>
        <stop offset="100%" stop-color="#0f3460"/>
      </linearGradient>
    </defs>
    {bars}
  </svg>
</div>"""

    # 生成 date
    date_str = datetime.now().strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — PaperQA2 分析报告</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", system-ui, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    color: #1a1a2e; line-height: 1.8; min-height: 100vh;
  }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 0 24px; }}

  /* header */
  header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #fff; padding: 64px 0 48px; text-align: center;
    position: relative; overflow: hidden;
  }}
  header::before {{
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(233,69,96,0.08) 0%, transparent 50%);
  }}
  header h1 {{ font-size: 2em; font-weight: 800; letter-spacing: -0.02em; position: relative; }}
  header .meta {{ margin-top: 16px; font-size: 0.85em; opacity: 0.65; position: relative; }}
  header .tags {{ margin-top: 20px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; position: relative; }}
  header .tag {{
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    padding: 4px 14px; border-radius: 20px; font-size: 0.78em; color: rgba(255,255,255,0.85);
  }}

  /* 主内容 */
  main {{ padding: 40px 0 60px; }}

  /* 统计卡片行 */
  .stats-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 40px;
  }}
  .stat-card {{
    background: #fff; border-radius: 14px; padding: 24px 20px; text-align: center;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06); border-top: 3px solid #e94560;
  }}
  .stat-card .num {{ font-size: 1.8em; font-weight: 800; color: #0f3460; line-height: 1.2; }}
  .stat-card .label {{ font-size: 0.82em; color: #666; margin-top: 6px; }}
  .stat-card .desc {{ font-size: 0.72em; color: #999; margin-top: 4px; line-height: 1.35; padding: 0 2px; }}

  /* 章节卡片 */
  .section-card {{
    background: #fff; border-radius: 14px; margin-bottom: 24px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06); overflow: hidden;
  }}
  .section-header {{
    display: flex; align-items: center; gap: 10px;
    padding: 18px 28px; background: #f8f9fc;
    border-bottom: 1px solid #eef0f4;
  }}
  .section-icon {{ font-size: 1.2em; }}
  .section-header h3 {{
    font-size: 1.1em; font-weight: 700; color: #1a1a2e; margin: 0;
  }}
  .section-body {{ padding: 24px 28px; }}
  .section-body p {{ margin-bottom: 12px; color: #2c2c3e; }}
  .section-body ul {{ list-style: none; padding: 0; }}
  .section-body li {{
    padding: 8px 0 8px 24px; position: relative; color: #2c2c3e;
    border-bottom: 1px solid #f0f0f4;
  }}
  .section-body li:last-child {{ border-bottom: none; }}
  .section-body li::before {{
    content: '▸'; position: absolute; left: 4px; color: #e94560; font-weight: bold;
  }}
  .section-body strong {{ color: #e94560; }}
  .section-body code {{
    background: #f0f2f5; padding: 2px 8px; border-radius: 4px;
    font-size: 0.9em; color: #0f3460;
  }}

  /* SVG 图表区域 */
  .chart-card {{
    background: #fff; border-radius: 14px; padding: 28px; margin-bottom: 24px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
  }}
  .chart-card h3 {{ font-size: 1em; font-weight: 700; color: #1a1a2e; margin-bottom: 20px; }}

  /* 论文原图卡片 */
  .figure-card {{
    background: #fff; border-radius: 12px; padding: 20px; margin: 16px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05); text-align: center;
  }}
  .figure-card img {{ max-width: 100%; max-height: 450px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .fig-desc {{
    text-align: left; margin-top: 12px; font-size: 0.88em; color: #555;
    background: #f8f9fc; padding: 12px 16px; border-radius: 8px;
    border-left: 3px solid #e94560;
  }}

  /* footer */
  footer {{
    text-align: center; padding: 32px; color: #888; font-size: 0.82em;
    border-top: 1px solid #e0e4ea;
  }}

  @media (max-width: 640px) {{
    header h1 {{ font-size: 1.4em; }}
    .stats-row {{ grid-template-columns: repeat(2, 1fr); }}
    .section-body {{ padding: 16px 18px; }}
  }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>{title}</h1>
    <div class="meta">PaperQA2 自动分析 · {date_str}</div>
    <div class="tags">
      <span class="tag">DeepSeek 文本分析</span>
      <span class="tag">Qwen3-VL-Plus 多模态</span>
      <span class="tag">st-all-mpnet-base-v2 嵌入</span>
    </div>
  </div>
</header>
<main class="container">
  {stats_html}
  {bar_charts_html}
  {parts_html}
</main>
<footer>
  <p>由 PaperQA2 自动分析 · DeepSeek + Qwen3-VL-Plus 双引擎驱动</p>
</footer>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# 配置管理命令
# ---------------------------------------------------------------------------
def configure(config_dir: Optional[str] = None) -> dict:
    """交互式配置向导: 逐项提示用户输入 LLM provider 信息并保存。

    可以在任何时间运行，已有值会作为默认值显示。
    留空按回车 = 保持原值。

    Args:
        config_dir: 配置文件目录 (默认 ~/.config/paperqa_skill/)

    Returns:
        完整的配置字典
    """
    existing = load_config(config_dir)
    cfg_dir = config_dir or _default_config_dir()
    cfg_path = get_config_path(cfg_dir)

    print("=" * 60)
    print("  PaperQA2 Skill 配置向导")
    print(f"  配置文件: {cfg_path}")
    print("=" * 60)

    for key, meta in _CONFIG_SCHEMA.items():
        current = existing.get(key, "") or meta["default"]
        env_val = os.environ.get(meta["env"])
        if env_val:
            hint = f" (已从环境变量 {meta['env']} 读取)"
            existing[key] = env_val
        else:
            hint = ""

        val = input(f"\n{meta['prompt']}{hint}\n  当前值: {current}\n> ").strip()
        if val:
            existing[key] = val
        # 如果环境变量有值，无论用户输入什么都保留环境变量
        if env_val:
            existing[key] = env_val

    path = save_config(existing, config_dir)
    print(f"\n✅ 配置已保存: {path}")

    # 显示最终配置摘要 (隐藏 key)
    print("\n当前配置摘要:")
    for k in _CONFIG_SCHEMA:
        v = existing.get(k, "")
        if "key" in k and v:
            v = v[:6] + "****" + v[-4:] if len(v) > 12 else "****"
        print(f"  {k}: {v}")

    return existing


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="PaperQA2 论文分析流水线 — 支持 PDF URL 或本地路径"
    )
    parser.add_argument("source", nargs="?", default=None,
                        help="PDF URL 或本地文件路径 (--configure 时可选)")
    parser.add_argument("--title", "-t", default="", help="论文标题")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--no-multimodal", action="store_true", help="禁用多模态")
    parser.add_argument("--full", "-f", action="store_true", help="运行完整流水线 (含 HTML 报告)")
    parser.add_argument("--configure", "-c", action="store_true",
                        help="运行配置向导 (设置 LLM API keys)")
    parser.add_argument("--config-dir", default=None,
                        help="配置文件目录 (默认 ~/.config/paperqa_skill/)")
    args = parser.parse_args()

    # 配置模式
    if args.configure:
        configure(args.config_dir)
        return 0

    # 分析模式: source 必填
    if not args.source:
        parser.print_help()
        print("\nError: 请提供 PDF URL 或本地路径 (或使用 --configure 进行配置)")
        return 1

    _cfg = load_config(args.config_dir)
    missing = check_config(_cfg)
    if missing:
        print("⚠️  以下配置项缺失，请先运行配置向导:")
        for m in missing:
            print(f"   - {m}")
        print(f"\n  运行: python -m paperqa_skill --configure")
        return 1

    if args.full:
        zh, en = asyncio.run(full_pipeline(
            pdf_source=args.source,
            paper_title=args.title,
            output_dir=args.output,
            multimodal=not args.no_multimodal,
        ))
        print(f"Done. ZH: {zh}\n     EN: {en}")
    else:
        queries = [
            f"Provide a comprehensive overview of the paper '{args.title or Path(args.source).stem}'. "
            "Describe key problem, approach, methodology, algorithms, experiments, and results."
        ]
        results = asyncio.run(analyze_paper(
            pdf_source=args.source,
            queries=queries,
            output_dir=args.output,
            multimodal=not args.no_multimodal,
        ))
        for k, v in results.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    exit(cli())
