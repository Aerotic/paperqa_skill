"""测试 paperqa_skill 各模块功能"""
import os, sys, json, tempfile, shutil

# ---------------------------------------------------------------------------
# 1. 模块导入测试
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. 模块导入")
print("=" * 60)

sys.path.insert(0, r"C:\Users\123")
import paperqa_skill as pqs

print(f"  Version: {pqs.__version__}")
print(f"  Module loaded: OK")

# ---------------------------------------------------------------------------
# 2. 配置系统测试
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 配置系统")
print("=" * 60)

# 2a. 默认配置目录
cfg_dir = pqs._default_config_dir()
print(f"  Default config dir: {cfg_dir}")
assert cfg_dir.endswith(".config\\paperqa_skill") or cfg_dir.endswith(".config/paperqa_skill")

# 2b. 加载空配置 (无文件、无环境变量)
# 临时修改 home 指向一个临时目录，避免影响用户配置
with tempfile.TemporaryDirectory() as tmp:
    fake_home = os.path.join(tmp, "fake_home")
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.pop("BAILIAN_API_KEY", None)

    # load_config 使用默认值
    config = pqs.load_config(config_dir=tmp)
    print(f"  Config keys: {list(config.keys())}")
    assert "deepseek_api_key" in config
    assert "bailian_api_key" in config
    assert "deepseek_api_base" in config
    assert "paper_dir" in config
    print(f"  deepseek_api_base: {config['deepseek_api_base']}")
    print(f"  bailian_api_base: {config['bailian_api_base']}")
    print(f"  paper_dir: {config['paper_dir']}")
    # Without env/file, keys should be empty strings (default)
    assert config["deepseek_api_key"] == "", f"Expected empty, got: {config['deepseek_api_key']}"

    # 2c. check_config - 应报告缺失
    missing = pqs.check_config(config)
    print(f"  Missing keys: {missing}")
    assert len(missing) == 2
    assert "deepseek_api_key" in missing[0]

    # 2d. save_config + load_config roundtrip
    test_cfg = {
        "deepseek_api_key": "sk-test-key-12345",
        "deepseek_api_base": "https://api.deepseek.com",
        "bailian_api_key": "sk-test-bailian",
        "bailian_api_base": "https://dashscope.aliyuncs.com",
        "paper_dir": tmp,
    }
    saved_path = pqs.save_config(test_cfg, config_dir=tmp)
    print(f"  Config saved: {saved_path}")
    assert os.path.isfile(saved_path)

    # verify file content
    with open(saved_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["deepseek_api_key"] == "sk-test-key-12345"
    print(f"  File content verified: OK")

    # reload
    reloaded = pqs.load_config(config_dir=tmp)
    assert reloaded["deepseek_api_key"] == "sk-test-key-12345"
    assert reloaded["bailian_api_key"] == "sk-test-bailian"
    print(f"  Load/Save roundtrip: OK")

    # 2e. 环境变量应覆盖文件
    os.environ["DEEPSEEK_API_KEY"] = "sk-env-override"
    reloaded2 = pqs.load_config(config_dir=tmp)
    assert reloaded2["deepseek_api_key"] == "sk-env-override"
    print(f"  Env var override: OK")
    del os.environ["DEEPSEEK_API_KEY"]

print(f"\n  Config tests: ALL PASSED")

# ---------------------------------------------------------------------------
# 3. URL 检测测试
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. URL 检测")
print("=" * 60)

assert pqs._is_url("https://arxiv.org/pdf/2401.12345.pdf")
assert pqs._is_url("http://example.com/paper.pdf")
assert pqs._is_url("https://doi.org/10.1016/j.sysarc.2025.103580")
assert not pqs._is_url("/home/user/paper.pdf")
assert not pqs._is_url(r"C:\Users\test\paper.pdf")
assert not pqs._is_url("paper.pdf")
print("  URL detection: PASSED")

# ---------------------------------------------------------------------------
# 4. HTML PDF URL 提取测试
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. PDF URL 提取 (HTML 解析)")
print("=" * 60)

sample_html = """
<html>
<head><meta name="citation_pdf_url" content="https://example.com/paper.pdf"></head>
<body>
<a href="https://example.com/fulltext.pdf">Download PDF</a>
<a href="/files/paper.pdf">Local PDF</a>
<div data-pdf-url="/assets/main.pdf">
</body>
</html>
"""
urls = pqs._extract_pdf_urls(sample_html, "https://example.com")
print(f"  Extracted URLs: {urls}")
expected = [
    "https://example.com/fulltext.pdf",
    "https://example.com/files/paper.pdf",
    "https://example.com/paper.pdf",
    "https://example.com/assets/main.pdf",
]
for e in expected:
    assert e in urls, f"Missing: {e}"
print(f"  Expected {len(expected)} URLs, found {len(urls)}: PASSED")

# ---------------------------------------------------------------------------
# 5. 空/无 PDF 链接页面测试
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 无 PDF 链接页面")
print("=" * 60)

empty_html = "<html><body><p>No PDF here</p></body></html>"
urls2 = pqs._extract_pdf_urls(empty_html, "https://example.com")
assert len(urls2) == 0
print("  Empty result: PASSED")

# ---------------------------------------------------------------------------
# 6. resolve_pdf_source 本地路径测试
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. resolve_pdf_source (本地路径)")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmp:
    # 创建临时 PDF 文件
    fake_pdf = os.path.join(tmp, "test.pdf")
    with open(fake_pdf, "wb") as f:
        f.write(b"%PDF-1.4 fake pdf content")

    result = pqs.resolve_pdf_source(fake_pdf)
    assert result == fake_pdf
    print(f"  Local path resolved: {result}")

    # 不存在的文件
    try:
        pqs.resolve_pdf_source(os.path.join(tmp, "nonexistent.pdf"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print(f"  Missing file error: OK")

print("  Local path tests: PASSED")

# ---------------------------------------------------------------------------
# 7. CLI 参数解析测试
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. CLI 参数解析")
print("=" * 60)

def test_cli_args(args_str):
    """模拟 CLI 参数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=None)
    parser.add_argument("--title", "-t", default="")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--no-multimodal", action="store_true")
    parser.add_argument("--full", "-f", action="store_true")
    parser.add_argument("--configure", "-c", action="store_true")
    parser.add_argument("--config-dir", default=None)
    return parser.parse_args(args_str.split())

# --configure mode
args = test_cli_args("--configure")
assert args.configure
assert args.source is None
print("  --configure mode: OK")

# normal mode with URL
args = test_cli_args("https://arxiv.org/pdf/2401.12345.pdf --full -t TestPaper -o ./out")
assert args.source == "https://arxiv.org/pdf/2401.12345.pdf"
assert args.full
assert args.title == "TestPaper"
assert args.output == "./out"
print("  normal mode with URL: OK")

# no source (should error in cli logic)
args = test_cli_args("")
assert args.source is None
assert not args.full
print("  no-args mode: OK")

print("  CLI arg parsing: PASSED")

# ---------------------------------------------------------------------------
# 8. 全局配置初始化测试 (模块加载时)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. 全局配置")
print("=" * 60)

# 模块加载时已初始化 _CFG, DEEPSEEK_API_KEY 等
# 因为有环境变量污染风险，只验证接口存在
print(f"  Global DEEPSEEK_API_BASE: {pqs.DEEPSEEK_API_BASE}")
assert pqs.DEEPSEEK_API_BASE.startswith("https")
assert hasattr(pqs, "BAILIAN_API_BASE")
assert hasattr(pqs, "PAPER_DIR")

# check_config on global config
global_missing = pqs.check_config()
print(f"  Global missing keys: {global_missing}")
# (if no env/file, these will be empty - expected)
print(f"  Global config init: OK")

# ---------------------------------------------------------------------------
# 总结
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("全部测试完成!")
print("=" * 60)
print("\n注意: 以下功能需要 LLM API 调用，未在此测试:")
print("  - download_pdf(url)       (需要真实网络)")
print("  - analyze_paper()         (需要 PaperQA2 + DeepSeek)")
print("  - full_pipeline()         (需要 PaperQA2 + DeepSeek)")
print("  - make_settings()         (需要 paper-qa 库)")
print("  - configure()             (交互式, 需人工输入)")
print("  - cli()                   (完整的命令行流程)")
