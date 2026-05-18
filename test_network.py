"""测试 configure() 交互配置 和 download_pdf() 网络下载"""
import os, sys, json, tempfile

sys.path.insert(0, r"C:\Users\123")
import paperqa_skill as pqs

print("=" * 60)
print("测试 1: configure() 交互式配置向导")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmp:
    # 模拟用户输入: 空行保持默认 + 自定义 key
    import io
    simulated_input = "\n".join([
        "",                    # deepseek_api_key → 输入空 = 保持默认值 (空字符串)
        "",                    # deepseek_api_base → 保持默认
        "sk-test-bailian-key", # bailian_api_key → 输入值
        "",                    # bailian_api_base → 保持默认
        tmp,                   # paper_dir → 输入临时目录
    ])
    sys.stdin = io.StringIO(simulated_input)

    try:
        config = pqs.configure(config_dir=tmp)
        print(f"\n  Result config:")
        print(f"    deepseek_api_key: {repr(config['deepseek_api_key'][:10])}...")
        print(f"    bailian_api_key: {config['bailian_api_key']}")
        print(f"    paper_dir: {config['paper_dir']}")

        # verify saved to file
        cfg_path = pqs.get_config_path(config_dir=tmp)
        assert os.path.isfile(cfg_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["bailian_api_key"] == "sk-test-bailian-key"
        assert saved["paper_dir"] == tmp
        print(f"  Config file saved and verified: PASSED")
    finally:
        sys.stdin = sys.__stdin__

print("\n" + "=" * 60)
print("测试 2: download_pdf() 网络下载 (arXiv 公开 PDF)")
print("=" * 60)

try:
    with tempfile.TemporaryDirectory() as tmp:
        # arXiv PDF - a small paper
        url = "https://arxiv.org/pdf/2401.00101.pdf"
        print(f"  Downloading from: {url}")
        pdf_path = pqs.download_pdf(url, output_dir=tmp, filename="test_arxiv")
        print(f"  Downloaded to: {pdf_path}")
        with open(pdf_path, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"%PDF", f"Not a PDF: {header}"
        size = os.path.getsize(pdf_path)
        print(f"  Valid PDF: {size} bytes, header={header[:8]}")
        print("  Download test: PASSED")
except Exception as e:
    print(f"  Download test FAILED (network issue?): {e}")
    print("  (This is expected if no internet or arXiv is blocked)")

print("\n" + "=" * 60)
print("测试 3: make_settings() 工厂函数 (需要 paper-qa)")
print("=" * 60)

try:
    settings = pqs.make_settings(
        paper_directory=tempfile.gettempdir(),
        multimodal=True,
    )
    print(f"  Settings created: OK")
    print(f"  llm: {settings.llm}")
    print(f"  embedding: {settings.embedding}")
    print(f"  multimodal: {settings.parsing.get('multimodal')}")
    assert "deepseek" in settings.llm.lower()
    assert settings.embedding == "st-all-mpnet-base-v2"
    print("  make_settings: PASSED")
except Exception as e:
    print(f"  make_settings FAILED: {e}")
    print("  (paper-qa library not installed?)")

print("\n" + "=" * 60)
print("测试 4: 命令行入口 (模拟 --configure)")
print("=" * 60)

try:
    with tempfile.TemporaryDirectory() as tmp:
        # 模拟 cli() with --configure
        sys.argv = ["paperqa_skill", "--configure", "--config-dir", tmp]
        simulated_input2 = "\n".join([
            "sk-test-ds-key",
            "",
            "sk-test-bl-key",
            "",
            tmp,
        ])
        sys.stdin = io.StringIO(simulated_input2)
        try:
            rc = pqs.cli()
            assert rc == 0
            print(f"  CLI --configure exit code: {rc}")
            # verify config was saved
            cfg_path = pqs.get_config_path(config_dir=tmp)
            assert os.path.isfile(cfg_path)
            # verify cli now passes check
            config = pqs.load_config(config_dir=tmp)
            missing = pqs.check_config(config)
            assert len(missing) == 0, f"Still missing: {missing}"
            print(f"  Config complete, no missing keys: PASSED")
        finally:
            sys.stdin = sys.__stdin__
finally:
    # reset argv
    sys.argv = ["paperqa_skill"]

print("\n" + "=" * 60)
print("全部测试完成!")
print("=" * 60)
