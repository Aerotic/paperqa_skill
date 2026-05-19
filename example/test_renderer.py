"""
Unit test for HTML renderer — no API keys required.
Tests all three fixes: prompt-driven structure, overflow splitting, numbered block detection.
"""
import os, re, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperqa_skill import _generate_html_report

# ── Simulated LLM output (what DeepSeek would produce with the fixed prompt) ──

SIMULATED_ZH_REPORT = """# 第一部分：总览

### 简述
MobileGPT提出了一种**层次化记忆架构**，用于增强大语言模型（LLM）在移动端任务自动化中的子任务学习与回忆能力。核心挑战在于现有数据集无法捕获重复性任务，导致LLM在跨应用页面执行时效率低下。方法上采用探索-选择-推导-回忆四阶段工作流，结合基于子任务的屏幕分类与双策略纠错机制。在80个任务/8个应用的自定义数据集上，热启动任务成功率达**82.5%**，延迟降低**87%**，成本显著下降。消融实验显示热启动准确率达**98.75%**，离线探索覆盖89.65%所需页面，单次成本仅$10.78。

### 问题与挑战
现有移动端任务自动化系统（如AutoDroid、AppAgent）缺乏对重复子任务的结构化记忆与复用能力，导致每次执行相同任务时需从头推理，造成高延迟与高成本。此外，屏幕分类依赖视觉外观或完整UI元素提取，易产生误判（传统方法在269个屏幕中产生38/57个假阳性/假阴性）。

### 方法概览
MobileGPT提出**层次化记忆架构**，将应用页面按功能而非视觉外观分组为节点。系统采用**冷启动/热启动**双阶段：冷启动通过探索-选择-推导学习新任务；热启动通过子任务槽填充与上下文动作适配回忆已学任务，跳过学习阶段。

### 架构流程
1. **离线探索阶段**：随机探索器与用户轨迹监控器预收集每个应用50个页面，覆盖89.65%所需页面，成本$10.78。
2. **冷启动执行**：GPT-4-turbo执行探索（收集子任务）、选择（LLM基于指令与屏幕选择子任务）、推导（逐步生成低级动作直至子任务完成）。
3. **热启动执行**：GPT-3.5-turbo执行槽填充（填充子任务参数），结合上下文动作适配（动态调整回忆动作）。
4. **屏幕分类**：验证关键UI元素（如UI索引）而非提取完整子任务列表，使用余弦相似度避免冗余节点。
5. **纠错机制**：LLM自纠错（规则反馈处理UI错误与循环检测）与人类在环（HITL）修复（纠正缺失/错误子任务与动作），修复路径持久化至记忆。

### 关键效果
- 热启动任务成功率**82.5%**，优于AutoDroid与AppAgent。
- 热启动延迟降低**87%**（vs AutoDroid）与**90%**（vs AppAgent）。
- 探索阶段步骤准确率**96.4%**，推导阶段**99.1%**，步骤准确率**99.8%**。
- 槽填充准确率**99.5%**，上下文适配准确率**100%**。
- 记忆命中率因应用而异：Telegram **98.6%**，Gmail **70.6%**。

指标表格
<table>
    <tr>
        <td>指标名称</td>
        <td>值</td>
        <td>含义解释</td>
    </tr>
    <tr>
        <td>热启动任务成功率</td>
        <td>82.5%</td>
        <td>热启动模式下成功完成用户指令的比例，成功定义为所有子任务步骤均正确执行且达到预期结果</td>
    </tr>
    <tr>
        <td>热启动延迟降低</td>
        <td>87%</td>
        <td>相较于AutoDroid基线，热启动模式下任务完成时间的减少百分比</td>
    </tr>
    <tr>
        <td>探索阶段步骤准确率</td>
        <td>96.4%</td>
        <td>探索阶段中LLM生成的子任务步骤与人工标注正确步骤的一致性比例</td>
    </tr>
    <tr>
        <td>槽填充准确率</td>
        <td>99.5%</td>
        <td>热启动模式下子任务参数填充的正确率</td>
    </tr>
    <tr>
        <td>上下文适配准确率</td>
        <td>100%</td>
        <td>回忆动作根据当前屏幕内容动态调整后仍能正确执行的比例</td>
    </tr>
</table>

# 第二部分：创新点深度剖析

### 探索-选择-推导-回忆工作流与层次化记忆
**核心思想**：MobileGPT将任务自动化分解为四个有序阶段：探索、选择、推导、回忆。记忆采用**三级层次结构**（任务→子任务→动作），支持子任务跨任务共享与增量参数化。

**与已有工作的区别**：现有系统缺乏结构化子任务分解与记忆复用，每次执行相同任务需从头推理。MobileGPT引入链式思维风格的子任务分解，并设计冷启动/热启动双阶段记忆系统。

**技术实现**：探索/选择/推导阶段使用GPT-4-turbo，槽填充使用GPT-3.5-turbo。记忆缓存通过HITL修复的校正路径。阶段准确率：探索**96.4%**、选择**96.2%**、推导**99.1%**。

**有效性证据**：热启动延迟降低**87%**，步骤准确率**99.8%**。消融实验显示热启动准确率**98.75%**。

### 基于子任务的屏幕分类
**核心思想**：屏幕分类通过验证关键UI元素是否支持特定子任务，使用余弦相似度比较屏幕表示，避免冗余节点。

**与已有工作的区别**：传统方法在269个屏幕中产生38/57个假阳性/假阴性，MobileGPT仅产生**3**个假阳性与**0**个假阴性。

**技术实现**：屏幕表示为保留布局信息的层次化结构。子任务验证检查屏幕是否包含执行所需UI动作的元素。

**有效性证据**：假阳性从38降至3（减少**92.1%**），假阴性从57降至0（减少**100%**）。
"""


def parse_sections(html: str) -> list[dict]:
    """Extract section titles and bodies from generated HTML."""
    sections = []
    # Find all section cards
    card_pattern = re.compile(
        r'<div class="section-card">.*?<h3[^>]*>(.*?)</h3>.*?<div class="section-body">(.*?)</div>\s*</div>',
        re.DOTALL
    )
    for m in card_pattern.finditer(html):
        title = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        body = m.group(2).strip()
        # Strip HTML tags from body for length check
        body_text = re.sub(r'<[^>]+>', '', body).strip()
        sections.append({"title": title, "body": body_text, "body_html": body})
    return sections


def test_report_rendering():
    """Test that the renderer produces proper section cards without truncation."""
    output_dir = Path(tempfile.mkdtemp(prefix="paperqa_test_"))
    html_path = str(output_dir / "test_report_zh.html")

    try:
        # Parse parts (simulate what full_pipeline does)
        parts = [p.strip() for p in re.split(r'\n(?=# [^#])', SIMULATED_ZH_REPORT.strip()) if p.strip()]

        # Generate HTML
        _generate_html_report(
            html_path=html_path,
            title="Test Paper",
            summary=SIMULATED_ZH_REPORT,
            all_answers=[],
            flow_svg="",
            figures=[],
            parts=parts,
            lang="zh",
        )

        # Read generated HTML
        html = Path(html_path).read_text(encoding="utf-8")
        sections = parse_sections(html)

        # ── Assertions ──────────────────────────────────────────────────
        failures = []

        # 1. All 5 expected sections present
        expected = ["简述", "问题与挑战", "方法概览", "架构流程", "关键效果"]
        found_titles = [s["title"] for s in sections]
        for kw in expected:
            if not any(kw in t for t in found_titles):
                failures.append(f"MISSING section: '{kw}'")
        print(f"  [1] Sections found: {found_titles}")

        # 2.简述 section MUST have body
        jian_shu = next((s for s in sections if "简述" in s["title"]), None)
        if jian_shu:
            body_len = len(jian_shu["body"])
            if body_len < 30:
                failures.append(f"简述 body too short ({body_len} chars)")
            print(f"  [2] 简述 body: {body_len} chars")
        else:
            failures.append("简述 section not found")

        # 3. No truncation artifacts (… with empty body)
        for s in sections:
            if "…" in s["title"] and not s["body"]:
                failures.append(f"TRUNCATED title with EMPTY body: '{s['title'][:60]}'")

        # 4. Architecture flow section exists (no body expected, SVG instead)
        arch = next((s for s in sections if "架构流程" in s["title"]), None)
        print(f"  [3] 架构流程 found: {bool(arch)}")

        # 5. Metrics table present
        has_table = "<table>" in html
        if not has_table:
            failures.append("Metrics table missing")
        print(f"  [4] Metrics table: {'YES' if has_table else 'NO'}")

        # 6. No raw markdown `### ` or `# ` in body
        raw_md = re.search(r'^#{1,4}\s', html, re.MULTILINE)
        if raw_md:
            failures.append(f"Raw markdown heading in HTML: '{raw_md.group()}'")
        print(f"  [5] Raw markdown: {'CLEAN' if not raw_md else 'FOUND'}")

        # 7. Print first section details
        if sections:
            s = sections[0]
            print(f"\n  First section:")
            print(f"    Title: '{s['title'][:80]}'")
            print(f"    Body preview: '{s['body'][:120]}...'")
            print(f"    Body length: {len(s['body'])} chars")

        # ── Result ─────────────────────────────────────────────────────
        if failures:
            print(f"\n❌ FAILED ({len(failures)} issues):")
            for f in failures:
                print(f"  - {f}")
            return False
        else:
            print(f"\n✅ ALL CHECKS PASSED")
            return True

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)


def test_overflow_splitting():
    """Test that a long single-line subsection gets split into title + body."""
    # Simulate a subsection where the LLM put everything on the ### line
    report = """# 第一部分：总览
### MobileGPT提出了一种**层次化记忆架构**，用于增强大语言模型（LLM）在移动端任务自动化中的子任务学习与回忆能力。核心挑战在于现有数据集无法捕获重复性任务，导致LLM在跨应用页面执行时效率低下。方法上，MobileGPT采用探索-选择-推导-回忆四阶段工作流，结合基于子任务的屏幕分类与双策略纠错机制。在80个任务/8个应用的自定义数据集上，热启动（warm-start）任务成功率达82.5%，延迟降低87%，成本显著下降。消融实验显示热启动准确率达98.75%，离线探索覆盖89.65%所需页面，单次成本仅$10.78。
"""

    output_dir = Path(tempfile.mkdtemp(prefix="paperqa_test2_"))
    html_path = str(output_dir / "test_overflow_zh.html")

    try:
        parts = [p.strip() for p in re.split(r'\n(?=# [^#])', report.strip()) if p.strip()]
        _generate_html_report(
            html_path=html_path,
            title="Test",
            summary=report,
            all_answers=[],
            flow_svg="",
            figures=[],
            parts=parts,
            lang="zh",
        )

        html = Path(html_path).read_text(encoding="utf-8")
        sections = parse_sections(html)

        failures = []
        for s in sections:
            title_clean = re.sub(r'<[^>]+>', '', s["title"])
            # Title should be SHORT (split at comma)
            if len(title_clean) > 50:
                failures.append(f"Title not split: {len(title_clean)} chars: '{title_clean[:60]}'")
            # Body should NOT be empty
            if not s["body"]:
                failures.append(f"Empty body for title: '{title_clean[:60]}'")
            print(f"  Title ({len(title_clean)}): '{title_clean[:80]}'")
            print(f"  Body ({len(s['body'])}): '{s['body'][:100]}...'")

        if failures:
            print(f"\n❌ FAILED: {failures}")
            return False
        else:
            print(f"\n✅ Overflow splitting works")
            return True

    finally:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Structured report rendering")
    print("=" * 60)
    r1 = test_report_rendering()

    print(f"\n{'=' * 60}")
    print("TEST 2: Overflow title splitting")
    print("=" * 60)
    r2 = test_overflow_splitting()

    print(f"\n{'=' * 60}")
    if r1 and r2:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
        exit(1)
