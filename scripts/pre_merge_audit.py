#!/usr/bin/env python3
"""
合并前审计脚本 — 防止两个来源改动互相覆盖、文件意外丢失。

用法:
    python scripts/pre_merge_audit.py <源分支>          # 试合并 + 审计, 结束后自动中止
    python scripts/pre_merge_audit.py <源分支> --keep   # 审计通过后保留试合并状态(需手动 commit)
    python scripts/pre_merge_audit.py <源分支> --skip-tests   # 跳过 pytest(只做静态审计)

审计项:
    1. 工作区干净检查
    2. 试合并, 列出冲突文件
    3. 列出从分叉点到源分支被删除的文件(需人工确认是否有意)
    4. 合并后 vs 源分支 diff(应为空, 证明源分支改动全部进入)
    5. 测试收集数对比(合并前后应一致)
    6. 全量 pytest(除非 --skip-tests)
    7. 关键模块导入检查

退出码: 0 = 全部通过, 1 = 有 FAIL 项, 2 = 环境错误(无法执行)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 需要确认完整性的关键模块(相对路径)
CRITICAL_MODULES = [
    "src/agents/router_agent.py",
    "src/retrieval/retriever.py",
    "src/graph/cs_graph.py",
    "src/llm/query_rewriter.py",
    "src/knowledge_graph/retriever.py",
    "src/tools/registry.py",
    "src/core/kb_settings.py",
]

# 已知的、可接受的删除文件前缀/关键词(匹配到则标记为 INFO 而非 WARN)
KNOWN_DELETION_PATTERNS = [
    "faiss",          # FAISS 彻底出局
    "_archive",       # 归档目录
    "migrate_",       # 一次性迁移脚本
]


def run(cmd: list[str], check: bool = False, capture: bool = True) -> subprocess.CompletedProcess:
    """执行 git/系统命令"""
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=capture, text=True, check=check
    )


def git(*args: str) -> str:
    """执行 git 命令并返回 stdout"""
    r = run(["git", *args])
    if r.returncode != 0:
        return r.stderr.strip()
    return r.stdout.strip()


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def result(label: str, status: str, detail: str = "") -> None:
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "INFO": "ℹ️ "}.get(status, "  ")
    line = f"{icon} [{status}] {label}"
    if detail:
        line += f"\n       {detail}"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="合并前审计")
    parser.add_argument("source", help="要合并的源分支(如 origin/kb-governance)")
    parser.add_argument("--keep", action="store_true", help="审计通过后保留试合并状态")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 pytest")
    parser.add_argument("--python", default=None, help="Python 解释器路径(默认 sys.executable)")
    args = parser.parse_args()

    py = args.python or sys.executable
    failures: list[str] = []
    warnings: list[str] = []

    # ── 0. 当前状态 ──────────────────────────────────────────
    section("0. 当前状态")
    current_branch = git("branch", "--show-current")
    print(f"  当前分支: {current_branch}")
    print(f"  源分支:   {args.source}")

    # ── 1. 工作区干净检查 ────────────────────────────────────
    section("1. 工作区干净检查")
    dirty = git("status", "--porcelain")
    if dirty:
        files = dirty.splitlines()
        tracked = [f for f in files if not f.startswith("??")]
        if tracked:
            result("工作区有未提交改动", "FAIL",
                   f"{len(tracked)} 个已跟踪文件有改动, 请先 commit/stash:\n" +
                   "\n".join(f"       {f}" for f in tracked[:10]))
            failures.append("工作区不干净")
            return 1
        else:
            result("仅有未跟踪文件", "INFO", f"{len(files)} 个未跟踪文件, 不影响合并")
    else:
        result("工作区干净", "PASS")

    # ── 2. 试合并 ────────────────────────────────────────────
    section("2. 试合并 (--no-commit --no-ff)")
    merge_r = run(["git", "merge", args.source, "--no-commit", "--no-ff"])
    merge_out = (merge_r.stdout + merge_r.stderr).strip()

    if "CONFLICT" in merge_out:
        conflict_files = git("diff", "--name-only", "--diff-filter=U").splitlines()
        conflict_files = [f for f in conflict_files if f]
        result("存在合并冲突", "WARN",
               f"{len(conflict_files)} 个文件冲突, 需人工解决:\n" +
               "\n".join(f"       - {f}" for f in conflict_files))
        warnings.append(f"合并冲突: {', '.join(conflict_files)}")
    elif merge_r.returncode != 0 and "Already up to date" not in merge_out:
        result("合并执行失败", "FAIL", merge_out[:500])
        failures.append("合并执行失败")
        git("merge", "--abort")
        return 1
    else:
        result("无冲突自动合并", "PASS")

    # ── 3. 被删除文件清单 ────────────────────────────────────
    section("3. 从分叉点到源分支被删除的文件")
    base = git("merge-base", "HEAD", args.source)
    deleted_raw = git("diff", "--name-status", base, args.source)
    deleted = [line.split("\t", 1)[1] for line in deleted_raw.splitlines()
               if line.startswith("D") and "\t" in line]

    if not deleted:
        result("无文件被删除", "PASS")
    else:
        known = [f for f in deleted if any(p in f.lower() for p in KNOWN_DELETION_PATTERNS)]
        unknown = [f for f in deleted if f not in known]
        if known:
            result(f"{len(known)} 个已知可接受的删除", "INFO",
                   "\n".join(f"       - {f}" for f in known))
        if unknown:
            result(f"{len(unknown)} 个删除需人工确认", "WARN",
                   "\n".join(f"       - {f}" for f in unknown))
            warnings.append(f"未确认删除: {', '.join(unknown)}")
        else:
            result("所有删除均为已知可接受", "PASS")

    # ── 4. 合并后 vs 源分支 diff ─────────────────────────────
    section("4. 合并后 vs 源分支 (应为空, 证明源分支改动全部进入)")
    diff_files = git("diff", "--name-only", args.source, "HEAD").splitlines()
    diff_files = [f for f in diff_files if f]
    if not diff_files:
        result("合并后与源分支完全一致", "PASS", "源分支所有改动已完整进入当前分支")
    else:
        result("合并后与源分支存在差异", "WARN",
               f"{len(diff_files)} 个文件不同(可能是当前分支独有的改动, 需确认):\n" +
               "\n".join(f"       - {f}" for f in diff_files[:15]))
        warnings.append(f"合并后差异: {len(diff_files)} 个文件")

    # ── 5. 测试收集数对比 ────────────────────────────────────
    section("5. 测试收集数对比")
    def count_tests() -> int | None:
        r = run([py, "-m", "pytest", "tests/", "--collect-only", "-q"])
        out = r.stdout + r.stderr
        for line in out.splitlines():
            if "test" in line and "collected" in line:
                try:
                    return int(line.split()[0])
                except (ValueError, IndexError):
                    pass
        return None

    after = count_tests()
    # 中止合并后再测一次"合并前"
    git("merge", "--abort")
    before = count_tests()
    # 重新试合并(保持状态一致)
    run(["git", "merge", args.source, "--no-commit", "--no-ff"])

    if before is None or after is None:
        result("测试收集数无法获取", "WARN", "pytest --collect-only 输出解析失败")
    elif before == after:
        result(f"测试数一致: {after}", "PASS")
    else:
        result(f"测试数变化: {before} → {after}", "WARN",
               f"差异 {after - before:+d}, 需确认是否有意(新增/删除测试)")
        warnings.append(f"测试数变化: {before}→{after}")

    # ── 6. 全量 pytest ───────────────────────────────────────
    if not args.skip_tests:
        section("6. 全量 pytest")
        print("  运行中... (可能需要 1-3 分钟)")
        r = run([py, "-m", "pytest", "tests/", "-q"])
        out = (r.stdout + r.stderr).strip().splitlines()
        summary = next((l for l in reversed(out) if "passed" in l or "failed" in l), "")
        if r.returncode == 0:
            result("pytest 全部通过", "PASS", summary)
        else:
            failed = [l for l in out if l.startswith("FAILED")]
            result("pytest 有失败", "FAIL",
                   summary + "\n" + "\n".join(f"       {l}" for l in failed[:10]))
            failures.append(f"pytest 失败: {len(failed)} 个")
    else:
        section("6. 全量 pytest (已跳过)")
        result("跳过 pytest", "INFO", "使用了 --skip-tests")

    # ── 7. 关键模块导入 ──────────────────────────────────────
    section("7. 关键模块导入检查")
    missing = []
    for mod in CRITICAL_MODULES:
        if not (REPO_ROOT / mod).exists():
            missing.append(mod)
    if missing:
        result("关键模块缺失", "FAIL",
               "\n".join(f"       - {m}" for m in missing))
        failures.append(f"关键模块缺失: {', '.join(missing)}")
    else:
        # 尝试导入
        import_script = (
            "import sys; sys.path.insert(0, '.'); "
            + "; ".join(f"import {m.replace('/', '.').replace('.py', '')}"
                       for m in CRITICAL_MODULES if m.startswith("src/"))
        )
        r = run([py, "-c", import_script])
        if r.returncode == 0:
            result(f"全部 {len(CRITICAL_MODULES)} 个关键模块存在且可导入", "PASS")
        else:
            result("关键模块导入失败", "WARN",
                   (r.stdout + r.stderr).strip()[:300])
            warnings.append("关键模块导入失败")

    # ── 收尾 ─────────────────────────────────────────────────
    section("审计总结")
    print(f"  FAIL: {len(failures)} 项")
    print(f"  WARN: {len(warnings)} 项")
    if failures:
        print("\n  ❌ 存在 FAIL 项, 不建议合并:")
        for f in failures:
            print(f"     - {f}")
    if warnings:
        print("\n  ⚠️  以下 WARN 项需人工确认:")
        for w in warnings:
            print(f"     - {w}")

    # 处理试合并状态
    in_merge = Path(REPO_ROOT / ".git" / "MERGE_HEAD").exists()
    if in_merge:
        if args.keep and not failures:
            print("\n  ℹ️  已保留试合并状态, 请人工解决 WARN 项后 git commit")
        else:
            git("merge", "--abort")
            print("\n  ℹ️  已自动中止试合并, 工作区恢复原样")

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
