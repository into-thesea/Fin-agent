"""sync_kb 的纯函数自检: 指纹比对 / 版本号 / 事实源清单

这些是"源没变就不重建"的全部判断依据, 错了会导致该重建时跳过、或每次都重建。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import sync_kb


def test_diff_keys_detects_all_three_kinds():
    old = {"a.md": "111", "b.md": "222", "c.md": "333"}
    new = {"a.md": "111", "b.md": "999", "d.md": "444"}
    added, removed, modified = sync_kb.diff_keys(old, new)
    assert added == ["d.md"]
    assert removed == ["c.md"]
    assert modified == ["b.md"]


def test_diff_keys_identical_is_empty():
    fp = {"a.md": "111"}
    assert sync_kb.diff_keys(fp, dict(fp)) == ([], [], [])


def test_version_is_stable_and_sensitive():
    """同一内容必得同一版本号; 任一文件内容变了版本号必须变"""
    fp = {"a.md": "111", "b.md": "222"}
    assert sync_kb.compute_version(fp) == sync_kb.compute_version(dict(fp))
    assert sync_kb.compute_version(fp) != sync_kb.compute_version({**fp, "b.md": "999"})
    assert len(sync_kb.compute_version(fp)) == 12


def test_source_files_excludes_underscore_and_includes_catalog():
    files = [os.path.basename(f) for f in sync_kb.source_files()]
    assert files, "事实源清单不应为空"
    assert not [f for f in files if f.startswith("_")], "下划线开头的文件必须跳过(与 build 规则一致)"
    assert "catalog.jsonl" in files
    assert "kg_domain_triples.jsonl" in files
    assert "risk_level.md" in files


def test_generated_artifacts_are_not_sources():
    """产物绝不能进指纹 —— 否则生成物反馈回来会自我触发重建, 每次都"检测到变化"。

    kg_triples.jsonl 现在由 build_finance_kb 从 catalog + kg_domain_triples 生成,
    所以它必须不在事实源清单里 (手写的领域知识在 kg_domain_triples.jsonl)。
    """
    files = [os.path.basename(f) for f in sync_kb.source_files()]
    for artifact in ("kg_triples.jsonl", "chunks_processed.jsonl", "bm25_index.pkl"):
        assert artifact not in files, f"{artifact} 是产物, 不应作为事实源"


def test_builder_script_is_fingerprinted():
    """构建脚本决定派生结果怎么渲染, 改它必须触发重建 —— 漏了就会出现
    '改了模板但索引没更新' 的静默不一致。"""
    assert sync_kb.BUILDER in sync_kb.source_files()


def test_fingerprint_covers_every_source():
    fp = sync_kb.fingerprint()
    assert set(fp) == {os.path.relpath(f, sync_kb.ROOT).replace("\\", "/")
                       for f in sync_kb.source_files()}
    assert all(len(h) == 16 for h in fp.values())


def test_graph_failure_returns_nonzero_and_is_retried_next_run(monkeypatch):
    """灌库失败必须非零退出 + 不记 graph_version, 下次跑自动补灌

    回归点: 以前 reload_graph() 把失败吞成一句"跳过", main() 照样返回 0,
    源又没再变 → 图谱永久落后于源, 只能靠人肉对账发现 (实测落后 24 条边)。
    """
    fp = {"a.md": "111"}
    version = sync_kb.compute_version(fp)
    monkeypatch.setattr(sys, "argv", ["sync_kb.py"])
    monkeypatch.setattr(sync_kb, "fingerprint", lambda: dict(fp))
    monkeypatch.setattr(sync_kb, "load_manifest",
                        lambda: {"version": version, "sources": dict(fp)})
    saved = {}
    monkeypatch.setattr(sync_kb, "save_manifest", lambda m: saved.update(m))

    # 第一次: 库没起 → 非零退出, 且不许把这一版记成已收敛
    monkeypatch.setattr(sync_kb, "reload_graph", lambda: False)
    assert sync_kb.main() == 3
    assert "graph_version" not in saved

    # 第二次: 库起来了。manifest 还是旧的(没记 graph_version) → 只补灌, 不重跑重建
    monkeypatch.setattr(sync_kb, "reload_graph", lambda: True)
    assert sync_kb.main() == 0
    assert saved["graph_version"] == version
