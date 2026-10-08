"""
重排器测试 (不发真实网络请求)

覆盖两点:
  1. ApiReranker 的解析/排序/越界与失败处理逻辑
  2. 重排**默认关闭** —— 这是个有数据支撑的决定, 不该被误改

背景: 两种重排器在理财 golden 上都实测掉分 (自研 SparseBoost 与语义精排
gte-rerank-v2 均如此), 详见 src/retrieval/reranker.py 的模块说明。
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest


def _fake_response(payload: dict):
    """伪造 urlopen 的返回 (支持 with 语句)"""
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Resp(json.dumps(payload).encode("utf-8"))


class TestApiReranker:

    def _reranker(self):
        from src.retrieval.reranker import ApiReranker
        return ApiReranker(api_key="test-key")

    def test_returns_empty_and_short_lists_without_calling_api(self, monkeypatch):
        """空候选 / 候选不过 top_k 时不该发请求 (省一次外部调用)"""
        import urllib.request

        def _boom(*a, **k):
            raise AssertionError("不该发起网络请求")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        r = self._reranker()
        assert r.rerank("q", [], top_k=3) == []
        short = [{"content": "a"}, {"content": "b"}]
        assert r.rerank("q", short, top_k=3) == short

    # 候选池必须**大于** top_k, 否则会走「候选不足, 不必重排」的提前返回
    _CANDS = [{"content": f"doc{i}"} for i in range(5)]

    def test_orders_by_relevance_score(self, monkeypatch):
        """按 relevance_score 降序, 并带上 rerank_score 便于追溯"""
        import urllib.request

        payload = {"output": {"results": [
            {"index": 3, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
            {"index": 4, "relevance_score": 0.3},
            {"index": 0, "relevance_score": 0.1},
            {"index": 2, "relevance_score": 0.05},
        ]}}
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: _fake_response(payload))
        out = self._reranker().rerank("q", self._CANDS, top_k=3)
        assert [c["content"] for c in out] == ["doc3", "doc1", "doc4"]
        assert out[0]["rerank_score"] == pytest.approx(0.9)

    def test_skips_out_of_range_indices(self, monkeypatch):
        """接口返回越界下标时跳过, 不让异常数据污染结果"""
        import urllib.request

        payload = {"output": {"results": [
            {"index": 99, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.1},
        ]}}
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: _fake_response(payload))
        out = self._reranker().rerank("q", self._CANDS, top_k=3)
        assert [c["content"] for c in out] == ["doc1", "doc0"]

    def test_raises_on_empty_results(self, monkeypatch):
        """接口返回空结果必须抛错 —— 静默返回原序会让「重排没生效」
        和「重排后顺序恰好没变」无法区分。"""
        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: _fake_response({"output": {"results": []}}))
        with pytest.raises(RuntimeError, match="返回空结果"):
            self._reranker().rerank("q", self._CANDS, top_k=2)

    def test_requires_api_key(self, monkeypatch):
        """没配 key 时必须构造失败, 而不是跑到线上再报错"""
        from src.config import settings
        from src.retrieval.reranker import ApiReranker
        # 注意: 单独传 api_key="" 会被 settings 兜住, 必须把 settings 也清掉
        monkeypatch.setattr(settings, "dashscope_api_key", "")
        with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
            ApiReranker(api_key="")


class TestRerankIsOffByDefault:

    def test_disabled_by_default(self):
        """重排默认关闭 —— 实测掉分, 默认开会让线上检索质量变差。

        要打开请在 .env 设 RERANK_ENABLED=true, 并先跑
        scripts/eval_retrieval.py --rerank --rerank-type api 做 A/B。
        """
        from src.config import settings
        assert settings.rerank_enabled is False

    def test_retriever_has_no_reranker_when_disabled(self):
        from src.retrieval.retriever import HybridRetriever
        r = HybridRetriever()
        if r.reranker is not None:
            pytest.skip("本环境显式开启了 RERANK_ENABLED")
        assert r.rerank_pool == 0

    def test_sparse_boost_documented_as_negative(self):
        """SparseBoost 的实测负收益结论要写在代码里, 防止有人再把它接回去。"""
        import inspect

        from src.retrieval import reranker as mod
        doc = inspect.getdoc(mod)
        assert "负收益" in doc
