"""检索的"产品对齐检查"—— 该拦的拦, 但不许把能答的问题变成答不了。

背景（2026-09-22 实测）: 查询"结构性存款保本吗？和大额存单哪个更适合保守型客户？"
命中了 8 条分块, 但来源被清空、返回"知识库中未检索到「大额存单3年期」…不应采用"。
两个结构缺陷:
  1. 要求**目录全名**逐字出现在检索文本里, 而文本里写的是口语简称("大额存单")
  2. **一票否决**: 任一产品没找到就把整份结果丢掉 —— 于是另一个明明找到的产品也答不了
历史: 同类故障之前出过一次(别名表缺省 "" 恒真 → 判成点名 13 个产品 → 整份丢弃),
当时只改了别名缺省值, 没动"一票否决"这个结构 —— 这次是同一结构的另一次触发。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))


from src.tools import registry


def _patch_retriever(monkeypatch, local):
    class FakeRetriever:
        def hybrid_retrieve(self, query, top_k=5, **kw):
            return {"local": local, "global": [], "graph": []}

    monkeypatch.setattr("src.retrieval.retriever.HybridRetriever", FakeRetriever)


def test_partial_hit_keeps_content_and_warns_about_missing(monkeypatch):
    """两个产品只找到一个 → 保留找到的, 并显式提示缺哪个 (不再一票否决)。"""
    _patch_retriever(monkeypatch, [
        {"content": "大额存单属于一般性存款，受存款保险保障。", "source": "prod_p006.md"},
    ])
    r = registry.retrieve_knowledge("结构性存款保本吗？和大额存单哪个更适合保守型客户？")
    assert r["sources"] == ["prod_p006.md"], "找到的产品不该被丢掉"
    assert "大额存单" in r["text"]
    assert "结构性存款" in r["text"], "缺的那个要在提示里点名"


def test_alias_counts_as_found(monkeypatch):
    """文本里写口语简称(大额存单)也算命中, 不该因为不是目录全名就被判未检索到。"""
    _patch_retriever(monkeypatch, [
        {"content": "大额存单的存款属性与保障口径见下。", "source": "prod_p006.md"},
    ])
    r = registry.retrieve_knowledge("大额存单安全吗？")
    assert r["sources"] == ["prod_p006.md"]


def test_all_missing_still_suppresses(monkeypatch):
    """点名产品一条都没命中 → 维持原有硬抑制(防用别的产品内容冒充)。"""
    _patch_retriever(monkeypatch, [
        {"content": "隐私安全：客服不会索要短信验证码。", "source": "privacy_security.md"},
    ])
    r = registry.retrieve_knowledge("稳盈添利30天的收益是多少？")
    assert r["sources"] == []
    assert "未检索到" in r["text"]


def test_no_product_named_keeps_result(monkeypatch):
    """没点名具体产品的泛问 → 不触发对齐检查。"""
    _patch_retriever(monkeypatch, [
        {"content": "理财产品怎么赎回。", "source": "redeem_and_fee.md"},
    ])
    r = registry.retrieve_knowledge("理财产品怎么赎回？")
    assert r["sources"] == ["redeem_and_fee.md"]
