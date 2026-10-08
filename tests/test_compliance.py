"""合规检查测试 — 敏感词/不当承诺掩码"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.core.compliance import check


def test_clean_answer():
    r = check("京东快递首重15元，续重每公斤5元。")
    assert r["clean"] is True
    assert r["hits"] == []
    assert r["masked"] == "京东快递首重15元，续重每公斤5元。"


def test_sensitive_word_masked():
    r = check("我们保证48小时内发货")
    assert r["clean"] is False
    assert "保证" in r["hits"]
    assert "保证" not in r["masked"]
    assert "**" in r["masked"]


def test_absolute_claim_masked():
    r = check("这是全网最低价，绝对安全")
    assert r["clean"] is False
    assert "最低价" in r["hits"] or "绝对" in r["hits"]
    assert "绝对" not in r["masked"]


def test_empty_input():
    r = check("")
    assert r["clean"] is True
