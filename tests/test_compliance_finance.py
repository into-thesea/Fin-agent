"""金融合规测试 — PII 掩码 / 收益承诺禁语 / 风险提示"""

from src.core.compliance import check, finalize, risk_note_needed


def test_pii_masked_phone_id_card():
    r = check("我的手机 13812345678，身份证 11010119900307623X，卡号 6222021234567890123")
    assert r["pii"] and r["clean"] is False
    assert "13812345678" not in r["masked"]
    assert "11010119900307623X" not in r["masked"]
    assert "6222021234567890123" not in r["masked"]
    assert "*" in r["masked"]


def test_finance_banned_words():
    r = check("这款理财稳赚不赔，零风险，承诺保本高收益")
    assert r["clean"] is False
    assert "稳赚不赔" in r["hits"]
    assert "稳赚" not in r["masked"]


def test_risk_note_needed_and_appended():
    text = "平衡增利180天业绩比较基准4.0%，收益不错。"
    assert risk_note_needed(text)
    out = finalize(text)
    assert "非存款" in out and "不构成收益承诺" in out


def test_risk_note_not_needed_for_deposit():
    text = "大额存单执行利率3.0%，到期还本付息。"
    assert not risk_note_needed(text)


def test_risk_note_not_duplicated_when_present():
    text = "稳盈添利30天业绩比较基准3.0%（仅为参考，不构成收益承诺），产品有风险。"
    assert not risk_note_needed(text)
    assert finalize(text).count("不构成收益承诺") == 1
