"""情绪处理测试 —— 关键词检测 + 情绪分级 + prompt 注入

覆盖 emotion.py 的四个入口与 cs_graph 的接线点 _emotion_prompt。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.core.emotion import (
    detect_emotion,
    is_negative,
    build_emotion_instruction,
    should_suggest_handoff,
)
from src.graph.cs_graph import _emotion_prompt, _empty_state


def test_is_negative():
    assert is_negative("银行理财亏了怎么办")
    assert is_negative("我要投诉")
    assert not is_negative("稳盈添利30天收益多少")


def test_detect_emotion_priority():
    # 投诉优先级高于焦虑: 同一句里同时命中时选 complaint
    assert detect_emotion("我要投诉，本金都亏了") == "complaint"
    assert detect_emotion("亏了怎么办") == "anxiety"
    assert detect_emotion("没听懂，能解释一下吗") == "confusion"
    assert detect_emotion("稳盈添利30天收益多少") == "normal"


def test_detect_emotion_escalates_on_repeated_negatives():
    # 本轮没有关键词, 但已连续两轮负面 → 升级为焦虑, 不再反复追问
    assert detect_emotion("然后呢", negative_turns=2) == "anxiety"
    assert detect_emotion("然后呢", negative_turns=0) == "normal"


def test_build_emotion_instruction():
    assert build_emotion_instruction("normal") == ""
    for emo in ("anxiety", "complaint", "confusion"):
        assert build_emotion_instruction(emo)
    # 未知情绪不应抛错
    assert build_emotion_instruction("whatever") == ""


def test_should_suggest_handoff():
    assert should_suggest_handoff("complaint", turn_count=1, negative_turns=0)
    assert should_suggest_handoff("normal", turn_count=1, negative_turns=2)
    assert should_suggest_handoff("anxiety", turn_count=3, negative_turns=0)
    # 一轮内的普通提问不该被推去人工
    assert not should_suggest_handoff("normal", turn_count=1, negative_turns=0)
    assert not should_suggest_handoff("confusion", turn_count=2, negative_turns=0)


def test_emotion_prompt_injection():
    base = "你是银行理财顾问。"
    # normal: 原样返回, 不污染提示词
    assert _emotion_prompt(base, {"emotion": "normal"}) == base
    assert _emotion_prompt(base, {}) == base

    # 焦虑: 追加共情指令, 保留原提示词
    p = _emotion_prompt(base, {"emotion": "anxiety"})
    assert p.startswith(base)
    assert "共情" in p

    # 连续负面: 追加转人工建议
    p2 = _emotion_prompt(base, {"emotion": "anxiety", "negative_turns": 2, "turn_count": 1})
    assert "转人工" in p2


def test_empty_state_carries_emotion_fields():
    # 无 session_id 时不查会话状态, 用默认值, 不得抛错
    st = _empty_state("稳盈添利30天收益多少")
    assert st["emotion"] == "normal"
    assert st["negative_turns"] == 0
    assert st["turn_count"] == 0
