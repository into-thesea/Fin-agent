"""greeting 不经 reviewer, 不得谎报分数."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHAT = open(os.path.join(ROOT, "src", "api", "routes", "chat.py"), encoding="utf-8").read()


def test_chat_does_not_hardcode_review_pass():
    """chat.py 里不得再出现写死的 review 分数或 pass 结论"""
    assert '"score": 85' not in CHAT, "chat.py 里仍有写死的 review 分数"
    assert re.search(r'review=\{?"score": 85', CHAT) is None


def test_greeting_review_is_marked_skipped():
    """greeting 的 review 必须标 skipped 且分数为 null (前端 types/api.ts 把 score 声明为必填)"""
    hits = re.findall(r'"review":\s*\{[^}]*\}', CHAT) + re.findall(r'review=\{[^}]*\}', CHAT)
    greeting_reviews = [h for h in hits if '"skipped"' in h]
    assert greeting_reviews, "没有找到标 skipped 的 review"
    for h in greeting_reviews:
        assert '"score": None' in h, f"skipped 的 review 应带 score: None, 实际: {h}"
        assert '"pass"' not in h


def test_greeting_confidence_is_not_faked():
    """greeting 没有真实置信度, 不得写死一个数字"""
    assert '"_confidence": 0.85' not in CHAT
    assert '"_confidence": None' in CHAT


if __name__ == "__main__":
    test_chat_does_not_hardcode_review_pass()
    test_greeting_review_is_marked_skipped()
    test_greeting_confidence_is_not_faked()
    print("OK: greeting review")
