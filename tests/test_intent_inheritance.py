"""续轮意图继承 —— 单元测试

覆盖: 验证码/确认/纯金额/短问法继承, 话题切换不继承, 首轮不继承,
咨询类意图短问法不继承 (避免错误传染)。
"""
import pytest
from src.core.intent_inheritance import try_inherit_intent, is_topic_switch


class TestTopicSwitch:
    def test_deposit_insurance_is_switch(self):
        assert is_topic_switch("存款保险保什么")

    def test_fraud_is_switch(self):
        assert is_topic_switch("我被骗了")

    def test_complaint_is_switch(self):
        assert is_topic_switch("我要投诉")

    def test_normal_query_not_switch(self):
        assert not is_topic_switch("犹豫期是多久")

    def test_income_keyword_is_switch(self):
        # 收益/利率属于新问询类型, 流程类续轮遇到时不应继承
        assert is_topic_switch("收益率是多少")


class TestInherit:
    def test_verification_code_inherits_buy(self):
        hit, intent = try_inherit_intent("验证码 123456", "buy_process")
        assert hit and intent == "buy_process"

    def test_verification_code_not_buy_no_inherit(self):
        # 上一轮不是申购, 验证码不应继承为 buy_process
        hit, _ = try_inherit_intent("验证码 123456", "product_consult")
        assert not hit

    def test_confirm_inherits_hold_redeem(self):
        hit, intent = try_inherit_intent("好的", "hold_redeem")
        assert hit and intent == "hold_redeem"

    def test_confirm_inherits_buy(self):
        hit, intent = try_inherit_intent("确认", "buy_process")
        assert hit and intent == "buy_process"

    def test_pure_amount_inherits_when_awaiting_amount(self):
        hit, intent = try_inherit_intent("50万", "buy_process", awaiting_slot="amount")
        assert hit and intent == "buy_process"

    def test_pure_amount_consult_no_inherit(self):
        # 咨询类意图下, 纯金额不触发规则继承 (规则 d 只对流程类意图生效)
        hit, _ = try_inherit_intent("50万", "product_consult")
        assert not hit

    def test_short_query_inherits_hold_redeem(self):
        # 犹豫期是退保流程的延续, 上一轮 hold_redeem → 继承
        hit, intent = try_inherit_intent("犹豫期是多久？", "hold_redeem")
        assert hit and intent == "hold_redeem"

    def test_short_query_with_income_no_inherit(self):
        # 含"收益"→ 话题切换, 不继承
        hit, _ = try_inherit_intent("收益率多少", "hold_redeem")
        assert not hit

    def test_principal_question_no_inherit(self):
        # "本金"属收益/保本问询, 话题切换, 不继承 (防止误继承为 hold_redeem)
        hit, _ = try_inherit_intent("那我的本金还能拿回来吗", "hold_redeem")
        assert not hit

    def test_trust_question_no_inherit(self):
        # "靠谱"属信任评价, 话题切换, 不继承
        hit, _ = try_inherit_intent("你们这个到底靠不靠谱", "hold_redeem")
        assert not hit

    def test_short_query_10chars_no_inherit(self):
        # 阈值 <10, 恰好10字不触发短问法继承
        hit, _ = try_inherit_intent("犹豫期到底是多少天呢", "hold_redeem")
        assert not hit

    def test_first_turn_no_inherit(self):
        hit, _ = try_inherit_intent("大额存单利率", None)
        assert not hit

    def test_consult_intent_short_query_no_inherit(self):
        # 咨询类意图的续轮仍交 LLM, 规则 d 不生效, 避免错误传染
        hit, _ = try_inherit_intent("风险等级呢", "product_consult")
        assert not hit

    def test_topic_switch_overrides_inherit(self):
        # 上一轮 buy_process, 这一轮问存款保险 → 切换, 不继承
        hit, _ = try_inherit_intent("存款保险保什么", "buy_process")
        assert not hit

    def test_fraud_keyword_no_inherit(self):
        hit, _ = try_inherit_intent("我被骗了怎么办", "hold_redeem")
        assert not hit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
