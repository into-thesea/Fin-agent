"""缓存隐私守卫测试 — 用户私有/资金动账类不命中共享缓存"""

from src.cache.privacy import is_private_or_money_query


def test_private_queries_skipped():
    assert is_private_or_money_query("我的持仓怎么样")
    assert is_private_or_money_query("验证码是123456")
    assert is_private_or_money_query("我要买5万稳盈添利30天")
    assert is_private_or_money_query("帮我把安心固收90天赎回")
    assert is_private_or_money_query("查一下我的账号余额")


def test_generic_queries_cacheable():
    assert not is_private_or_money_query("稳盈添利30天保本吗")
    assert not is_private_or_money_query("理财产品怎么买")
    assert not is_private_or_money_query("存款保险赔多少")
    assert not is_private_or_money_query("理财能随时赎回吗")
