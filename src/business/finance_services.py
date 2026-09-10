"""
金融理财产品业务服务 — 结构化产品/适当性/申购门控/持仓

统一返回结构: {"text": str(发给LLM), "sources": [], "contexts": {...}, "handoff": bool}
与 src/tools/registry.py 的 EXECUTE_TOOLS 约定一致。
产品主数据来自 data/finance_kb/catalog.jsonl (经 core.slot_filler 共享的 CATALOG)。

设计要点 (阶段二):
  - 资金动账(申购/赎回)由 cs_graph 的确定性门控节点执行并留痕,
    **不注册为 agent 可自由调用的工具** → 机器不赌 LLM 自觉放行。
  - 适当性门控: investor_level(测评) < product_risk → 拦截/转人工。
  - 归属: 持仓/申购只作用于当前用户 (thread-local get_current_user)。
"""

import logging

from .context import get_current_user, set_current_user
from . import mock_finance as mock
from src.core.slot_filler import CATALOG

logger = logging.getLogger(__name__)

_RANK = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}


def _ok(text, handoff=False):
    return {
        "text": text,
        "sources": [],
        "contexts": {"local": [], "global": [], "graph": []},
        "handoff": handoff,
    }


def _audit_money(event: str, user_id: str, detail: str) -> None:
    """资金操作审计 (Redis security:events; 不可用则跳过, 不阻塞)"""
    import time, json
    try:
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        if cache.client:
            cache.client.lpush(
                "security:events",
                json.dumps({"ts": int(time.time()), "event": event, "user": user_id or "anon",
                            "tool": event, "detail": detail[:200]}, ensure_ascii=False),
            )
            cache.client.ltrim("security:events", 0, 499)
    except Exception as e:
        logger.debug("资金操作审计写入失败 (可忽略): %s", e)


def find_product(name_or_id: str):
    """按 id 或名称(含子串)在产品目录中查产品; 找不到返回 None"""
    if not name_or_id:
        return None
    s = str(name_or_id).strip()
    for p in CATALOG:
        if p.get("id") == s or s in p.get("name", ""):
            return p
    return None


def _fmt_products(products: list) -> str:
    lines = []
    for p in products:
        insured = "受存款保险保障" if p.get("deposit_insured") else "非存款、不受存款保险保障"
        lines.append(
            f"- {p['name']}（{p['id']}）：类型 {p['type']}｜风险 {p['risk']}｜期限 {p['term']}｜"
            f"起购 {p.get('min_amount', '-')}{p.get('min_amount_unit', '')}｜{p['yield_desc']}｜{insured}"
        )
    return "\n".join(lines)


def query_products(product_type: str = "", risk: str = "", keyword: str = "") -> dict:
    """结构化产品筛选 (只读): 按类型/风险等级/关键词过滤产品目录并返回清单"""
    products = CATALOG
    if product_type:
        products = [p for p in products if product_type in p.get("type", "")]
    if risk:
        products = [p for p in products if p.get("risk", "").upper() == risk.upper()]
    if keyword:
        kw = keyword.strip()
        products = [p for p in products if kw in p.get("name", "") or kw in p.get("type", "")]

    if not products:
        return _ok("【产品查询】未找到匹配的产品。可尝试：调整风险等级（R1~R5）/ 产品类型（理财、存款、基金、保险）/ 关键词后再查。")
    text = "【产品查询】符合条件的产品：\n" + _fmt_products(products)
    return _ok(text)


def _suitability(product, investor_level: str) -> dict:
    """适当性判定 (纯规则): investor < product.risk → 不匹配"""
    if not product:
        return {"eligible": False, "handoff": False, "reason": "产品不存在"}
    if not investor_level:
        return {"eligible": False, "handoff": False, "reason": "未取得风险测评等级，请先完成测评"}
    p_rank = _RANK.get(str(product.get("risk", "")).upper(), 5)
    i_rank = _RANK.get(str(investor_level).upper(), 0)
    if i_rank >= p_rank:
        return {"eligible": True, "handoff": False, "reason": ""}
    return {
        "eligible": False, "handoff": True,
        "reason": (f"您的风险测评等级为 {investor_level}，低于 {product['name']} 的风险等级 {product['risk']}，"
                   f"按规定不能直接购买。已为您转人工坐席，由坐席完成风险揭示与双录确认（您也可先重新测评或改选 R{max(i_rank,1)} 及以下产品）。"),
    }


def check_suitability(product_id_or_name: str) -> dict:
    """工具: 校验某产品与当前用户测评等级是否匹配 (只读, 确定性判定)"""
    product = find_product(product_id_or_name)
    if not product:
        return _ok(f"【适当性校验】未在产品目录中找到「{product_id_or_name}」。")
    uid = get_current_user()
    profile = mock.get_profile(uid) if uid else None
    level = (profile or {}).get("investor_level") if profile else None
    verdict = _suitability(product, level)
    if verdict["handoff"]:
        return _ok(f"【适当性校验】{verdict['reason']}", handoff=True)
    if not verdict["eligible"]:
        return _ok(f"【适当性校验】{verdict['reason']}")
    return _ok(f"【适当性校验】您的测评等级为 {level}，可购买 {product['name']}（{product['risk']}）。"
               f"购买仍需二次确认与短信验证码。")


def query_holdings() -> dict:
    """工具: 查询当前用户持仓 (只返回本人数据)"""
    uid = get_current_user()
    if not uid:
        return _ok("【持仓查询】未识别到登录用户，请先登录后再查询。")
    holdings = mock.HOLDINGS.get(uid, [])
    if not holdings:
        return _ok("【持仓查询】您当前暂无持仓。可向我咨询理财产品进行配置。")
    lines = [
        f"- {h['product_name']}：{h['amount_wan']}万｜期限 {h['term']}｜状态 {h['status']}"
        for h in holdings
    ]
    return _ok("【持仓查询】您的持仓：\n" + "\n".join(lines))


def subscribe_gate(user_id: str, product_name: str, amount_wan: float, code: str = "") -> dict:
    """申购门控 (阶段二节点用): 归属→产品→测评→金额→验证码二次确认, 全通过才写持仓并留痕

    Returns: {"success": bool, "handoff": bool, "need_code": bool, "text": str, "product": dict|None}
    """
    if not user_id:
        return {"success": False, "handoff": False, "need_code": False,
                "text": "为保障资金安全，办理申购前请先登录（演示账号 u001/u002/u003）。", "product": None}
    profile = mock.get_profile(user_id)
    product = find_product(product_name)
    if not product:
        return {"success": False, "handoff": False, "need_code": False,
                "text": f"未在产品目录中找到「{product_name}」，请核对产品名称（可先问：有哪些 R2 的理财？）。", "product": None}

    # 1) 适当性门控
    level = (profile or {}).get("investor_level")
    verdict = _suitability(product, level)
    if verdict["handoff"]:
        _audit_money("gate_subscribe_rejected_suitability", user_id,
                     f"{product['id']} investor={level} risk={product['risk']}")
        return {"success": False, "handoff": True, "need_code": False, "text": verdict["reason"], "product": product}

    # 2) 金额校验
    try:
        amount = float(amount_wan)
    except (TypeError, ValueError):
        return {"success": False, "handoff": False, "need_code": False,
                "text": "请提供购买金额（万元），例如：申购 5 万。", "product": product}
    if amount < float(product.get("min_amount", 0)):
        return {"success": False, "handoff": False, "need_code": False,
                "text": f"{product['name']} 起购金额为 {product.get('min_amount')}{product.get('min_amount_unit','')}，"
                        f"您填写的 {amount} 万不足起购门槛。", "product": product}

    # 3) 验证码二次确认
    if not code or code != (profile or {}).get("verify_code"):
        return {"success": False, "handoff": False, "need_code": True,
                "text": (f"已为您暂存申购意向：{product['name']} {amount} 万（{product['risk']}）。"
                         f"请提供短信验证码完成二次确认（演示验证码已发送至您的手机）。"),
                "product": product}
    if mock.has_holding(user_id, product["id"]):
        return {"success": False, "handoff": False, "need_code": False,
                "text": f"您已持有 {product['name']}，避免重复申购，请先到持仓页查看。", "product": product}

    # 4) 写持仓 + 留痕
    mock.add_holding(user_id, product, amount)
    insured = "属存款、本息受存款保险保障" if product.get("deposit_insured") else "非存款、不受存款保险保障"
    _audit_money("subscribe_success", user_id, f"{product['id']} amount={amount}万")
    text = (f"✅ 申购已确认：{product['name']} {amount} 万（{product['risk']}，{product['term']}）。\n"
            f"{insured}。产品具体收益以说明书为准：业绩比较基准仅为参考、不构成收益承诺。\n"
            f"已记入您的持仓，可随时问我“查看我的持仓”。")
    return {"success": True, "handoff": False, "need_code": False, "text": text, "product": product}
