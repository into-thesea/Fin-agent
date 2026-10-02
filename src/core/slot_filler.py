"""
槽位填充 (Slot Filling) — 从查询/实体中提取金融理财产品槽位, 判断缺失

理财域槽位: product(产品名) / product_type(大类) / amount(金额, 万元) /
             term(期限) / risk_level(产品风险或测评等级 R1~R5)
产品名与产品类型以 data/finance_kb/catalog.jsonl 为唯一事实源。

extract_slots     从单轮查询 + 路由实体中提取槽位
missing_required  判断当前意图还缺哪些必需槽位 (用于追问, 资金动账流程在阶段二接入后启用)
"""

import os
import re
import json
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "finance_kb", "catalog.jsonl")


def _load_catalog() -> list:
    """读取产品目录 → [ {id,name,type,risk,...} ] (文件缺失/解析失败时返回空, 不阻塞启动)"""
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logger.warning("catalog.jsonl 读取失败, 产品词表为空: %s", e)
        return []


CATALOG = _load_catalog()
KNOWN_PRODUCTS = [p["name"] for p in CATALOG if p.get("name")]
KNOWN_PRODUCT_TYPES = list(dict.fromkeys(p.get("type", "") for p in CATALOG if p.get("type")))

# 金额: 如 50万 / 50000元 / 1万元 → 统一存万元 (float)
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万|w|元)")
# 期限: 如 30天 / 3个月 / 1年 / 3年期
_TERM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(天|个月|月|年)")
# 风险/测评等级 (中文后无词边界, 不能用 \b 包住 R)
_RISK_RE = re.compile(r"R([1-5])(?!\d)", re.IGNORECASE)
# 大类兜底关键词
_TYPE_KEYWORDS = [
    ("存款", "存款"), ("大额存单", "存款"), ("结构性存款", "存款"),
    ("现金管理", "现金管理(货币类)"), ("货币", "现金管理(货币类)"),
    ("固收增强", "固收增强"), ("固定收益", "固定收益"),
    ("纯债", "公募基金(纯债, 代销)"), ("混合基金", "公募基金(偏股混合, 代销)"),
    ("增额", "保险(增额终身寿, 代销)"), ("终身寿", "保险(增额终身寿, 代销)"),
    ("年金", "保险(养老年金, 代销)"), ("保险", "保险"),
    ("基金", "基金"), ("理财", "理财"),
]


def _to_wan(m: re.Match) -> float:
    """把金额匹配转成万元 float (50000元 → 5.0)"""
    num = float(m.group(1))
    unit = m.group(2)
    if unit in ("万", "w", "W"):
        return num
    return num / 10000.0  # 元 → 万


def extract_slots(query: str = "", entities: list = None) -> dict:
    """从查询文本 + 路由实体中提取理财槽位 → {product, product_type, amount, term, risk_level, ...}"""
    slots = {}
    q = query or ""

    # 产品名 (精确匹配目录, 最长优先)
    for name in sorted(KNOWN_PRODUCTS, key=len, reverse=True):
        if name in q:
            slots["product"] = name
            break

    # 产品大类 (文本关键词)
    for kw, val in _TYPE_KEYWORDS:
        if kw in q:
            slots["product_type"] = val
            break

    # 金额 → 万元
    m = _AMOUNT_RE.search(q)
    if m:
        slots["amount"] = _to_wan(m)
        slots["amount_raw"] = m.group(0)

    # 期限
    m2 = _TERM_RE.search(q)
    if m2:
        unit = m2.group(2)
        unit_map = {"天": "天", "个月": "个月", "月": "个月", "年": "年"}
        slots["term"] = f"{m2.group(1)}{unit_map[unit]}"

    # 风险/测评等级
    m3 = _RISK_RE.search(q)
    if m3:
        slots["risk_level"] = f"R{m3.group(1)}".upper()

    # 路由实体补充 (LLM 提取的 Product/ProductType/Amount/Term/RiskLevel)
    for ent in (entities or []):
        t = ent.get("type", "")
        n = ent.get("name", "")
        if t == "Product" and n and "product" not in slots:
            slots["product"] = n
        elif t == "ProductType" and n and "product_type" not in slots:
            slots["product_type"] = n
        elif t == "RiskLevel" and n and "risk_level" not in slots:
            slots["risk_level"] = str(n).upper()
        elif t == "Amount" and n and "amount" not in slots:
            mm = _AMOUNT_RE.search(str(n))
            if mm:
                slots["amount"] = _to_wan(mm)
        elif t == "Term" and n and "term" not in slots:
            slots["term"] = n

    return slots


def missing_required(intent: str = "", slots: dict = None) -> list:
    """返回当前意图缺失的必需槽位名列表.

    必填槽位**只在执行流生效**: 只有当用户这一轮构成了可执行的下单指令
    (购买意图 + 已给出金额), 却没说买哪款产品时, 才追问产品名。

    咨询/查询类一律不追问 —— 信息型问题本来就不保证带得出产品名,
    按产品域硬性要求会拦掉绝大多数正常提问 (实测 105 条 Golden 中 75 条)。
    与 cs_graph._should_enter_subscribe 的激活条件同源: 该函数要求
    product + amount 同时具备才进门控, 缺产品时本就落回产品分析, 这里补齐缺口。
    """
    slots = slots or {}
    if intent != "buy_process" or slots.get("amount") is None:
        return []
    return [] if slots.get("product") else ["product"]


def followup_question(missing: list) -> str:
    """根据缺失槽位生成追问话术"""
    prompt = {
        "product": "请问您咨询的是哪款产品？(如：稳盈添利30天 / 大额存单 / 安鑫纯债基金)",
        "product_type": "请问您指的是理财产品、存款还是基金/保险？",
        "amount": "请问您计划投入的金额是多少？",
        "term": "请问资金可以放多久（如 30 天 / 3 个月 / 1 年）？",
        "risk_level": "请问您的风险测评等级是 R1~R5 中的哪一档？",
    }
    return " ".join(prompt.get(s, f"请补充: {s}") for s in missing)
