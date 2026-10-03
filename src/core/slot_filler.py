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


# ── L0 产品域 (由 catalog 派生, 不要另写产品词表) ──────────────────────────
# 7 个产品域 + common(制度/通用, 判不出时的落点) + out_of_scope。
# 产品域与 KB 的一级大类一一对应 (见 kg_domain_triples.jsonl 的 is_subtype_of 链),
# 不是按 catalog.type 字符串个数拍出来的。
L0_DOMAINS = ("deposit", "wealth", "fund", "insurance", "bond", "trust", "gold",
              "common", "out_of_scope")

# 键 = catalog.type 的括号前缀, 闭合集合。新产品一般落在已有类别里; 真出现新类别时
# _UNMAPPED_TYPES 会告警、tests/test_l0_derivation.py 会失败 —— 提醒补这里, 而不是静默判错。
_L0_BY_TYPE = {
    # 存款 (is_subtype_of 存款, 受存款保险保障)
    "存款": "deposit", "结构性存款": "deposit",
    # 理财 —— 含现金管理类: KB 明确"现金管理类产品属于理财产品, 非存款"(catalog P001 自述 +
    # kg: 现金管理类理财 is_subtype_of 理财产品), 旧实现把它判成 deposit 是错的
    "现金管理": "wealth", "固定收益": "wealth", "固收增强": "wealth", "私银混合": "wealth",
    # 代销
    "公募基金": "fund", "保险": "insurance",
    # KB 里各自独立成类, 均 'is_not 存款'
    "债券": "bond", "信托": "trust", "贵金属": "gold",
}

# 能直接定域的意图 (制度/通用类); 其余靠产品槽位判, 判不出落 common
_L0_BY_INTENT = {
    "deposit_insurance": "common", "fraud_report": "common", "complaint": "common",
    "service_policy": "common", "chitchat": "common", "unknown": "common",
}


def _type_to_l0(product_type: str) -> str:
    """catalog 的 type 串 → L0; 认不出的类别返回 '' (不猜)"""
    return _L0_BY_TYPE.get((product_type or "").split("(")[0].strip(), "")


# 产品名 → L0 (与 KNOWN_PRODUCTS 同源; 产品名在 extract_slots 里已按 catalog 精确匹配)
_PRODUCT_TO_L0 = {p["name"]: _type_to_l0(p.get("type", "")) for p in CATALOG if p.get("name")}

_UNMAPPED_TYPES = ({p["type"].split("(")[0].strip() for p in CATALOG if p.get("type")}
                   - set(_L0_BY_TYPE))
if _UNMAPPED_TYPES:
    logger.warning("catalog 有未映射 L0 的产品类别 (将落 common): %s", sorted(_UNMAPPED_TYPES))


def intent_to_l0(intent: str = "", slots: dict = None) -> str:
    """推断 L0 产品域 (见 L0_DOMAINS)

    优先级: catalog 精确产品名 → 产品类型 → 意图映射 → common。
    "common" 是**判定不出**时的通用域, 不是假装知道 —— 旧实现默认返回 wealth,
    会让基金/存款类问题套上理财的合规话术。
    """
    slots = slots or {}
    l0 = (_PRODUCT_TO_L0.get(slots.get("product") or "", "")
          or _type_to_l0(slots.get("product_type") or ""))
    return l0 or _L0_BY_INTENT.get(intent or "", "common")

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


def is_executable_order(intent: str, slots: dict) -> bool:
    """这一轮是否构成可执行的下单指令: 明确购买意图 + 已给出金额.

    「申购先分析」语义的唯一定义点 —— cs_graph._should_enter_subscribe (有产品→进门控)
    与 missing_required (缺产品→追问) 都从这一条派生, 避免两处判据各自漂移。
    """
    return intent == "buy_process" and (slots or {}).get("amount") is not None


def missing_required(intent: str = "", slots: dict = None) -> list:
    """返回当前意图缺失的必需槽位名列表.

    必填槽位**只在执行流生效**: 只有当用户这一轮构成了可执行的下单指令
    (购买意图 + 已给出金额), 却没说买哪款产品时, 才追问产品名。

    咨询/查询类一律不追问 —— 信息型问题本来就不保证带得出产品名,
    按产品域硬性要求会拦掉绝大多数正常提问 (实测 105 条 Golden 中 75 条)。
    判据与 cs_graph._should_enter_subscribe 共用 is_executable_order:
    该函数要求 product + amount 同时具备才进门控, 缺产品时本就落回产品分析, 这里补齐缺口。
    """
    slots = slots or {}
    if not is_executable_order(intent, slots):
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
