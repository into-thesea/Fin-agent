"""
Fin-Agent 语义缓存系统 (v2.0 — 轻量 + 持久化)

三级缓存匹配:
  L1 — 精确匹配: MD5(归一化查询) 直接命中 (O(1), ~0ms)
  L2 — 轻量语义: TF-IDF 向量余弦相似度 (独立于 BGE 模型, ~5ms)
  L3 — 深度语义: BGE embedding 余弦相似度 (需要加载 BGE, ~200ms 首加载)

存储层级:
  Memory: LRU 热缓存 (最快, 进程级)
  SQLite: 持久化缓存索引 (跨进程/重启, 100ms 级)
  Redis: 分布式共享缓存 (多 worker 间共享, 网络延迟)

特性:
  - 自动降级: L2 在无 sklearn 时回退到 char n-gram Jaccard
  - LRU 淘汰: 超出 MAX_ENTRIES 时淘汰最久未命中条目
  - 命中统计: 记录 hit/miss/avg_latency
  - 后台刷新: 定时清理过期条目
  - 超时安全: 所有外部调用都有 timebudget
"""

import os
import re
import json
import time
import hashlib
import logging
import sqlite3
import threading
from collections import OrderedDict
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── 配置 ────────────────────────────────────────
CACHE_TTL = 3600            # 1 小时
MAX_ENTRIES = 500           # LRU 最大条目数
L2_THRESHOLD = 0.50         # L2 相似度阈值 (配合年份守卫; 0.28 曾导致同公司不同问题串答)
L3_THRESHOLD = 0.90         # L3 (BGE) 阈值。实测: 真同义 0.947~0.951, 误命中最高 0.771,
                            # 0.90 落在安全区中间 (原先 0.92 会漏掉"存款保险赔多少"vs"赔付额度是多少"=0.911)
AUTO_CLEAN_INTERVAL = 300   # 后台清理间隔 (秒)

# ── 项目根 ──────────────────────────────────────
# 项目根路径统一走 src.infra.paths。
# 原先这里自己算, 但少了一层 —— 根被算成 src/, 缓存库落到 src/data/semantic_cache.db,
# 而 scripts/clear_cache.py 清的是 data/semantic_cache.db, 于是"清缓存"从未真正生效。
try:
    from src.infra.paths import DATA_DIR as CACHE_DB_DIR
except Exception:  # 兜底: 从 src/core/ 往上三级
    CACHE_DB_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CACHE_DB_PATH = os.path.join(CACHE_DB_DIR, "semantic_cache.db")

# 确保 data 目录存在
os.makedirs(CACHE_DB_DIR, exist_ok=True)

# ── 全局状态 ────────────────────────────────────
_lock = threading.Lock()

# L1: 精确缓存 {md5_key: (data_json, timestamp)}
_l1_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

# L2 引擎
_l2_entries: list[dict] = [] # [{key, query, data_json, ts}, ...]
_l2_dirty = False             # 需要重建索引

# 统计
_stats = {
    "l1_hits": 0, "l1_misses": 0,
    "l2_hits": 0, "l2_misses": 0,
    "l3_hits": 0, "l3_misses": 0,
    "total_entries": 0,
    "evictions": 0,
    "started_at": time.time(),
}

# ──────────────────────────────────────────────
# SQLite 持久化
# ──────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """获取 SQLite 连接 (线程安全, 每个线程独立连接)"""
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=5, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")       # 写前日志, 高并发读
    conn.execute("PRAGMA synchronous=NORMAL")     # 平衡性能与安全
    conn.execute("PRAGMA cache_size=-8000")       # 8MB 页缓存
    return conn


def _init_db():
    """建表 (幂等)"""
    try:
        conn = _get_db()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                md5_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                data TEXT NOT NULL,
                embedding TEXT,          -- JSON float list or NULL
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                access_count INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_cache_access
            ON semantic_cache(last_access)
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("SQLite 缓存初始化失败 (降级为仅内存): %s", e)


def _load_from_db() -> int:
    """启动时从 SQLite 加载缓存到内存 (只加载未过期的)"""
    global _l1_cache, _l2_entries
    try:
        conn = _get_db()
        now = time.time()
        cutoff = now - CACHE_TTL
        rows = conn.execute(
            "SELECT md5_key, query, data, embedding, created_at, last_access, access_count "
            "FROM semantic_cache WHERE created_at > ? ORDER BY last_access DESC LIMIT ?",
            (cutoff, MAX_ENTRIES)
        ).fetchall()
        conn.close()

        count = 0
        for row in rows:
            md5_key, query, data_json, emb_json, created_at, last_access, access_count = row
            # 入 L1
            with _lock:
                _l1_cache[md5_key] = (data_json, created_at)
                # L2 条目 (含查询文本)
                _l2_entries.append({
                    "key": md5_key,
                    "query": query,
                    "data_json": data_json,
                    "ts": created_at,
                })
            count += 1

        # 标记需要重建 TF-IDF 矩阵
        global _l2_dirty
        _l2_dirty = True

        logger.info("从 SQLite 加载了 %d 条缓存", count)
        return count
    except Exception as e:
        logger.warning("SQLite 加载失败: %s", e)
        return 0


def _save_to_db(md5_key: str, query: str, data_json: str, embedding: Optional[list] = None):
    """写入单条到 SQLite"""
    try:
        conn = _get_db()
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO semantic_cache "
            "(md5_key, query, data, embedding, created_at, last_access, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            (md5_key, query, data_json,
             json.dumps(embedding) if embedding else None,
             now, now)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("SQLite 写入失败: %s", e)


def _clean_expired_db():
    """清理 SQLite 中过期的缓存条目"""
    try:
        conn = _get_db()
        cutoff = time.time() - CACHE_TTL
        cursor = conn.execute("DELETE FROM semantic_cache WHERE created_at < ?", (cutoff,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted:
            logger.info("SQLite 过期清理: %d 条", deleted)
    except Exception as e:
        logger.debug("SQLite 清理失败: %s", e)


# ──────────────────────────────────────────────
# L2: 轻量文本相似度 (TF-IDF / N-gram 降级)
# ──────────────────────────────────────────────

def _rebuild_l2_index():
    """L2 索引占位 (目前使用 n-gram, 无需重建)"""
    global _l2_dirty
    _l2_dirty = False


# 金融关键词 — 带权重的语义等价组
_SYNONYM_GROUPS = [
    {"营收", "收入", "营业收入", "营业总收入"},
    {"利润", "净利润", "净利", "盈利"},
    {"毛利率", "毛利润", "毛利"},
    {"销量", "销售", "出售", "卖出"},
    {"市值", "估值", "价值"},
    {"增长", "增加", "上升", "上涨", "提高"},
    {"下降", "减少", "下跌", "降低", "下滑"},
    {"对比", "比较", "vs", "versus", "与", "跟"},
    {"趋势", "走势", "变化", "发展"},
    {"多少", "怎么", "如何", "为何", "为什么", "是什么"},
    {"今年", "去年", "明年"},

]
# 展平为 {word: group_id}
_SYNONYM_MAP = {}
for gid, group in enumerate(_SYNONYM_GROUPS):
    for word in group:
        _SYNONYM_MAP[word] = gid


def _preprocess(q: str) -> str:
    """预处理: 去标点, 归一化, 保中文/英文/数字"""
    q = q.lower().strip()
    q = re.sub(r'[^\w一-鿿]', '', q)
    return q


def _extract_keywords(q: str) -> set:
    """提取金融查询关键词 (单字+双字+公司名+数字)"""
    q = _preprocess(q)
    words = set()

    # 双字词
    words.update(q[i:i+2] for i in range(len(q)-1))
    # 三字词
    words.update(q[i:i+3] for i in range(len(q)-2))
    # 单独的数字
    for m in re.findall(r'\d+', q):
        words.add(m)
    # 单独英文词
    for m in re.findall(r'[a-z]+', q):
        words.add(m)

    return words


def _keyword_jaccard(q1: str, q2: str) -> float:
    """
    金融关键词 Jaccard (带同义词组映射)

    将词映射到同义词组 ID，实现"营收↔营业收入"类匹配
    """
    k1 = _extract_keywords(q1)
    k2 = _extract_keywords(q2)

    if not k1 or not k2:
        return 0.0

    # 映射到同义词组
    def _map_group(words: set) -> set:
        mapped = set()
        for w in words:
            if w in _SYNONYM_MAP:
                mapped.add(f"__syn_{_SYNONYM_MAP[w]}__")
            else:
                mapped.add(w)
        return mapped

    m1 = _map_group(k1)
    m2 = _map_group(k2)

    inter = len(m1 & m2)
    union = len(m1 | m2)
    return inter / union if union > 0 else 0.0


def _char_ngram_jaccard(q1: str, q2: str, n: int = 3) -> float:
    """字符级 n-gram Jaccard"""
    a = _preprocess(q1)
    b = _preprocess(q2)
    if not a or not b:
        return 0.0

    grams_a = set(a[i:i+n] for i in range(len(a)-n+1))
    grams_b = set(b[i:i+n] for i in range(len(b)-n+1))

    if not grams_a or not grams_b:
        return 0.0

    inter = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return inter / union if union > 0 else 0.0


# 已知公司名列表 (用于跨公司假阳性防护)
_KNOWN_COMPANIES = [
    "比亚迪", "宁德时代", "特斯拉", "小米", "蔚来", "理想", "小鹏",
    "腾讯", "阿里", "华为", "字节", "美团", "京东", "拼多多",
    "百度", "网易", "茅台", "中石油", "中石化", "工商银行",
    "中国平安", "中国人寿", "宁德", "同花顺", "BYD", "TSLA", "CATL",
]


def _extract_company(q: str) -> str | None:
    """提取查询中的公司名"""
    q = _preprocess(q)
    for c in _KNOWN_COMPANIES:
        if c in q:
            return c
    # 尝试匹配 "XX集团" "XX股份" 模式
    m = re.search(r'([一-鿿]{2,6})(?:集团|股份|有限)', q)
    if m:
        return m.group(1)
    return None


def _has_multiple_companies(q: str) -> bool:
    """检测查询是否涉及多公司 (对比类查询)"""
    q = _preprocess(q)
    found = 0
    for c in _KNOWN_COMPANIES:
        if c in q:
            found += 1
            if found >= 2:
                return True
    return False


# ── 理财域守卫 ──────────────────────────────────
# 原有守卫 (公司名/年份) 是为年报问答建的, 在理财产品域全部空转:
# _extract_company 对"稳盈添利30天""天天利货币基金"一律返回 None。
# 实测裸相似度下"理财产品怎么赎回" vs "怎么申购" = 0.518 > 阈值 0.50 → 会串答。
# 下面补两道理财域守卫。

_PRODUCT_CACHE = {"mtime": 0.0, "names": []}


def _known_products() -> list:
    """从 catalog.jsonl 读产品名 (按 mtime 缓存)"""
    path = os.path.join(CACHE_DB_DIR, "finance_kb", "catalog.jsonl")
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return []
    if mt != _PRODUCT_CACHE["mtime"]:
        names = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n = json.loads(line).get("name", "")
                        if n:
                            names.append(n)
        except Exception as e:
            logger.debug("产品名读取失败: %s", e)
        _PRODUCT_CACHE["names"] = sorted(names, key=len, reverse=True)  # 长名优先, 防子串误配
        _PRODUCT_CACHE["mtime"] = mt
    return _PRODUCT_CACHE["names"]


def _extract_product(q: str) -> Optional[str]:
    for name in _known_products():
        if name in q:
            return name
    return None


# 动作词分组: 跨组即视为不同意图 (申购 ≠ 赎回, 这是最容易串答的一对)
_ACTION_GROUPS = {
    "买入": ("申购", "购买", "买入", "认购", "追加", "下单", "买"),
    "卖出": ("赎回", "卖出", "取出", "变现", "撤单", "退出"),
    "查询": ("查询", "查看", "查一下", "看看", "查"),
}


def _action_groups(q: str) -> set:
    return {g for g, kws in _ACTION_GROUPS.items() if any(k in q for k in kws)}


def _l2_similarity(q1: str, q2: str) -> float:
    """
    L2 混合相似度: 关键词 Jaccard (0.6) + 字 n-gram (0.4)

    跨公司防护: 如果检测到公司名不同, 直接返回 0

    - 关键词带同义词组映射 → 处理"营收↔收入"
    - 字 n-gram → 处理语序变化
    """
    # 跨公司防护
    c1 = _extract_company(q1)
    c2 = _extract_company(q2)
    if c1 and c2 and c1 != c2 and c1 not in c2 and c2 not in c1:
        return 0.0

    # 对比查询不应匹配单公司查询
    if _has_multiple_companies(q1) != _has_multiple_companies(q2):
        return 0.0

    # 年份守卫: 明确年份不同 → 不命中 (2024 数据 ≠ 2025 数据)
    _y1 = re.search(r"(20\d{2})", q1)
    _y2 = re.search(r"(20\d{2})", q2)
    if _y1 and _y2 and _y1.group(1) != _y2.group(1):
        return 0.0

    # 内容守卫: 剥掉公司名+年份后, 剩余内容完全不重叠 → 不同问题 (如"营收" vs "销量")
    _c1 = _preprocess(q1)
    _c2 = _preprocess(q2)
    for _c in _KNOWN_COMPANIES:
        _c1 = _c1.replace(_c, "")
        _c2 = _c2.replace(_c, "")
    _c1 = re.sub(r"20\d{2}", "", _c1)
    _c2 = re.sub(r"20\d{2}", "", _c2)
    if _c1 and _c2:
        _b1 = set(_c1[i:i+2] for i in range(len(_c1)-1))
        _b2 = set(_c2[i:i+2] for i in range(len(_c2)-1))
        if not (_b1 & _b2):
            return 0.0

    # ── 理财域守卫 (上面几道是为年报域建的, 此处基本空转) ──

    # 产品名守卫: 两边都点了具体产品且不是同一个 → 不同问题
    _p1 = _extract_product(q1)
    _p2 = _extract_product(q2)
    if _p1 and _p2 and _p1 != _p2:
        return 0.0

    # 动作守卫: 一边申购一边赎回 → 不同意图 (实测"怎么赎回"vs"怎么申购"=0.518, 会被串答)
    _g1 = _action_groups(q1)
    _g2 = _action_groups(q2)
    if _g1 and _g2 and not (_g1 & _g2):
        return 0.0

    kw_sim = _keyword_jaccard(q1, q2)
    ng_sim = _char_ngram_jaccard(q1, q2, n=3)
    return 0.6 * kw_sim + 0.4 * ng_sim


def _l2_search(query: str) -> tuple[Optional[str], float]:
    """
    L2 轻量语义搜索: 混合 Jaccard (关键词 + n-gram)

    使用同义词组映射处理"换表述"场景:
      "营收"↔"营业收入", "销量"↔"销售", "对比"↔"比较"

    Returns:
        (data_json, best_score) — 未命中时 data_json=None
    """
    if not _l2_entries:
        return None, 0.0

    if _l2_dirty:
        _rebuild_l2_index()

    best_score, best_data = 0.0, None
    now = time.time()
    for entry in _l2_entries:
        if now - entry["ts"] > CACHE_TTL:
            continue
        score = _l2_similarity(query, entry["query"])
        if score > best_score:
            best_score = score
            best_data = entry["data_json"]

    if best_score >= L2_THRESHOLD:
        logger.debug("L2 语义命中 (score=%.3f): %s", best_score, query[:30])
        return best_data, best_score

    return None, best_score


# ──────────────────────────────────────────────
# L3: BGE embedding 深度语义 (需要模型)
# ──────────────────────────────────────────────

_embedding_model = None
_embedding_lock = threading.Lock()


def _get_bge_model():
    """延迟获取 BGE 模型 (复用 HybridRetriever 单例)"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                try:
                    from src.retrieval.retriever import HybridRetriever
                    _embedding_model = HybridRetriever().model
                except Exception as e:
                    logger.debug("BGE 模型获取失败: %s", e)
                    return None
    return _embedding_model


def compute_embedding(query: str) -> Optional[list[float]]:
    """计算 BGE embedding (带超时保护)"""
    try:
        model = _get_bge_model()
        if model is None:
            return None
        emb = model.encode([query], normalize_embeddings=True)
        return emb[0].tolist()
    except Exception as e:
        logger.debug("BGE embedding 计算失败: %s", e)
        return None


def _l3_search(query: str) -> tuple[Optional[str], float]:
    """
    L3 深度语义搜索: BGE embedding 余弦相似度
    仅在 L2 未命中时调用
    """
    emb = compute_embedding(query)
    if emb is None:
        return None, 0.0

    # 从 SQLite 加载所有 embedding 进行比较
    try:
        conn = _get_db()
        cutoff = time.time() - CACHE_TTL
        rows = conn.execute(
            "SELECT data, embedding FROM semantic_cache "
            "WHERE embedding IS NOT NULL AND created_at > ?",
            (cutoff,)
        ).fetchall()
        conn.close()

        import numpy as np
        qv = np.array(emb, dtype=np.float32)
        best_score, best_data = 0.0, None

        for data_json, emb_json in rows:
            if not emb_json:
                continue
            cv = np.array(json.loads(emb_json), dtype=np.float32)
            dot = float(np.dot(qv, cv))
            norm = float(np.linalg.norm(qv) * np.linalg.norm(cv))
            score = dot / norm if norm > 0 else 0.0
            if score > best_score:
                best_score = score
                best_data = data_json

        if best_score >= L3_THRESHOLD:
            return best_data, best_score
        return None, best_score
    except Exception as e:
        logger.debug("L3 搜索失败: %s", e)
        return None, 0.0


# ──────────────────────────────────────────────
# 归一化工具
# ──────────────────────────────────────────────

def _normalize_query(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r'[^\w一-鿿]', '', q)
    return q


def _exact_key(query: str) -> str:
    raw = _normalize_query(query).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# ──────────────────────────────────────────────
# 知识库版本守卫
# ──────────────────────────────────────────────
# 版本号由 scripts/sync_kb.py 写进 data/.kb_manifest.json。
# 没有这道守卫时, 知识库更新后旧答案会在 TTL(1h) 内继续被返回 —— 内容已经改了,
# 但缓存没有依据知道该作废。

_MANIFEST_PATH = os.path.join(CACHE_DB_DIR, ".kb_manifest.json")
_kb_guard = {"mtime": 0.0, "file_version": None, "seen": None}


def _current_kb_version() -> str:
    """读知识库版本 (按 mtime 缓存, 每次 lookup 只多一次 stat)"""
    try:
        mt = os.path.getmtime(_MANIFEST_PATH)
    except OSError:
        return "unknown"
    if mt != _kb_guard["mtime"]:
        try:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
                _kb_guard["file_version"] = json.load(f).get("version", "unknown")
        except Exception as e:
            logger.debug("知识库版本读取失败: %s", e)
            _kb_guard["file_version"] = "unknown"
        _kb_guard["mtime"] = mt
    return _kb_guard["file_version"] or "unknown"


def _invalidate_on_kb_change() -> None:
    """知识库版本变了 → 作废全部回答缓存

    ponytail: 粗粒度全清。知识库一变, 所有缓存答案都不可信, 全清最简单也最安全;
    若将来缓存热到全清代价明显, 再按受影响的 chunk 做精确失效。
    """
    version = _current_kb_version()
    if _kb_guard["seen"] is None:      # 进程内首次: 只记基线, 不清
        _kb_guard["seen"] = version
        return
    if version != _kb_guard["seen"]:
        logger.warning("知识库版本变化 %s → %s, 作废全部回答缓存",
                       _kb_guard["seen"], version)
        _kb_guard["seen"] = version
        invalidate(None)


# ──────────────────────────────────────────────
# 主接口
# ──────────────────────────────────────────────

def get_cached(query: str) -> Optional[dict]:
    """
    获取缓存 (L1 → L2 → L3)

    Returns:
        dict 结果, 或 None (未命中)
    """
    if not query or not query.strip():
        return None

    # 知识库版本守卫: 知识库更新过就作废旧答案 (在取锁之前调用, invalidate 内部要加锁)
    _invalidate_on_kb_change()

    # 隐私守卫: 用户私有/资金动账类不读共享缓存 (防跨用户串答)
    from src.cache.privacy import is_private_or_money_query
    if is_private_or_money_query(query):
        return None

    md5_key = _exact_key(query)
    now = time.time()

    # ── L1: 精确匹配 ──
    with _lock:
        entry = _l1_cache.get(md5_key)
        if entry:
            data_json, ts = entry
            if now - ts < CACHE_TTL:
                # LRU: 移到末尾 (最近使用)
                _l1_cache.move_to_end(md5_key)
                _stats["l1_hits"] += 1
                logger.debug("L1 精确命中: %s", query[:30])
                try:
                    return json.loads(data_json)
                except Exception as e:
                    logger.debug("L1 缓存 JSON 解析失败: %s", e)
            else:
                # 过期了
                del _l1_cache[md5_key]

    _stats["l1_misses"] += 1

    # ── L2: 轻量语义 (TF-IDF / n-gram) ──
    l2_start = time.time()
    data_json, score = _l2_search(query)
    l2_latency = (time.time() - l2_start) * 1000

    if data_json:
        _stats["l2_hits"] += 1
        logger.info("L2 语义命中 (score=%.3f, %.0fms): %s",
                     score, l2_latency, query[:30])
        # 提升到 L1
        try:
            data = json.loads(data_json)
            with _lock:
                _l1_cache[md5_key] = (data_json, now)
                _trim_l1()
            return data
        except Exception as e:
            logger.debug("L2 缓存 JSON 解析失败: %s", e)

    _stats["l2_misses"] += 1

    # ── L3: BGE 深度语义 (仅在 L2 未命中且置信度不太低时触发) ──
    # 如果 L2 得分离阈值很远 (<0.5), 跳过 L3 (不可能命中)
    if score < 0.5:
        _stats["l3_misses"] += 1
        return None

    l3_start = time.time()
    data_json, score = _l3_search(query)
    l3_latency = (time.time() - l3_start) * 1000

    if data_json:
        _stats["l3_hits"] += 1
        logger.info("L3 深度命中 (score=%.3f, %.0fms): %s",
                     score, l3_latency, query[:30])
        try:
            data = json.loads(data_json)
            with _lock:
                _l1_cache[md5_key] = (data_json, now)
                _trim_l1()
            return data
        except Exception as e:
            logger.debug("L3 缓存 JSON 解析失败: %s", e)

    _stats["l3_misses"] += 1
    return None


def _is_error_response(data: dict) -> bool:
    """判断响应是否为错误/降级/动态数据结果 (不应缓存, 否则会被长期命中)"""
    if not isinstance(data, dict):
        return False
    if data.get("_intent") == "error":
        return True
    # 动态业务数据 (订单/物流/退款/投诉): 实时信息, 禁止缓存
    if data.get("_intent") in ("order", "shipping", "refund", "complaint"):
        return True
    ans = str(data.get("answer", ""))
    if "分析引擎出错" in ans or ans.startswith("⚠️"):
        return True
    return False


def set_cache(query: str, data: dict, embedding: Optional[list] = None):
    """
    写入缓存 (L1 + L2 + SQLite)
    可选传入预计算的 embedding 避免重复计算
    """
    if not query or not data:
        return

    # 知识库版本守卫: 别把旧版本下算出来的答案写进缓存
    _invalidate_on_kb_change()

    # 隐私守卫: 用户私有/资金动账类不写共享缓存
    from src.cache.privacy import is_private_or_money_query
    if is_private_or_money_query(query):
        logger.info("跳过缓存: 用户私有/动账类 (query=%s)", query[:30])
        return

    # 不缓存错误/降级响应 (否则瞬时错误会被长期缓存命中, 如 Gemini 400 地区封锁)
    if _is_error_response(data):
        logger.info("跳过缓存: 错误/降级响应 (query=%s)", query[:30])
        return

    md5_key = _exact_key(query)
    data_json = json.dumps(data, ensure_ascii=False, default=str)
    now = time.time()

    # L1: 内存 LRU
    with _lock:
        _l1_cache[md5_key] = (data_json, now)
        _trim_l1()

    # L2: 语义索引
    with _lock:
        # 删除同 key 旧条目
        _l2_entries[:] = [e for e in _l2_entries if e["key"] != md5_key]
        _l2_entries.append({
            "key": md5_key,
            "query": query,
            "data_json": data_json,
            "ts": now,
        })
        global _l2_dirty
        _l2_dirty = True

    # L2 修剪 (只保留 MAX_ENTRIES 条)
    _trim_l2()

    # SQLite 持久化
    _save_to_db(md5_key, query, data_json, embedding=embedding)

    # 更新统计
    with _lock:
        _stats["total_entries"] = len(_l1_cache)


def _trim_l1():
    """LRU 淘汰: 超出 MAX_ENTRIES 时淘汰最早条目"""
    while len(_l1_cache) > MAX_ENTRIES:
        _l1_cache.popitem(last=False)
        _stats["evictions"] += 1


def _trim_l2():
    """L2 条目裁剪: 保留最新的 MAX_ENTRIES 条"""
    with _lock:
        if len(_l2_entries) > MAX_ENTRIES:
            _l2_entries.sort(key=lambda e: e["ts"], reverse=True)
            _l2_entries[:] = _l2_entries[:MAX_ENTRIES]
            _l2_dirty = True


def invalidate(query: str = None):
    """
    使缓存失效
    - query=None: 全量失效
    - query=str: 仅失效该查询相关缓存
    """
    if query is None:
        # 全量失效
        with _lock:
            _l1_cache.clear()
            _l2_entries.clear()
            _l2_dirty = False
        # 清 SQLite
        try:
            conn = _get_db()
            conn.execute("DELETE FROM semantic_cache")
            conn.commit()
            conn.close()
            logger.info("全量缓存已失效")
        except Exception as e:
            logger.warning("SQLite 清空失败: %s", e)
        return

    # 单条失效
    md5_key = _exact_key(query)
    with _lock:
        _l1_cache.pop(md5_key, None)
        _l2_entries[:] = [e for e in _l2_entries if e["key"] != md5_key]
        _l2_dirty = True
    try:
        conn = _get_db()
        conn.execute("DELETE FROM semantic_cache WHERE md5_key = ?", (md5_key,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("SQLite 单条删除失败: %s", e)


def get_stats() -> dict:
    """获取缓存命中统计"""
    with _lock:
        total_l1 = _stats["l1_hits"] + _stats["l1_misses"]
        total_l2 = _stats["l2_hits"] + _stats["l2_misses"]
        total_l3 = _stats["l3_hits"] + _stats["l3_misses"]
        total_all = _stats["l1_hits"] + _stats["l2_hits"] + _stats["l3_hits"]
        total_all += _stats["l1_misses"] + _stats["l2_misses"] + _stats["l3_misses"]

        return {
            "l1_hits": _stats["l1_hits"],
            "l1_misses": _stats["l1_misses"],
            "l2_hits": _stats["l2_hits"],
            "l2_misses": _stats["l2_misses"],
            "l3_hits": _stats["l3_hits"],
            "l3_misses": _stats["l3_misses"],
            "total_hits": _stats["l1_hits"] + _stats["l2_hits"] + _stats["l3_hits"],
            "total_misses": _stats["l1_misses"] + _stats["l2_misses"] + _stats["l3_misses"],
            "hit_ratio": round(total_all / total_l1 * 100, 1) if total_l1 > 0 else 0.0,
            "l2_hit_ratio": round(_stats["l2_hits"] / total_l2 * 100, 1) if total_l2 > 0 else 0.0,
            "l3_hit_ratio": round(_stats["l3_hits"] / total_l3 * 100, 1) if total_l3 > 0 else 0.0,
            "entries_l1": len(_l1_cache),
            "entries_l2": len(_l2_entries),
            "evictions": _stats["evictions"],
            "uptime_seconds": int(time.time() - _stats["started_at"]),
        }


def warmup_from_db():
    """启动预热: 从 SQLite 加载缓存到内存"""
    count = _load_from_db()
    if count:
        _rebuild_l2_index()
    _clean_expired_db()
    return count


# ── 模块初始化 ──────────────────────────────────
_init_db()
warmup_from_db()

logger.info("answer_cache v2.0 就绪 (SQLite=%s, L1=%d条, MAX=%d)",
             CACHE_DB_PATH, len(_l1_cache), MAX_ENTRIES)
