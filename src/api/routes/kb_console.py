"""
知识库管理台接口 —— /api/v1/kb/*

整组 admin-only, 与 /api/v1/knowledge 一致。
校验交给 kb_validate (纯函数), 读写交给 kb_settings, 本层只负责 HTTP 语义。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.routes.auth import require_roles

# 模块级 import: 测试靠 patch 这个名字模拟 worker 不在线, 放函数里 patch 不生效
from src.celery_app import is_celery_worker_running
from src.config import settings
from src.core import kb_settings
from src.core.kb_validate import ValidationError, validate
from src.infra.jsonl_io import read_lines

# 模块级 import: 测试靠 patch 这个名字把分块文件指到 tmp, 放函数里 patch 不生效
from src.infra.paths import CHUNKS_PROCESSED_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 后端强制鉴权: 整组挂守卫, 新增端点默认继承, 不靠逐个端点记得加
router = APIRouter(prefix="/api/v1/kb", tags=["知识库管理台"],
                   dependencies=[Depends(require_roles("admin"))])


class StrategyPut(BaseModel):
    section: str
    values: dict


@router.get("/strategy")
async def get_strategy():
    """当前生效值 + 代码默认值 + 哪些偏离了默认"""
    cfg, source = kb_settings.get_all()
    out = {"source": source,
           "updated_at": cfg.get("updated_at", ""),
           "updated_by": cfg.get("updated_by", "")}
    for section in ("chunking", "retrieval"):
        current = cfg[section]
        default = kb_settings.DEFAULTS[section]
        out[section] = {
            "current": current,
            "default": default,
            "changed": sorted(k for k, v in current.items() if default.get(k) != v),
        }
    return out


@router.put("/strategy")
async def put_strategy(body: StrategyPut, user=Depends(require_roles("admin"))):
    """保存某个配置段。校验失败返回 400 并指明具体字段与规则。"""
    try:
        values = validate(body.section, body.values)
    except ValidationError as e:
        # 前端直接读 detail.field / detail.rule 定位输入框, 故原样返回
        raise HTTPException(status_code=400, detail=e.as_dict())

    # 缺 key 时开重排会让**整轮检索**失败(重排器构造在精排的 try 之外, 那是有意的:
    # 运行时故障要响亮)。但这不该由管理页上勾一个框触发 —— 在保存这一刻就拦下来。
    # 这道校验看不见 env, 属于运行时依赖, 所以放路由层, 不塞进纯函数的 kb_validate。
    if body.section == "retrieval" and values.get("rerank_enabled") and not settings.dashscope_api_key:
        raise HTTPException(status_code=400, detail={
            "field": "rerank_enabled",
            "value": True,
            "rule": "缺少 DASHSCOPE_API_KEY，启用重排后检索会失败；请先在 .env 配好",
        })

    before, _ = kb_settings.get_all()
    cfg = kb_settings.save(body.section, values, user=user.get("user_id", "admin"))
    after = cfg[body.section]

    changed = sorted(k for k in values if before[body.section].get(k) != after[k])
    return {
        "saved": True,
        "changed": changed,
        # 切分参数只在下次上传时由 smart_chunk_pdf 使用, 重建也不影响已有内容
        "applies_to": "next_request" if body.section == "retrieval" else "next_upload",
        "config": after,
    }


@router.get("/chunks")
async def list_chunks(
    source: str = Query("", description="按来源文件精确筛选"),
    section: str = Query("", description="按小节标题子串筛选"),
    q: str = Query("", description="内容关键词子串"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),   # 上限 100, 不设等于允许一次拉全库
):
    """分块浏览 (走文件, 不走向量——这是浏览不是检索)

    翻到越界页或筛完没结果都只返回空列表, 不报错 —— 这个页面是排查
    "检索为什么没命中" 的入口, 报错会被当成知识库坏了。
    """
    rows = read_lines(CHUNKS_PROCESSED_PATH)
    if source:
        rows = [r for r in rows if r.get("source") == source]
    if section:
        rows = [r for r in rows if section in (r.get("section") or "")]
    if q:
        rows = [r for r in rows if q in (r.get("content") or "")]

    start = (page - 1) * size
    items = [
        {
            "chunk_id": r.get("chunk_id", ""),
            "source": r.get("source", ""),
            "section": r.get("section", ""),
            "page": r.get("page", 1),
            "content": r.get("content", ""),
            "content_hash": r.get("content_hash", ""),
            "length": len(r.get("content") or ""),
        }
        for r in rows[start:start + size]
    ]
    return {"total": len(rows), "page": page, "size": size, "items": items}


@router.post("/rebuild")
async def rebuild_kb(user=Depends(require_roles("admin"))):
    """手动触发全量重建（索引维护用，与切分参数无关）

    异步执行, 进度走已有的 GET /api/v1/tasks/{task_id} —— 不新增进度通道。
    """
    if not is_celery_worker_running():
        raise HTTPException(
            status_code=503,
            detail="ETL Worker 未运行，无法执行重建。请启动 Celery Worker 后重试。",
        )
    from src.tasks.kb_tasks import rebuild_kb_task
    task = rebuild_kb_task.delay()
    try:
        from src.database import db_manager
        db_manager.create_task(task.id, "__kb_rebuild__")
    except Exception as e:
        logger.error("task_logs 写入失败 (%s): %s", task.id, e)
    logger.info("知识库重建任务已提交: task_id=%s", task.id)
    return {"task_id": task.id, "status": "accepted"}


class RetrievalTestIn(BaseModel):
    query: str
    top_k: int = 5
    only_single_path: bool = False


@router.post("/retrieval-test")
async def retrieval_test(body: RetrievalTestIn):
    """跑一次检索并返回各通路明细 —— 用于排查"为什么这条没搜到" """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    # 越界值会原样透给 Milvus/BM25, 在入口拦下比让它变成 500 好定位
    if not (1 <= body.top_k <= 20):
        raise HTTPException(status_code=400, detail="返回条数必须在 1~20 之间")

    from src.retrieval.retriever import HybridRetriever
    try:
        # HybridRetriever 是全局单例(__new__ 接管), 构造无重复开销;
        # 但它是同步的且要打 Milvus/BM25, 必须挪出事件循环, 否则一次调试就卡住全服务
        out = await asyncio.to_thread(
            HybridRetriever().retrieve_debug, query, body.top_k
        )
    except Exception as e:
        logger.error("检索调试失败: %s", e, exc_info=True)
        raise HTTPException(status_code=504, detail=f"检索执行失败：{e}")

    single = [i for i in out["fused"] if len(i.get("found_by") or []) == 1]
    if body.only_single_path:
        out["fused"] = single
    out["single_path_count"] = len(single)
    # 稀疏路为空时 retrieve_debug 会静默退化成 dense[:top_k], found_by 全读成 ["dense"],
    # 会被误读成"关键词检索没命中"。这是唯一能看出二者的地方, 故从响应派生后透出
    out["sparse_empty"] = len(out.get("sparse") or []) == 0
    # graph.error (若图谱检索失败) 原样透传, 不吞 —— 调试台要显示失败原因
    return out


# ── 知识图谱三元组审核 (LLM抽取 → 人工审核 → 入库) ──────────

import hashlib
from pathlib import Path

PENDING_DIR = Path(PROJECT_ROOT) / "data" / "kg_pending"
# 审核通过写"源", 不写"产物": kg_triples.jsonl 由 build_finance_kb.render_kg_triples()
# 从 kg_domain_triples.jsonl + catalog 重新生成, 直接追加会被下次构建整个覆盖。
KG_TRIPLES_PATH = Path(PROJECT_ROOT) / "data" / "finance_kb" / "kg_domain_triples.jsonl"


def _triple_id(t: dict) -> str:
    """用 head+relation+tail 的 hash 作为三元组唯一标识"""
    raw = f"{t.get('head','')}|{t.get('relation','')}|{t.get('tail','')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _load_pending() -> list:
    """加载所有待审核三元组"""
    if not PENDING_DIR.exists():
        return []
    items = []
    for f in sorted(PENDING_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("status") == "pending":
                t["id"] = _triple_id(t)
                t["source_file"] = f.name
                items.append(t)
    return items


def _save_pending(items: list):
    """把更新后的待审核列表写回各文件"""
    by_file = {}
    for t in items:
        fname = t.pop("source_file", None) or "unknown.jsonl"
        by_file.setdefault(fname, []).append(t)
    for fname, triples in by_file.items():
        path = PENDING_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            for t in triples:
                t.pop("id", None)
                f.write(json.dumps(t, ensure_ascii=False) + "\n")


@router.get("/triples/pending")
async def get_pending_triples():
    """获取待审核的知识图谱三元组列表 (LLM抽取后待人工确认)"""
    items = _load_pending()
    return {
        "total": len(items),
        "items": [
            {
                "id": t["id"],
                "head": t.get("head", ""),
                "relation": t.get("relation", ""),
                "tail": t.get("tail", ""),
                "doc": t.get("doc", ""),
                "source_text": t.get("source_text", ""),
                "source_file": t.get("source_file", ""),
            }
            for t in items
        ],
    }


class TripleAuditIn(BaseModel):
    id: str
    action: str  # "approve" | "reject"


@router.post("/triples/audit")
async def audit_triple(body: TripleAuditIn, user=Depends(require_roles("admin"))):
    """审核三元组: 通过则写入 kg_domain_triples.jsonl (源) + Neo4j, 拒绝则标记 rejected

    交互键: /api/v1/kb/triples/audit
    body: {"id": "<三元组id>", "action": "approve"|"reject"}

    写入的是手写源文件 kg_domain_triples.jsonl, 由 build_finance_kb 派生出
    kg_triples.jsonl 再灌图谱; 直接写产物会被下次构建覆盖, 审核结果会丢。
    Neo4j 立即 MERGE 使改动当场生效, 但本地图谱降级层 (knowledge_graph/core.py
    读 kg_triples.jsonl) 要等 scripts/sync_kb.py 重建后才跟上。
    """
    if body.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action 只能是 approve 或 reject")

    items = _load_pending()
    target = next((t for t in items if t["id"] == body.id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"未找到三元组 id={body.id}")

    if body.action == "reject":
        target["status"] = "rejected"
        _save_pending(items)
        logger.info("三元组审核拒绝: %s → %s", body.id, target.get("head"))
        return {"status": "rejected", "id": body.id}

    # approve: 写入 kg_domain_triples.jsonl (去重) + Neo4j
    existing = set()
    if KG_TRIPLES_PATH.exists():
        for line in KG_TRIPLES_PATH.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                t = json.loads(line)
                existing.add((t.get("head"), t.get("relation"), t.get("tail")))

    key = (target["head"], target["relation"], target["tail"])
    if key in existing:
        target["status"] = "rejected"
        _save_pending(items)
        return {"status": "duplicate", "id": body.id, "message": "三元组已存在，自动跳过"}

    # 追加到源文件 (下次构建派生出 kg_triples.jsonl, 不会被覆盖)
    new_triple = {
        "head": target["head"],
        "relation": target["relation"],
        "tail": target["tail"],
        "doc": target.get("doc", ""),
        "note": target.get("source_text", "")[:100],
    }
    with open(KG_TRIPLES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(new_triple, ensure_ascii=False) + "\n")

    # 加载到 Neo4j (幂等 MERGE)
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        neo_user = os.getenv("NEO4J_USER", "neo4j")
        neo_pwd = os.getenv("NEO4J_PASSWORD", "kefu2026neo")
        driver = GraphDatabase.driver(uri, auth=(neo_user, neo_pwd))
        with driver.session() as s:
            s.run("CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
            s.run(
                "MERGE (a:Entity {name:$h}) MERGE (b:Entity {name:$t}) "
                "MERGE (a)-[r:REL {name:$rel}]->(b) SET r.doc=$doc, r.note=$note",
                h=target["head"], t=target["tail"], rel=target["relation"],
                doc=target.get("doc", ""), note=target.get("source_text", "")[:100],
            )
        driver.close()
        neo_status = "loaded"
    except Exception as e:
        logger.warning("Neo4j 写入失败 (已写入jsonl, 可手动重跑load_kg_to_neo4j): %s", str(e)[:80])
        neo_status = f"jsonl_only: {str(e)[:60]}"

    target["status"] = "approved"
    _save_pending(items)
    logger.info("三元组审核通过: %s → %s [%s] %s", body.id, target["head"], target["relation"], target["tail"])
    return {"status": "approved", "id": body.id, "neo4j": neo_status}
