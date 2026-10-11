"""
知识库全量重建 Celery 任务

定位是**索引维护**: 怀疑索引与分块文件不一致时的手动全量重建。
与切分参数无关 —— 重建不会把新的切分参数应用到已有内容(见设计文档 3.3)。
重建只**刷新内置知识库**; 上传文档的分块在 build_finance_kb.rebuild_local 里
被原样保留(不丢, 也不重切), 见该函数注释。
"""
import logging
import time

from celery import Task

from src.celery_app import celery_app
from src.tasks.etl_tasks import ProgressReporter

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="kb_rebuild")
def rebuild_kb_task(self: Task):
    """从 data/finance_kb/*.md 全量重建 chunks / BM25 / Milvus / 图谱"""
    reporter = ProgressReporter(self, total_steps=3)
    start = time.time()
    try:
        from scripts.build_finance_kb import build_all
        reporter.report("rebuild", step=1)
        n = build_all()                     # collect → jsonl+BM25 → 三元组 → Milvus
        reporter.report("graph", step=2)
        duration = round(time.time() - start, 2)
        reporter.complete(doc_id=0, duration=duration)
        logger.info("[KB重建] 完成: %d 个分块, 耗时 %.1fs", n, duration)
        return {"status": "success", "chunks": n, "duration": duration}
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[KB重建] 失败: %s", msg, exc_info=True)
        reporter.fail(msg)
        raise


@celery_app.task(bind=True, name="extract_triples")
def extract_triples_task(self: Task, doc_id: int):
    """上传文档审核通过后, 后台抽取三元组进待审核队列。

    图谱里的关系边一直靠手写: 新文档入库后没有任何机制把它正文里的关系抽出来,
    于是每来一份文档都得人肉补图谱。这个任务把那段接上 ——
    入库 → 抽取 → 人工审核 → 通过后写 kg_domain_triples.jsonl(源) → 收敛进 Neo4j。

    挂在审核通过**之后**而不是预览阶段: 审核不通过就白抽了, 一份 PDF 是几百次
    LLM 调用。分块从正式分块文件按 source 取 —— 此时预览期的临时文件已被
    approve_document 删掉。
    """
    from src.database import db_manager
    from src.infra.jsonl_io import read_lines
    from src.infra.paths import CHUNKS_PROCESSED_PATH

    reporter = ProgressReporter(self, total_steps=1)
    doc = db_manager.get_document_by_id(doc_id)
    if not doc:
        msg = f"文档 {doc_id} 不存在"
        logger.error("[三元组抽取] %s", msg)
        reporter.fail(msg)
        return {"status": "failed", "error": msg}

    filename = doc["filename"]
    chunks = [c for c in read_lines(CHUNKS_PROCESSED_PATH) if c.get("source") == filename]
    if not chunks:
        msg = f"分块文件里没有 {filename} 的内容, 无法抽取"
        logger.warning("[三元组抽取] %s", msg)
        reporter.fail(msg)
        return {"status": "failed", "error": msg}

    try:
        from scripts.extract_triples import extract_chunks_to_pending
        stats = extract_chunks_to_pending(chunks, filename)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[三元组抽取] 失败 (%s): %s", filename, msg, exc_info=True)
        reporter.fail(msg)
        raise

    reporter.complete(doc_id=doc_id, duration=0)
    logger.info("[三元组抽取] %s: %d 个父块 → 抽取 %d 条, 去掉已知 %d 条, 待审 %d 条",
                filename, stats["chunks"], stats["extracted"],
                stats["skipped_known"], stats["queued"])
    return {"status": "success", "doc_id": doc_id, **stats}
