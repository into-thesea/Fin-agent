"""
知识库全量重建 Celery 任务

定位是**索引维护**: 怀疑索引与分块文件不一致时的手动全量重建。
与切分参数无关 —— 重建不会把新的切分参数应用到已有内容(见设计文档 3.3)。
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
