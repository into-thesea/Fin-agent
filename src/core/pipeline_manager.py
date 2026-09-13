"""
Fin-Agent ETL 流水线管理器 (企业版)

核心改进:
  1. 内容寻址去重 (SHA256) — 替代原有的文件名去重
  2. 版本追踪 — 同一文件不同内容自动创建新版本
  3. 原子性 — 每步失败不污染下一步
  4. 数据库记录 — 全量操作记录在元数据库
  5. 异步兼容 — 函数签名已适配 Celery 任务

用法:
  from src.core.pipeline_manager import ingest_document
  result = ingest_document("/path/to/report.pdf")
"""

import os
import sys
import time
import logging
from typing import Optional

from src.core.smart_chunk import smart_chunk_pdf as extract_chunks_from_pdf
from src.retrieval.vector_indexer import add_chunks_to_index
from src.database import db_manager
from src.cache.redis_client import RedisCache

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.infra.paths import DATA_REPORTS_DIR


def _append_chunks_to_store(chunks: list) -> None:
    """把分块追加到全量分块文件 (BM25 重建与全量重建的数据源)

    两条 ETL 路径共用这一处, 避免出现「向量索引里有、BM25 里没有」的半截状态
    —— 那会让新上传的文档在混合检索里只能被稠密路召回。
    """
    import json
    from src.infra.paths import CHUNKS_PROCESSED_PATH
    os.makedirs(os.path.dirname(CHUNKS_PROCESSED_PATH), exist_ok=True)
    with open(CHUNKS_PROCESSED_PATH, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ingest_document(
    pdf_path: str,
    source_label: str = "upload",
    version_tag: str = "",
    enable_cache_invalidation: bool = True,
    task_id_for_progress: Optional[str] = None,  # NEW: 同步模式进度回写
) -> dict:
    """
    全自动 ETL 流水线 (内容寻址 + 版本管理)

    流程:
      1. 复制文件到 data_reports/
      2. 计算 SHA256 哈希
      3. 数据库去重检查 (content_hash)
      4. PDF 解析 → 向量化 → 实体提取 → 图谱注入
      5. 记录元数据库 + 失效缓存

    Args:
        pdf_path: 源 PDF 文件路径
        source_label: 来源标签 ("upload" | "scraper" | "manual")
        version_tag: 版本标签 (默认自动生成)
        enable_cache_invalidation: 是否在完成后失效缓存

    Returns:
        {"status": "success"|"skipped"|"failed", "doc_id": int, ...}
    """
    start_time = time.time()
    file_name = os.path.basename(pdf_path)
    target_path = os.path.join(DATA_REPORTS_DIR, file_name)

    # 同步模式进度上报器
    _progress_cache = None
    if task_id_for_progress:
        _progress_cache = RedisCache()
    def _report(stage: str, step: int, total: int = 4):
        if _progress_cache and task_id_for_progress:
            _progress_cache.set_task_status(
                task_id=task_id_for_progress,
                status="processing",
                progress=min(int(step / total * 100), 99),
                stage=stage,
            )

    # 自动生成版本标签
    if not version_tag:
        version_tag = f"etl_{time.strftime('%Y%m%d_%H%M%S')}"

    # ─── Step 0: 内容寻址去重 ───────────────────
    content_hash = db_manager.compute_file_hash(pdf_path)

    existing = db_manager.find_document_by_hash(content_hash)
    if existing:
        logger.info(
            f"文件 {file_name} 内容已存在 (hash: {content_hash[:12]}...), "
            f"跳过入库 (现有 ID: {existing['id']}, v{existing['version']})"
        )
        return {
            "status": "skipped",
            "message": f"Document content already exists (ID: {existing['id']})",
            "doc_id": existing["id"],
            "content_hash": content_hash,
        }

    # ─── Step 0.5: 确保文件在 data_reports 中 ────
    if os.path.normpath(pdf_path) != os.path.normpath(target_path):
        import shutil
        os.makedirs(DATA_REPORTS_DIR, exist_ok=True)
        shutil.copy2(pdf_path, target_path)
        logger.info(f"文件已持久化: {target_path}")

    # ─── Step 0.6: 占位登记 (幂等的关键) ─────────
    # 必须在写任何索引之前登记, 不能等到流水线末尾。
    # 原先 DB 记录最后才写, 于是两条并发的流水线(同批次重复文件 / Celery 重投递)
    # 都能通过上面的去重检查, 各自往 jsonl 和 Milvus 写一份向量, 最后 add_document
    # 只留一行 —— 数据库看着正常, 索引里却是双份, 且这些多余向量再也没有东西
    # 能把它们和文档对应起来(删除时清不掉)。
    _report("dedup", 0)
    doc_id = db_manager.add_document(
        filename=file_name,
        status="processing",
        count=0,
        content_hash=content_hash,
        file_size=os.path.getsize(target_path) if os.path.exists(target_path) else 0,
        source_label=source_label,
    )

    # ─── 流水线 ─────────────────────────────
    steps = []

    try:
        # Step 1: PDF 解析 (章节感知分块)
        logger.info("Step 1/4: PDF 解析...")
        _report("parsing", 1)
        # 切分参数来自可编辑配置(data/kb_settings.json); 只影响本次上传,
        # 不回溯已入库内容 —— 内置知识库走 build_finance_kb.parse_faq_markdown,
        # 按 '## ' 标题切分, 不读这两个参数。
        from src.core.kb_settings import get_chunking
        _ck = get_chunking()
        chunks = extract_chunks_from_pdf(
            target_path, chunk_size=_ck["chunk_size"], overlap=_ck["overlap"],
        )
        if not chunks:
            db_manager.update_document_status(doc_id, "failed")
            return {"status": "failed", "error": "No content extracted from PDF."}
        steps.append(f"解析: {len(chunks)} 分块")

        # Step 2: 向量索引 (追加分块 → FAISS + Milvus → 重建 BM25)
        logger.info("Step 2/4: 向量索引...")
        _report("vectorizing", 2)
        _append_chunks_to_store(chunks)
        steps.append(f"分块入库: {len(chunks)} 条")

        added = add_chunks_to_index(chunks, document_id=file_name)
        steps.append(f"向量化: {added} 条 (FAISS + Milvus)")

        # BM25 重建: 原先吞异常跳过 → 会让新文档进不了稀疏路(检索不到),
        # 而混合检索的稀疏路一旦缺内容, 检索结果就会静默变差。
        from src.infra.paths import CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH
        from src.retrieval.bm25_index import rebuild_bm25_index
        rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)
        steps.append("BM25 索引已重建")

        # ─── 收尾: 占位记录转为完成 ─────────
        _report("db_record", 3)
        db_manager.update_document_status(doc_id, "done", count=len(chunks))

        # ─── 缓存失效 ───────────────────────
        _report("cache_invalidation", 4)
        if enable_cache_invalidation:
            cache = RedisCache()
            cache.invalidate_document_cache(file_name)
            logger.info("知识库缓存已失效")

        duration = round(time.time() - start_time, 2)
        logger.info(f"文档 {file_name} 入库完成! 耗时: {duration}s | 步骤: {' → '.join(steps)}")

        return {
            "status": "success",
            "doc_id": doc_id,
            "steps": steps,
            "duration": duration,
            "content_hash": content_hash,
            "version_tag": version_tag,
        }

    except Exception as e:
        logger.error(f"ETL 流水线崩溃 ({file_name}): {e}", exc_info=True)
        # 占位记录已存在, 标记失败即可 —— 不再新增一行(否则失败会留下重复记录)
        db_manager.update_document_status(doc_id, "failed")
        return {"status": "failed", "error": str(e), "steps": steps}


# === 兼容旧接口 ===
def ingest_document_legacy(pdf_path: str) -> dict:
    """旧版接口 (仅文件名去重) — 保留兼容"""
    file_name = os.path.basename(pdf_path)
    target_path = os.path.join(PROJECT_ROOT, "data_reports", file_name)

    if os.path.exists(target_path):
        return {"status": "skipped", "message": f"File {file_name} already exists."}

    return ingest_document(pdf_path)


if __name__ == "__main__":
    # CLI 测试
    if len(sys.argv) > 1:
        result = ingest_document(sys.argv[1])
        print(f"\n结果: {result['status']}")
        if result.get("steps"):
            print(f"步骤: {' → '.join(result['steps'])}")
    else:
        print("用法: python src/pipeline_manager.py <pdf_path>")
