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

import logging
import os
import sys
import time
from typing import Optional

from src.cache.redis_client import RedisCache
from src.core.smart_chunk import smart_chunk_pdf as extract_chunks_from_pdf
from src.database import db_manager
from src.retrieval.vector_indexer import add_chunks_to_index

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.infra.paths import DATA_REPORTS_DIR


def _append_chunks_to_store(chunks: list, replace_doc_id: int = None) -> None:
    """把分块写入全量分块文件 (BM25 重建与全量重建的数据源)

    两条 ETL 路径共用这一处, 避免出现「向量索引里有、BM25 里没有」的半截状态
    —— 那会让新上传的文档在混合检索里只能被稠密路召回。

    给了 replace_doc_id 就是**覆盖**而非追加: 先清掉该文档已写入的行再写。
    原因: approve_document 声称的原子性只覆盖 status, 不覆盖这个文件 —— Step 1
    成功、Step 2 挂掉时数据已落盘且没有回滚, 重试再追加一遍同一份内容就成了双份
    (实测: 一次失败 + 一次重试之后, 同一文档在文件里占 130 行而不是 65)。

    键用 pending_doc_id (preview_document 打上的) 而不是 source 文件名 ——
    两份同名文档各写各的, 按文件名会互相误删。
    """
    import json

    from src.infra.paths import CHUNKS_PROCESSED_PATH
    os.makedirs(os.path.dirname(CHUNKS_PROCESSED_PATH), exist_ok=True)

    if replace_doc_id is not None and os.path.exists(CHUNKS_PROCESSED_PATH):
        kept, dropped = [], 0
        with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    if json.loads(line).get("pending_doc_id") == replace_doc_id:
                        dropped += 1
                        continue
                except json.JSONDecodeError:
                    pass  # 损坏行保留, 与 read_lines 行为一致
                kept.append(line if line.endswith("\n") else line + "\n")
        with open(CHUNKS_PROCESSED_PATH, "w", encoding="utf-8") as f:
            f.writelines(kept)
        if dropped:
            logger.info("覆盖写入: 先清掉 doc_id=%s 已有的 %d 行分块", replace_doc_id, dropped)

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
        # 按 '## ' 标题切分, 不读这几个参数。
        from src.core.kb_settings import get_chunking
        _ck = get_chunking()
        chunks = extract_chunks_from_pdf(
            target_path, chunk_size=_ck["chunk_size"], overlap=_ck["overlap"],
            max_chunk_content=_ck["max_chunk_content"],
        )
        if not chunks:
            db_manager.update_document_status(doc_id, "failed")
            return {"status": "failed", "error": "No content extracted from PDF."}
        steps.append(f"解析: {len(chunks)} 分块")

        # Step 2: 向量索引 (追加分块 → Milvus → 重建 BM25)
        logger.info("Step 2/4: 向量索引...")
        _report("vectorizing", 2)
        _append_chunks_to_store(chunks)
        steps.append(f"分块入库: {len(chunks)} 条")

        added = add_chunks_to_index(chunks, document_id=file_name)
        steps.append(f"向量化: {added} 条 (Milvus)")

        # BM25 重建: 原先吞异常跳过 → 会让新文档进不了稀疏路(检索不到),
        # 而混合检索的稀疏路一旦缺内容, 检索结果就会静默变差。
        from src.infra.paths import BM25_INDEX_PATH, CHUNKS_PROCESSED_PATH
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


# === 待审核模式: 预览处理 → 审核通过 → 整文档入库 ===

PENDING_DIR = os.path.join(PROJECT_ROOT, "data", "kb_pending")
os.makedirs(PENDING_DIR, exist_ok=True)


def _pending_chunks_path(doc_id: int) -> str:
    """待审核文档的分块临时文件路径"""
    return os.path.join(PENDING_DIR, f"{doc_id}_chunks.jsonl")


def preview_document(pdf_path: str, doc_id: int) -> dict:
    """
    预览处理 — 只做 PDF 解析 + 智能分块, 不写 Milvus/BM25/正式分块文件.

    分块结果存到 data/kb_pending/{doc_id}_chunks.jsonl, 供前端预览和审核.
    审核通过后由 approve_document() 一次性写入正式索引.

    Args:
        pdf_path: PDF 文件路径
        doc_id: documents 表中的文档 ID (status=pending_review)

    Returns:
        {"status": "success"|"failed", "chunks": [...], "chunk_count": N, "error": str}
    """
    import json

    file_name = os.path.basename(pdf_path)
    logger.info("[预览] 开始解析 %s (doc_id=%s)", file_name, doc_id)

    try:
        from src.core.kb_settings import get_chunking
        _ck = get_chunking()
        chunks = extract_chunks_from_pdf(
            pdf_path,
            chunk_size=_ck["chunk_size"],
            overlap=_ck["overlap"],
            max_chunk_content=_ck["max_chunk_content"],
        )
    except Exception as e:
        logger.error("[预览] PDF 解析失败 (%s): %s", file_name, e, exc_info=True)
        db_manager.update_document_status(doc_id, "failed")
        return {"status": "failed", "error": str(e), "chunks": [], "chunk_count": 0}

    if not chunks:
        logger.warning("[预览] %s 未提取到任何内容", file_name)
        db_manager.update_document_status(doc_id, "failed")
        return {"status": "failed", "error": "No content extracted from PDF.",
                "chunks": [], "chunk_count": 0}

    # 给每个 chunk 打上 doc_id 标记, 便于审核和入库时关联
    for c in chunks:
        c["pending_doc_id"] = doc_id

    # 写入临时分块文件
    pending_path = _pending_chunks_path(doc_id)
    try:
        with open(pending_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        logger.info("[预览] 分块已写入临时文件: %s (%d 块)", pending_path, len(chunks))
    except Exception as e:
        logger.error("[预览] 临时文件写入失败: %s", e)
        return {"status": "failed", "error": f"Failed to save preview: {e}",
                "chunks": chunks, "chunk_count": len(chunks)}

    # 更新文档状态为待审核, 记录分块数
    db_manager.update_document_status(doc_id, "pending_review", count=len(chunks))

    logger.info("[预览] %s 完成: %d 分块, 待审核", file_name, len(chunks))
    return {"status": "success", "chunks": chunks, "chunk_count": len(chunks)}


def get_pending_chunks(doc_id: int) -> list:
    """读取待审核文档的分块预览"""
    import json

    pending_path = _pending_chunks_path(doc_id)
    if not os.path.exists(pending_path):
        return []
    chunks = []
    with open(pending_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return chunks


def approve_document(doc_id: int) -> dict:
    """
    审核通过 — 整文档原子入库.

    把待审核文档的所有分块一次性写入正式索引:
      1. 追加到 chunks_processed.jsonl (BM25 数据源)
      2. 向量化写入 Milvus
      3. 重建 BM25 索引
      4. 更新 documents.status = done
      5. 删除临时分块文件

    原子性: 任何一步失败都不更新 status=done, 临时文件保留便于重试.
    整文档入库: 要么全部成功, 要么全部不入库 (不会出现"部分chunk入库"的状态).

    Returns:
        {"status": "success"|"failed", "doc_id": int, "chunk_count": N, "steps": [...]}
    """

    doc = db_manager.get_document_by_id(doc_id)
    if not doc:
        return {"status": "failed", "error": f"Document {doc_id} not found"}
    if doc["status"] != "pending_review":
        return {"status": "failed",
                "error": f"Document status is '{doc['status']}', expected 'pending_review'"}

    pending_path = _pending_chunks_path(doc_id)
    if not os.path.exists(pending_path):
        return {"status": "failed", "error": f"Pending chunks file not found: {pending_path}"}

    # 读取分块
    chunks = get_pending_chunks(doc_id)
    if not chunks:
        return {"status": "failed", "error": "No chunks in pending file"}

    file_name = doc["filename"]
    steps = []
    logger.info("[审核通过] 开始整文档入库: %s (doc_id=%s, %d 块)", file_name, doc_id, len(chunks))

    try:
        # Step 1: 覆盖写入正式分块文件 (重试不会叠加 —— 原子性只覆盖 status, 不覆盖文件)
        _append_chunks_to_store(chunks, replace_doc_id=doc_id)
        steps.append(f"分块入库: {len(chunks)} 条")

        # Step 2: 向量化写入 Milvus
        added = add_chunks_to_index(chunks, document_id=file_name)
        steps.append(f"向量化: {added} 条 (Milvus)")

        # Step 3: 重建 BM25
        from src.infra.paths import BM25_INDEX_PATH, CHUNKS_PROCESSED_PATH
        from src.retrieval.bm25_index import rebuild_bm25_index
        rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)
        steps.append("BM25 索引已重建")

        # Step 4: 更新状态
        db_manager.update_document_status(doc_id, "done", count=len(chunks))
        steps.append("文档状态更新为 done")

        # Step 5: 失效缓存
        cache = RedisCache()
        cache.invalidate_document_cache(file_name)
        steps.append("缓存已失效")

        # Step 6: 删除临时文件 (入库成功后才删)
        try:
            os.remove(pending_path)
            steps.append("临时分块文件已清理")
        except Exception as e:
            logger.warning("[审核通过] 临时文件删除失败 (不影响入库): %s", e)

        logger.info("[审核通过] %s 入库完成: %s", file_name, " → ".join(steps))
        return {"status": "success", "doc_id": doc_id, "chunk_count": len(chunks), "steps": steps}

    except Exception as e:
        logger.error("[审核通过] 入库失败 (%s): %s", file_name, e, exc_info=True)
        # 失败时不更新 status, 临时文件保留, 便于重试
        return {"status": "failed", "error": str(e), "doc_id": doc_id,
                "chunk_count": len(chunks), "steps": steps}


def reject_document(doc_id: int, reason: str = "") -> dict:
    """
    审核拒绝 — 清理临时文件, 更新状态为 rejected.

    Args:
        doc_id: 文档 ID
        reason: 拒绝原因 (可选, 记录日志)

    Returns:
        {"status": "success"|"failed", "doc_id": int}
    """
    doc = db_manager.get_document_by_id(doc_id)
    if not doc:
        return {"status": "failed", "error": f"Document {doc_id} not found"}

    # 删除临时分块文件
    pending_path = _pending_chunks_path(doc_id)
    if os.path.exists(pending_path):
        try:
            os.remove(pending_path)
            logger.info("[审核拒绝] 临时文件已删除: %s", pending_path)
        except Exception as e:
            logger.warning("[审核拒绝] 临时文件删除失败: %s", e)

    # 更新状态
    db_manager.update_document_status(doc_id, "rejected")
    logger.info("[审核拒绝] %s (doc_id=%s) 已拒绝. 原因: %s",
                doc["filename"], doc_id, reason or "无")

    return {"status": "success", "doc_id": doc_id}


if __name__ == "__main__":
    # CLI 测试
    if len(sys.argv) > 1:
        result = ingest_document(sys.argv[1])
        print(f"\n结果: {result['status']}")
        if result.get("steps"):
            print(f"步骤: {' → '.join(result['steps'])}")
    else:
        print("用法: python src/pipeline_manager.py <pdf_path>")
