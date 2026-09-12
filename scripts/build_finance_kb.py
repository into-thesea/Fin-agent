"""
构建金融理财产品知识库索引 (阶段一/二用)

从 data/finance_kb/*.md 按 '## ' 标题分节生成 chunk (跳过 _ 开头的文件, 如 catalog 由 jsonl 提供),
重建活动索引到 data/output_analysis/ (chunks_processed.jsonl + FAISS + BM25),
若 Milvus 可用则同时把同一批向量灌入 collection (kefu_chunks) — 保证 Milvus/FAISS 两路一致,
避免"向量走电商、稀疏走金融"的混搭。

用法:
    .venv/Scripts/python.exe scripts/build_finance_kb.py [--skip-milvus]
"""

import os
import re
import sys
import json
import glob
import hashlib
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_finance_kb")

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "finance_kb")


def _stable_chunk_id(stem: str, title: str, used: set) -> str:
    """稳定身份 ID — 由 (文档, 小节标题) 决定, 不随小节顺序变化

    原先用 kb_{stem}_{序号}: 中间插一节会让后面所有 chunk 的 id 整体位移,
    golden 引用 / 缓存键 / 跨版本追踪全部失准。改成标题哈希后, 插节不影响他人。

    注意: 身份稳定 ≠ 内容不变。内容是否变化看 content_hash, 两者分开存 ——
    只用内容哈希当 id 的话, 改一个字就变成"删一条+加一条", 丢失"这条被更新了"的语义。
    """
    base = f"kb_{stem}_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:10]}"
    if base in used:                      # 同文档内标题重复 → 加序号区分
        n = 2
        while f"{base}_{n}" in used:
            n += 1
        base = f"{base}_{n}"
    used.add(base)
    return base


def _content_hash(text: str) -> str:
    """内容指纹 (忽略空白差异) — 用于判重和判断"这块内容变了没有\""""
    return hashlib.sha256(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()


def parse_faq_markdown(md_path: str, source: str) -> list:
    """按 '## ' 标题切分为 section chunk (与电商版 parse_faq_markdown 契约一致)"""
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    sections = []
    current_title = None
    current_lines = []
    for line in lines:
        if line.startswith("## "):
            if current_title:
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
        elif line.strip() and current_title is not None:
            current_lines.append(line.strip())
    if current_title:
        sections.append((current_title, current_lines))

    stem = os.path.splitext(os.path.basename(md_path))[0]
    used = set()
    chunks = []
    for title, body_lines in sections:
        content = f"【{title}】\n" + "\n".join(body_lines)
        chunks.append({
            "chunk_id": _stable_chunk_id(stem, title, used),
            "source": source,
            "page": 1,
            "section": title,
            "content": content,
            "content_hash": _content_hash(content),
        })
    logger.info("解析 %s → %d chunk", os.path.basename(md_path), len(chunks))
    return chunks


CATALOG_PATH = os.path.join(KB_DIR, "catalog.jsonl")
KG_DOMAIN_PATH = os.path.join(KB_DIR, "kg_domain_triples.jsonl")
KG_TRIPLES_PATH = os.path.join(KB_DIR, "kg_triples.jsonl")

_RISK_LABEL = {"R1": "低风险", "R2": "中低风险", "R3": "中风险",
               "R4": "中高风险", "R5": "高风险"}

# catalog.type 前缀 → 图谱里的规范类型节点名 (与既有 is_type_of 的 tail 保持一致)
_TYPE_CANON = {
    "现金管理": "现金管理类理财",
    "固定收益": "固定收益理财",
    "固收增强": "固收增强理财",
    "私银混合": "私银混合理财",
    "存款": "存款",
    "结构性存款": "结构性存款",
    "公募基金": "公募基金",
    "保险": "保险产品",
    "债券": "债券",
    "信托": "信托",
    "贵金属": "贵金属",
}
_FORM_DESC = {
    "自营理财": "本行自营理财产品",
    "自营": "本行自营产品",
    "代销": "本行代销产品",
    "代销保险": "本行代销的保险产品",
    "存款": "本行发行的存款类产品",
}


def render_product_chunks() -> list:
    """从 catalog.jsonl 渲染产品文档 chunk

    产品只在 catalog.jsonl 里录一次。这里把结构化字段展开成「产品要素」小节,
    再接上该产品自己的 sections (话术/合规口径)。原先手写的 prod_pNNN.md 的
    「产品要素」节是这些字段的纯重复, 删掉后不再有漂移可能。

    source 仍写作 prod_pNNN.md: 这些文档现在是虚拟的, 但保留该标识可以让
    golden 的 evidence 维持在产品粒度 (否则只能退化成整个 catalog.jsonl)。
    """
    if not os.path.exists(CATALOG_PATH):
        logger.warning("catalog.jsonl 不存在, 跳过产品文档生成")
        return []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        products = [json.loads(line) for line in f if line.strip()]

    chunks = []
    for p in products:
        pid = p.get("id", "")
        if not pid:
            continue
        unit = p.get("min_amount_unit", "")
        amount = f"{p.get('min_amount', '')} {'万元' if unit == '万' else unit}".strip()
        elements = (
            f"{p.get('name', '')}（{pid}）为"
            f"{_FORM_DESC.get(p.get('form', ''), p.get('form', ''))}，"
            f"产品类型为{p.get('type', '')}，"
            f"风险等级 {p.get('risk', '')}（{_RISK_LABEL.get(p.get('risk', ''), '')}）。"
            f"起购金额 {amount}，期限与申赎安排：{p.get('term', '')}。"
            f"{p.get('elements_note', '')}"
        )
        rows = [("产品要素", elements)]
        rows += [(s.get("title", ""), s.get("body", "")) for s in p.get("sections", [])]

        used = set()
        for title, body in rows:
            content = f"【{title}】\n{body}"
            chunks.append({
                "chunk_id": _stable_chunk_id(f"prod_{pid.lower()}", title, used),
                "source": f"prod_{pid.lower()}.md",
                "page": 1,
                "section": title,
                "content": content,
                "content_hash": _content_hash(content),
            })
    logger.info("catalog.jsonl → %d 个产品 chunk (%d 个产品)", len(chunks), len(products))
    return chunks


def render_kg_triples() -> int:
    """生成 kg_triples.jsonl = 手写领域知识 + catalog 派生的产品三元组

    每个产品固定 3 条 (is_type_of / risk_level / deposit_insured), 全是 catalog
    字段的机械展开 —— 手写就会像 P012-P024 那样只加一半、图谱缺新产品。
    领域知识 (covered_by / not_covered_by / requires 这类多跳边) 仍需手写,
    放在 kg_domain_triples.jsonl。
    """
    if not os.path.exists(CATALOG_PATH):
        logger.warning("catalog.jsonl 不存在, 跳过三元组生成")
        return 0
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        products = [json.loads(line) for line in f if line.strip()]

    triples = []
    if os.path.exists(KG_DOMAIN_PATH):
        with open(KG_DOMAIN_PATH, "r", encoding="utf-8") as f:
            triples = [json.loads(line) for line in f if line.strip()]
    else:
        logger.warning("kg_domain_triples.jsonl 不存在, 只生成产品三元组")
    n_domain = len(triples)

    for p in products:
        pid = p.get("id", "")
        if not pid:
            continue
        doc = f"prod_{pid.lower()}.md"
        prefix = re.split(r"[（(]", p.get("type", ""))[0].strip()
        insured = bool(p.get("deposit_insured"))
        triples += [
            {"head": p.get("name", ""), "relation": "is_type_of",
             "tail": _TYPE_CANON.get(prefix, prefix), "doc": doc, "note": "产品类型"},
            {"head": p.get("name", ""), "relation": "risk_level",
             "tail": p.get("risk", ""), "doc": doc,
             "note": _RISK_LABEL.get(p.get("risk", ""), "")},
            {"head": p.get("name", ""), "relation": "deposit_insured",
             "tail": "是" if insured else "否", "doc": doc,
             "note": "受存款保险保障" if insured else "非存款, 不受存款保险保障"},
        ]

    with open(KG_TRIPLES_PATH, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("kg_triples.jsonl: 领域 %d + 产品派生 %d = %d 条",
                n_domain, len(triples) - n_domain, len(triples))
    return len(triples)


def collect_chunks() -> list:
    """收集 finance_kb 目录下所有 md 的 chunk (跳过 _ 开头文件) + catalog 派生的产品文档"""
    files = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    chunks = []
    for f in files:
        if os.path.basename(f).startswith("_"):
            continue
        chunks.extend(parse_faq_markdown(f, os.path.basename(f)))
    chunks.extend(render_product_chunks())
    if not chunks:
        logger.error("未收集到任何 finance_kb chunk, 终止")
        sys.exit(1)

    # 内容完全相同的 chunk 只留一份。
    # 典型场景: 同一类产品的通用口径 (如四只基金的「代销口径」逐字相同)。
    # 不去重的话检索会返回 4 条一模一样的结果, 挤占 top-k 且分数并列无法区分。
    seen, deduped, dropped = {}, [], []
    for c in chunks:
        h = c.get("content_hash")
        if h and h in seen:
            dropped.append((c["chunk_id"], seen[h]))
            continue
        if h:
            seen[h] = c["chunk_id"]
        deduped.append(c)
    if dropped:
        logger.info("去重: 丢弃 %d 条内容完全相同的 chunk", len(dropped))
        for cid, kept in dropped:
            logger.info("  %s 与 %s 内容相同, 只保留前者", cid, kept)

    logger.info("finance_kb 共 %d 个 chunk", len(deduped))
    return deduped


def rebuild_local(chunks: list) -> None:
    """写 chunks_processed.jsonl + 重建 FAISS + BM25"""
    from src.infra.paths import (
        CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR, BM25_INDEX_PATH,
    )
    os.makedirs(os.path.dirname(CHUNKS_PROCESSED_PATH), exist_ok=True)
    with open(CHUNKS_PROCESSED_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("chunks 写入: %s (%d)", CHUNKS_PROCESSED_PATH, len(chunks))

    from src.retrieval.vector_indexer import rebuild_full_index
    rebuild_full_index(CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR)

    from src.retrieval.bm25_index import rebuild_bm25_index
    rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)
    logger.info("本地 FAISS+BM25 重建完成")


def sync_milvus() -> None:
    """若 Milvus 可用, 从 FAISS 索引 reconstruct 向量后重灌 collection (保持两路一致)"""
    import faiss
    from src.infra.paths import FAISS_INDEX_DIR
    from src.vectorstore.milvus_manager import MilvusManager

    mgr = MilvusManager()
    if not mgr.available:
        logger.info("Milvus 不可用, 跳过 (系统将走 FAISS 向量后端)")
        return

    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    meta_path = os.path.join(FAISS_INDEX_DIR, "metadata.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        logger.error("FAISS 索引文件缺失, 无法同步 Milvus")
        return
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    vectors, metas = [], []
    for i, meta in enumerate(metadata):
        if not meta.get("content"):
            continue
        vectors.append(index.reconstruct(i).reshape(1, -1))
        metas.append({
            "content": meta.get("content", ""),
            "source": meta.get("source", ""),
            "document_id": meta.get("document_id", ""),
            "page": int(meta.get("page", 1)),
            "chunk_id": meta.get("chunk_id", ""),
            "section": meta.get("section", ""),
        })
    if not vectors:
        logger.error("无有效向量可同步 Milvus")
        return
    mgr.reset()
    mgr.add_embeddings(np.vstack(vectors).astype("float32"), metas)
    logger.info("Milvus 已重灌 %d 条 (collection=%s)", mgr.total_count, mgr._collection_name)


def build_all(skip_milvus: bool = False) -> int:
    """完整构建: chunks → FAISS/BM25 → Milvus → 知识图谱三元组

    sync_kb.py 也走这个入口, 保证流水线只有一处定义 (漏了图谱那次就是因为
    步骤散在各处、靠人记)。返回 chunk 数。
    """
    chunks = collect_chunks()
    rebuild_local(chunks)
    render_kg_triples()
    if not skip_milvus:
        sync_milvus()
    logger.info("金融知识库构建完成")
    return len(chunks)


def main():
    build_all(skip_milvus="--skip-milvus" in sys.argv)


if __name__ == "__main__":
    main()
