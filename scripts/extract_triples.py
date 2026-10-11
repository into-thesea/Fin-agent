"""
三元组 LLM 自动抽取脚本 — 从知识库文档中抽取 (head, relation, tail) 三元组

流程:
  1. 读取 data/finance_kb/ 下的 .md/.txt 文档
  2. 按 chunk 分块 (每块 ~800 字)
  3. 用 LLM 从每块中抽取三元组
  4. 输出到 data/kg_pending/ 待审核 (审核通过后才入库)

用法:
  python scripts/extract_triples.py                    # 抽取所有文档
  python scripts/extract_triples.py --doc deposit_insurance.md  # 只抽指定文档
  python scripts/extract_triples.py --limit 5          # 只抽前5个文档

输出:
  data/kg_pending/{doc_name}.jsonl — 每条一行, 含 head/relation/tail/doc/source_text/confidence
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def force_utf8_console() -> None:
    """Windows 控制台默认 GBK, 强制 UTF-8 输出避免 emoji 打印报错。

    必须整段包 try: Celery worker 里 sys.stdout 是 LoggingProxy, 没有 reconfigure,
    裸调用会让"import 本模块"这一步就炸 —— 任务连函数体都进不去, 只能看到
    `AttributeError: 'LoggingProxy' object has no attribute 'reconfigure'`。
    """
    try:
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


force_utf8_console()

KB_DIR = PROJECT_ROOT / "data" / "finance_kb"
PENDING_DIR = PROJECT_ROOT / "data" / "kg_pending"
# 图谱里"已经有什么"的现行快照。它是产物, 但每次构建都重写, 且包含了 catalog
# 派生的产品三元组 —— 拿它去重, 审核队列才不会塞满已知边。
KG_TRIPLES = KB_DIR / "kg_triples.jsonl"

# ponytail: 单文档抽取上限。父块一章一次 LLM 调用, 大 PDF 能有几百章, 而 Celery
# worker 是 --pool=solo 串行 —— 不限量会把上传/预览一起堵住。真遇到超长文档再改成配置项。
MAX_CHUNKS_PER_DOC = 120

# 兜底词汇表: 图谱快照读不到时用 (首次构建前 / 文件缺失)。
# 刻意不含 has_attribute / related_to —— 它们图谱一条都没用, 却是 LLM 最顺手的
# 关系名 (实测 16 条里 14 条是 has_attribute), 放进词表等于让审核员去拒一堆
# "产品 -has_attribute-> 2017年7月20日" 这种没有关系语义的字段。
FALLBACK_RELATIONS = {
    "is_type_of", "is_subtype_of", "risk_level", "deposit_insured",
    "covered_by", "not_covered_by", "can_buy", "requires",
    "has_fee", "redemption_rule", "compare_with", "has_period",
    "coverage_limit", "applies_to", "excludes", "sold_as",
}


# 图谱里有、但不邀请 LLM 去用的关系。这类是"逃生口": 语义泛到什么都能塞,
# LLM 会拿它把字段罗列一股脑倒出来, 而不是去找准确的那个关系。
#   has —— 图谱里就 1 条 (保险产品 -has-> 犹豫期15天, 本该是 has_period)。
#          实测邀请它之后, 51 条里 14 条是 has, 全是"投资范围包含股票/ETF/期权"这种没有
#          合适关系的罗列。摘掉后 LLM 要么用准确关系, 要么不抽。图谱里已有的边不受影响。
_EXCLUDE_FROM_EXTRACTION = {"has"}


def _load_allowed_relations() -> set:
    """从图谱现行快照派生允许的关系类型。

    原先这里写死 13 种, 注释还写着"与现有 kg_triples.jsonl 对齐" —— 实际图谱在用
    28 种, 两边对不上: 词表里 2 种图谱从不用, 图谱里 17 种抽不出来 (can_buy /
    has_fee / redemption_rule / compare_with / deposit_insured 这些主力边全在里头)。
    硬编码一份必然漂移, 所以直接从快照取。
    """
    try:
        rels = {json.loads(line)["relation"]
                for line in KG_TRIPLES.read_text(encoding="utf-8").splitlines() if line.strip()}
        rels -= _EXCLUDE_FROM_EXTRACTION
        return rels or set(FALLBACK_RELATIONS)
    except Exception as e:
        print(f"  ⚠️  读取 {KG_TRIPLES.name} 取关系词表失败, 用兜底表: {str(e)[:80]}")
        return set(FALLBACK_RELATIONS)


ALLOWED_RELATIONS = _load_allowed_relations()

EXTRACT_PROMPT = """你是金融知识图谱三元组抽取专家。从下面的文档片段中抽取 (主体, 关系, 客体) 三元组。

规则:
1. 只抽取明确陈述的事实，不要推断或概括
2. 主体和客体必须是文档中出现的具体名词（产品名、概念、制度）
3. 关系只能从以下列表选择: {relations}
4. 产品说明书里的字段罗列（登记编码、内部代码、成立日、币种、销售渠道这类）没有关系语义，
   不要抽。宁缺毋滥 —— 只抽能被列表里某个关系**准确**表达的事实，宁可少抽也不要硬套
5. 每个三元组附原文片段作为证据
6. 输出 JSON 数组，每个元素: {{"head": "", "relation": "", "tail": "", "evidence": ""}}
7. 不要输出解释，只输出 JSON

文档片段:
{text}

JSON:"""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """按字数分块"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_from_text(text: str, doc_name: str) -> list:
    """用 LLM 从文本中抽取三元组"""
    from src.llm.llm_client import create_client

    client = create_client(cheap=True)
    prompt = EXTRACT_PROMPT.format(
        relations=", ".join(sorted(ALLOWED_RELATIONS)),
        text=text[:1500],  # 限制长度
    )
    try:
        resp = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        # 解析 JSON
        resp = resp.strip()
        if resp.startswith("```"):
            resp = resp.split("```")[1]
            if resp.startswith("json"):
                resp = resp[4:]
        data = json.loads(resp)
        if isinstance(data, dict):
            data = [data]
        triples = []
        for item in data:
            rel = item.get("relation", "").strip()
            if rel not in ALLOWED_RELATIONS:
                continue
            head = item.get("head", "").strip()
            tail = item.get("tail", "").strip()
            if not head or not tail:
                continue
            triples.append({
                "head": head,
                "relation": rel,
                "tail": tail,
                "doc": doc_name,
                "source_text": item.get("evidence", "")[:200],
                "confidence": 0.0,  # 待人工审核
                "status": "pending",
            })
        return triples
    except Exception as e:
        print(f"  ⚠️  抽取失败: {str(e)[:80]}")
        return []


def existing_triple_keys() -> set:
    """图谱里已有的 (head, relation, tail)"""
    if not KG_TRIPLES.exists():
        return set()
    keys = set()
    try:
        for line in KG_TRIPLES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                keys.add((t.get("head"), t.get("relation"), t.get("tail")))
    except Exception as e:
        print(f"  ⚠️  读取 {KG_TRIPLES.name} 失败, 跳过去重: {str(e)[:80]}")
    return keys


def extract_chunks_to_pending(chunks: list, doc_name: str,
                              max_chunks: int = MAX_CHUNKS_PER_DOC) -> dict:
    """对已分块的文档做抽取, 结果写进待审核队列。返回统计。

    给上传文档审核通过后的后台任务用 —— 文档此时已经入库, 分块在正式分块文件里,
    不再是预览期的临时文件。

    只抽**父块**(chunk_type == "parent"): 章节正好是关系最密集的粒度, 而子块与父块
    内容重叠, 抽子块既多花 6 倍调用, 又抽出一堆重复三元组 (这 6 份 PDF: 父块 205 次
    调用 vs 子块 1273 次)。没有父块的旧格式文档退化成整篇按 800 字切。
    """
    texts = [c.get("content", "") for c in chunks if c.get("chunk_type") == "parent"]
    if not texts:
        texts = chunk_text("\n".join(c.get("content", "") for c in chunks))
    if len(texts) > max_chunks:
        print(f"  ⚠️  {doc_name}: {len(texts)} 个父块, 只抽前 {max_chunks} 个 (上限 MAX_CHUNKS_PER_DOC)")
        texts = texts[:max_chunks]

    known = existing_triple_keys()
    all_triples = []
    for text in texts:
        all_triples.extend(extract_from_text(text, doc_name))

    seen, unique, skipped = set(), [], 0
    for t in all_triples:
        key = (t["head"], t["relation"], t["tail"])
        if key in seen:
            continue
        seen.add(key)
        if key in known:
            skipped += 1
            continue
        unique.append(t)

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PENDING_DIR / f"{Path(doc_name).stem}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for t in unique:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    return {"chunks": len(texts), "extracted": len(all_triples),
            "queued": len(unique), "skipped_known": skipped, "path": str(out_path)}


def main():
    parser = argparse.ArgumentParser(description="LLM 抽取知识图谱三元组")
    parser.add_argument("--doc", type=str, default=None, help="只抽取指定文档")
    parser.add_argument("--limit", type=int, default=None, help="只抽取前N个文档")
    args = parser.parse_args()

    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    # 收集文档
    docs = []
    for ext in ["*.md", "*.txt"]:
        docs.extend(KB_DIR.glob(ext))
    docs = sorted(docs)
    if args.doc:
        docs = [d for d in docs if d.name == args.doc]
    if args.limit:
        docs = docs[:args.limit]

    if not docs:
        print(f"❌ 未找到文档 (目录: {KB_DIR})")
        return

    print(f"🚀 开始抽取: {len(docs)} 个文档")
    print(f"   输出目录: {PENDING_DIR}")
    print("=" * 60)

    total_triples = 0
    for doc_path in docs:
        doc_name = doc_path.name
        print(f"\n📄 {doc_name}")

        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"   分块: {len(chunks)} 块")

        all_triples = []
        for i, chunk in enumerate(chunks):
            triples = extract_from_text(chunk, doc_name)
            all_triples.extend(triples)
            if triples:
                print(f"   块{i+1}: 抽取 {len(triples)} 条")

        # 去重 (head+relation+tail)
        seen = set()
        unique = []
        for t in all_triples:
            key = (t["head"], t["relation"], t["tail"])
            if key not in seen:
                seen.add(key)
                unique.append(t)

        # 写入待审核文件
        out_path = PENDING_DIR / f"{doc_path.stem}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for t in unique:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        print(f"   ✅ 去重后 {len(unique)} 条 → {out_path.name}")
        total_triples += len(unique)

    print("\n" + "=" * 60)
    print(f"📊 抽取完成: {len(docs)} 个文档, {total_triples} 条三元组 (待审核)")
    print(f"   审核入口: 管理员端 → 知识图谱审核 → 通过后自动入库 Neo4j")


if __name__ == "__main__":
    main()
