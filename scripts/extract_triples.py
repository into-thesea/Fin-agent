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

# Windows 控制台默认 GBK, 强制 UTF-8 输出避免 emoji 打印报错
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

KB_DIR = PROJECT_ROOT / "data" / "finance_kb"
PENDING_DIR = PROJECT_ROOT / "data" / "kg_pending"

# 允许的关系类型 (与现有 kg_triples.jsonl 对齐)
ALLOWED_RELATIONS = {
    "is_subtype_of", "is_not", "covered_by", "not_covered_by",
    "requires", "has_risk", "sold_as", "applies_to", "excludes",
    "coverage_limit", "is", "has_attribute", "related_to",
}

EXTRACT_PROMPT = """你是金融知识图谱三元组抽取专家。从下面的文档片段中抽取 (主体, 关系, 客体) 三元组。

规则:
1. 只抽取明确陈述的事实，不要推断或概括
2. 主体和客体必须是文档中出现的具体名词（产品名、概念、制度）
3. 关系只能从以下列表选择: {relations}
4. 每个三元组附原文片段作为证据
5. 输出 JSON 数组，每个元素: {{"head": "", "relation": "", "tail": "", "evidence": ""}}
6. 不要输出解释，只输出 JSON

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
