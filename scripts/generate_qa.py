"""
Chunk 级 QA 标注生成脚本 — 最快路径

基于 chunks_processed.jsonl 逐 chunk 让 LLM 生成"真实用户视角"问题，
golden chunk 天然 = 当前 chunk，无需人工定位。

流程:
  1. 读取 data/output_analysis/chunks_processed.jsonl
  2. 过滤无效 chunk (内容过短)
  3. 均匀采样 N 个 chunk (覆盖多份文档)
  4. 逐 chunk 调 LLM 生成 K 个 QA (JSON 模式)
  5. 回标校验: answer 中的数字必须能在 chunk 原文中找到
  6. 丢弃"问题泄露答案"的条目 + 去重
  7. 写出 scripts/eval_qa.jsonl

用法:
    python scripts/generate_qa.py                 # 默认 40 chunk × 2 条
    python scripts/generate_qa.py --limit 20 --per-chunk 3
    python scripts/generate_qa.py --cheap         # 用便宜模型 (路由/审查档)
    python scripts/generate_qa.py --dry-run       # 只打印采样计划，不调用 LLM
"""

import sys
import os
import re
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK, 强制 UTF-8 输出避免 emoji 打印报错
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.infra.paths import CHUNKS_PROCESSED_PATH
from src.llm.llm_client import create_client

# 数字断言模式 (与 reviewer_agent.quick_check 一致, 客服单位)
NUMBER_PATTERN = re.compile(
    r"([\d,]+\.?\d*)\s*(元|折|天|件|次|个|小时|分钟|公里|克|公斤|%|号)"
)

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_qa_customer.jsonl")

GEN_PROMPT = """你是一个智能客服 QA 标注助手。给定一段客服知识片段（FAQ/政策条款/产品参数），请生成 {n} 个"真实用户会问的问题"及其标准答案。

要求:
1. 问题必须像用户自然提问，例如"退货政策是什么？""京东快递首重运费多少？"——不要写成"这段文本讲了什么"
2. 答案必须能从该片段中找到依据，政策条款/金额/天数不能改动
3. 只输出 JSON 数组，不要输出任何其他文字:
[{{"question": "...", "answer": "..."}}]

【客服知识片段】:
{content}"""


def load_chunks():
    if not os.path.exists(CHUNKS_PROCESSED_PATH):
        print(f"❌ 分块文件不存在: {CHUNKS_PROCESSED_PATH}\n   请先运行 ETL 流水线生成知识库")
        sys.exit(1)
    chunks = []
    with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


def valid_chunks(chunks, min_len=60):
    """过滤内容过短的 chunk (页眉/封面/空块)"""
    return [c for c in chunks if len((c.get("raw_content") or c.get("content") or "").strip()) >= min_len]


def sample_uniform(chunks, limit):
    """均匀采样, 保证覆盖多份文档而不是只取前几个"""
    if len(chunks) <= limit:
        return chunks
    if limit <= 1:
        return [chunks[0]]
    step = (len(chunks) - 1) / (limit - 1)
    return [chunks[int(round(i * step))] for i in range(limit)]


def load_records(path):
    """读取已有 JSONL 评测记录 (供 --resume 增量模式使用)"""
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def parse_llm_json(raw):
    """解析 LLM 输出, 兼容 markdown 代码块 / 夹杂文本 / 多个 JSON 对象 (Extra data)"""
    raw = (raw or "").strip()
    # 剥离 markdown 代码块
    raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()

    decoder = json.JSONDecoder()
    # 1) 整段就是一个 JSON 值
    try:
        obj, _ = decoder.raw_decode(raw)
        return obj
    except json.JSONDecodeError:
        pass
    # 2) 扫描首个 [ 或 { 位置, 用 raw_decode 取第一个可解析对象
    #    (处理 "Extra data" — LLM 输出多个对象或夹杂多余文本)
    for m in re.finditer(r"[\[{]", raw):
        try:
            obj, _ = decoder.raw_decode(raw[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法解析 LLM 输出: {raw[:200]}")


# 客服 QA 支撑校验停用词 (疑问/连接词不参与匹配)
_SUPPORT_STOPWORDS = {
    "怎么", "如何", "什么", "多少", "为什么", "请问", "是否", "能不能",
    "可以", "需要", "应该", "如果", "但是", "然后", "还有", "或者",
    "怎么办", "哪些", "哪个", "几种", "是什么", "吗", "呢", "了", "的", "请",
}


def _extract_terms(text):
    """提取 2-12 字中文片段 (去掉停用词), 用于支撑判定"""
    out = set()
    for part in re.split(r"[\s，。！？；：、,.!?;:\d]+", text or ""):
        p = part.strip()
        if 2 <= len(p) <= 12 and re.search(r"[一-鿿]", p):
            out.add(p)
    return out - _SUPPORT_STOPWORDS


def check_supported(answer, context):
    """回标校验 (客服版): 答案的关键词须在 chunk 原文中命中 (≥60%), 避免金融数字正则空转"""
    ctx = context or ""
    terms = _extract_terms(answer)
    if not terms:
        return True, []  # 无可验证关键词, 视为支持
    missing = [t for t in terms if t not in ctx]
    supported = len(missing) <= len(terms) * 0.4
    return supported, missing


def check_leak(question, answer):
    """问题泄露检测: 问题里不能出现答案中的数字 (检索作弊)"""
    q_norm = re.sub(r"[,\s]", "", question)
    leaked = [num for num, _ in NUMBER_PATTERN.findall(answer)
              if re.sub(r"[,\s]", "", num) in q_norm]
    return len(leaked) > 0, leaked


def main():
    parser = argparse.ArgumentParser(description="Chunk 级 QA 标注生成 (最快路径)")
    parser.add_argument("--limit", type=int, default=40, help="采样 chunk 数 (默认 40)")
    parser.add_argument("--per-chunk", type=int, default=2, help="每 chunk 生成条数 (默认 2)")
    parser.add_argument("--cheap", action="store_true", help="用便宜模型 (路由/审查档)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出文件 (默认 scripts/eval_qa.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="只打印采样计划, 不调用 LLM")
    parser.add_argument("--resume", action="store_true", help="增量模式: 读取已有输出, 只补跑失败的 chunk")
    args = parser.parse_args()

    chunks = valid_chunks(load_chunks())
    selected = sample_uniform(chunks, args.limit)
    print(f"📂 有效 chunk: {len(chunks)} 个, 采样 {len(selected)} 个")
    print(f"   预估 LLM 调用: {len(selected) * args.per_chunk} 次 (--cheap={args.cheap})")

    if args.dry_run:
        print("\n📄 采样计划:")
        for c in selected:
            preview = (c.get("raw_content") or "").replace("\n", " ")[:50]
            print(f"  · {c['chunk_id']}: {preview}")
        return

    # --resume 增量模式: 读取已有输出, 跳过已完成 chunk, 只补跑失败的
    records = []
    seen = set()
    if args.resume and os.path.exists(args.out):
        records = load_records(args.out)
        seen = {r.get("question", "") for r in records}
        done_chunks = set()
        for r in records:
            done_chunks.update(r.get("golden_chunk_ids") or [])
        selected = [c for c in selected if c["chunk_id"] not in done_chunks]
        print(f"📂 resume: 已存在 {len(records)} 条, 本次补跑 {len(selected)} 个 chunk")

    client = create_client(cheap=args.cheap)
    dropped_unsupported = 0
    dropped_leak = 0
    failed = 0

    for i, c in enumerate(selected, 1):
        content = (c.get("raw_content") or c.get("content") or "").strip()
        prompt = GEN_PROMPT.format(n=args.per_chunk, content=content[:2500])
        try:
            raw = client.chat(
                messages=[
                    {"role": "system", "content": "你是智能客服标注助手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                json_mode=True,
            )
            items = parse_llm_json(raw)
            if not isinstance(items, list):
                items = [items]
            for it in items:
                q = (it.get("question") or "").strip()
                a = (it.get("answer") or "").strip()
                if not q or not a or q in seen:
                    continue
                ok, unsup = check_supported(a, content)
                if not ok:
                    dropped_unsupported += 1
                    print(f"  ⚠️  {c['chunk_id']}: 丢弃, 数字无支撑 {unsup[:2]}")
                    continue
                leaked, nums = check_leak(q, a)
                if leaked:
                    dropped_leak += 1
                    print(f"  ⚠️  {c['chunk_id']}: 丢弃, 问题泄露答案 {nums[:2]}")
                    continue
                seen.add(q)
                records.append({
                    "id": len(records) + 1,
                    "question": q,
                    "answer": a,
                    "golden_chunk_ids": [c["chunk_id"]],
                    "intent": "faq",
                })
        except Exception as e:
            failed += 1
            print(f"  ❌ {c['chunk_id']} 生成失败: {e}")
        print(f"  [{i}/{len(selected)}] {c['chunk_id']} → 累计 {len(records)} 条")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 已生成 {len(records)} 条 QA → {args.out}")
    print(f"   丢弃: 数字无支撑 {dropped_unsupported} 条, 问题泄露 {dropped_leak} 条, 生成失败 {failed} 个 chunk")
    print(f"\n下一步: python scripts/eval_retrieval.py")


if __name__ == "__main__":
    main()
