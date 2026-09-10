"""
6 种 RAG 策略精简实现 — 基于项目核心逻辑，去掉 try/except/日志/降级
运行：.venv\Scripts\python.exe rag_strategies.py

策略清单：
  1. rag_vector      — 纯向量检索（最简 RAG）
  2. rag_hybrid      — 混合检索（向量 + BM25 + RRF 融合）
  3. rag_hybrid_mmr  — 混合检索 + MMR 多样性重排
  4. rag_graph       — 混合检索 + 知识图谱关系推理注入
  5. rag_agentic     — Agentic RAG（Function Calling 工具循环）
  6. rag_audited     — 带反幻觉审核的 RAG（生成→审核→重生成循环）
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ═══════════════════════════════════════════════════════
# 公共部分：知识库 / 模型 / BM25 / LLM
# ═══════════════════════════════════════════════════════

KNOWLEDGE = [
    "稳盈添利30天是固定收益类理财产品，风险等级R2，业绩比较基准3.00%，起购1万元，封闭30天。",
    "日日盈现金管理类是货币类理财产品，风险等级R1，七日年化2.10%，起购100元，T+0随时赎回。",
    "大额存单3年期是存款产品，风险等级R1，执行利率3.00%，起购20万元，本息受存款保险保障。",
    "平衡增利180天是固收增强类理财产品，风险等级R3，业绩比较基准4.00%，起购5万元，封闭180天。",
]

# 知识图谱三元组（策略4用）
KG_TRIPLES = [
    ("理财产品", "不属于", "存款"),
    ("存款保险", "只保障", "存款"),
    ("稳盈添利30天", "属于", "理财产品"),
    ("大额存单3年期", "属于", "存款"),
]

# 加载 embedding 模型（复用项目的 bge）
model = SentenceTransformer("D:/aiproject/models/BAAI/bge-base-zh-v1.5")
doc_embeddings = model.encode(KNOWLEDGE, normalize_embeddings=True)

# BM25 索引（中文按字分词，项目里用 jieba，这里精简用字级）
bm25 = BM25Okapi([list(doc) for doc in KNOWLEDGE])

# LLM 客户端（复用项目）
from src.llm.llm_client import create_client
llm = create_client()


def _generate(query, contexts):
    """统一的生成函数：把检索结果拼到 prompt 里调 LLM"""
    prompt = f"你是理财客服，基于以下参考资料回答用户问题，资料中没有的信息如实说明，不要编造。\n\n"
    prompt += "\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    prompt += f"\n\n用户问题：{query}\n回答："
    return llm.chat([{"role": "user", "content": prompt}], temperature=0.0)


# ═══════════════════════════════════════════════════════
# 策略 1：纯向量检索
# ═══════════════════════════════════════════════════════

def rag_vector(query, top_k=2):
    """最简 RAG：问题向量化 → 余弦相似度 → 取 top_k → 拼 prompt 生成"""
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = [float(np.dot(query_emb, e)) for e in doc_embeddings]
    top_idx = np.argsort(scores)[::-1][:top_k]
    contexts = [KNOWLEDGE[i] for i in top_idx]
    return _generate(query, contexts)


# ═══════════════════════════════════════════════════════
# 策略 2：混合检索（向量 + BM25 + RRF 融合）
# ═══════════════════════════════════════════════════════

def rrf_fuse(rank_lists, k=60):
    """
    RRF（倒数秩融合）：把多个检索系统的排名合并
    rank_lists: [[doc_idx, ...], ...] 每个系统的排名（从高到低）
    k=60 是平滑常数，防止第1名分数过高垄断
    """
    scores = {}
    for ranks in rank_lists:
        for rank, doc_idx in enumerate(ranks):
            scores[doc_idx] = scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def rag_hybrid(query, top_k=2):
    """两路检索 → RRF 融合 → 取 top_k → 生成"""
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    # 向量路：余弦相似度排名
    dense_scores = [float(np.dot(query_emb, e)) for e in doc_embeddings]
    dense_ranks = list(np.argsort(dense_scores)[::-1])

    # BM25路：关键词匹配排名
    bm25_scores = bm25.get_scores(list(query))
    sparse_ranks = list(np.argsort(bm25_scores)[::-1])

    # RRF 融合
    fused = rrf_fuse([dense_ranks, sparse_ranks])[:top_k]
    contexts = [KNOWLEDGE[i] for i in fused]
    return _generate(query, contexts)


# ═══════════════════════════════════════════════════════
# 策略 3：混合检索 + MMR 多样性重排
# ═══════════════════════════════════════════════════════

def mmr_rerank(query_emb, candidates, top_k, lambda_param=0.5):
    """
    MMR（最大边际相关性）：在相关性和多样性之间平衡
    MMR = λ * 与问题相关性 - (1-λ) * 与已选段落的最大相似度
    λ=0.5 表示相关性和多样性各占一半
    candidates: [(doc_idx, embedding), ...]
    """
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_score, best_idx = -float("inf"), -1
        for idx in remaining:
            rel = float(np.dot(query_emb, candidates[idx][1]))
            max_sim = max((float(np.dot(candidates[idx][1], candidates[s][1])) for s in selected), default=0)
            mmr = lambda_param * rel - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i][0] for i in selected]


def rag_hybrid_mmr(query, top_k=2):
    """混合检索取 top10 候选 → MMR 重排选 top_k → 生成"""
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    dense_scores = [float(np.dot(query_emb, e)) for e in doc_embeddings]
    dense_ranks = list(np.argsort(dense_scores)[::-1])
    bm25_scores = bm25.get_scores(list(query))
    sparse_ranks = list(np.argsort(bm25_scores)[::-1])

    # 先取 10 个候选（多取一些给 MMR 选择空间）
    fused = rrf_fuse([dense_ranks, sparse_ranks])[:10]
    candidates = [(idx, doc_embeddings[idx]) for idx in fused]

    # MMR 重排
    final_idx = mmr_rerank(query_emb, candidates, top_k)
    contexts = [KNOWLEDGE[i] for i in final_idx]
    return _generate(query, contexts)


# ═══════════════════════════════════════════════════════
# 策略 4：混合检索 + 知识图谱关系推理
# ═══════════════════════════════════════════════════════

def graph_retrieve(query):
    """图谱检索：提取查询中的实体，找相关三元组，转成自然语言证据"""
    matched = []
    for head, relation, tail in KG_TRIPLES:
        if head in query or tail in query:
            matched.append(f"{head} {relation} {tail}")
    return matched


def rag_graph(query, top_k=2):
    """混合检索 + MMR + 图谱关系证据注入 → 生成"""
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    dense_scores = [float(np.dot(query_emb, e)) for e in doc_embeddings]
    dense_ranks = list(np.argsort(dense_scores)[::-1])
    bm25_scores = bm25.get_scores(list(query))
    sparse_ranks = list(np.argsort(bm25_scores)[::-1])
    fused = rrf_fuse([dense_ranks, sparse_ranks])[:10]
    candidates = [(idx, doc_embeddings[idx]) for idx in fused]
    final_idx = mmr_rerank(query_emb, candidates, top_k)

    contexts = [KNOWLEDGE[i] for i in final_idx]

    # 图谱证据注入（关系类查询才有用）
    graph_evidence = graph_retrieve(query)
    if graph_evidence:
        contexts.append("【关系推理】" + "；".join(graph_evidence))

    return _generate(query, contexts)


# ═══════════════════════════════════════════════════════
# 策略 5：Agentic RAG（Function Calling 工具循环）
# ═══════════════════════════════════════════════════════

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "检索理财知识库，返回相关产品资料原文",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索关键词或问题"}},
            "required": ["query"],
        },
    },
}]


def search_knowledge(query):
    """工具实现：混合检索（策略3）"""
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    dense_scores = [float(np.dot(query_emb, e)) for e in doc_embeddings]
    dense_ranks = list(np.argsort(dense_scores)[::-1])
    bm25_scores = bm25.get_scores(list(query))
    sparse_ranks = list(np.argsort(bm25_scores)[::-1])
    fused = rrf_fuse([dense_ranks, sparse_ranks])[:3]
    return "\n".join(KNOWLEDGE[i] for i in fused)


def rag_agentic(query):
    """
    Agentic RAG：LLM 自主决定是否调用检索工具
    循环最多 6 轮，直到 LLM 返回纯文本（不再要求调用工具）
    """
    messages = [
        {"role": "system", "content": "你是理财客服。可调用 search_knowledge 工具检索资料，基于资料回答。资料中没有的如实说明。"},
        {"role": "user", "content": query},
    ]

    for _ in range(6):
        result = llm.chat_with_tools(messages, TOOLS)
        if result["type"] != "tool_calls":
            return result["text"]  # LLM 返回纯文本，结束

        # LLM 要求调用工具 → 执行工具 → 结果追加回 messages → 继续循环
        messages.append({"role": "assistant", "tool_calls": result["calls"]})
        for call in result["calls"]:
            tool_result = search_knowledge(call["arguments"]["query"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result})

    return "工具调用次数过多，请简化问题"


# ═══════════════════════════════════════════════════════
# 策略 6：带反幻觉审核的 RAG
# ═══════════════════════════════════════════════════════

def quick_check(answer, contexts):
    """
    QuickCheck 正则审核：检查回答中的数字是否都在上下文中出现
    这是项目里 reviewer_agent 的核心预检逻辑
    """
    context_text = " ".join(contexts)
    numbers = re.findall(r"[\d.]+", answer)
    for num in numbers:
        if len(num) > 1 and num not in context_text:
            return False, f"数字 '{num}' 在参考资料中找不到"
    return True, "通过"


def rag_audited(query, max_retry=2):
    """
    带审核的 RAG：检索 → 生成 → QuickCheck 审核
    审核不通过则注入反馈重新生成，最多重试 max_retry 次
    """
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = [float(np.dot(query_emb, e)) for e in doc_embeddings]
    top_idx = np.argsort(scores)[::-1][:2]
    contexts = [KNOWLEDGE[i] for i in top_idx]

    for retry in range(max_retry + 1):
        if retry == 0:
            answer = _generate(query, contexts)
        else:
            # 重试时注入反馈：明确要求不要编造资料中没有的数字
            prompt = f"你是理财客服，严格基于以下参考资料回答，**绝对不要编造资料中没有的数字和事实**。\n\n"
            prompt += "\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
            prompt += f"\n\n用户问题：{query}\n回答："
            answer = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)

        passed, reason = quick_check(answer, contexts)
        if passed:
            return answer
        print(f"  [审核] 第{retry+1}次未通过：{reason}，重新生成")

    return answer  # 重试用完，返回最后一次结果


# ═══════════════════════════════════════════════════════
# 运行：对比 6 种策略
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    query = "稳盈添利30天的风险等级和收益是多少？理财受存款保险保护吗？"

    print(f"\n{'='*60}")
    print(f"用户问题：{query}")
    print(f"{'='*60}\n")

    strategies = [
        ("策略1 纯向量检索", rag_vector),
        ("策略2 混合检索(RRF)", rag_hybrid),
        ("策略3 混合+MMR重排", rag_hybrid_mmr),
        ("策略4 混合+知识图谱", rag_graph),
        ("策略5 Agentic工具循环", rag_agentic),
        ("策略6 带审核重试", rag_audited),
    ]

    for name, func in strategies:
        print(f"--- {name} ---")
        answer = func(query)
        print(f"{answer}\n")
