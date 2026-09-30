"""
延迟专项测试
- 固定一批覆盖10类意图的典型问题
- 预热排除冷启动(模型加载/连接建立)
- 实测端到端耗时, 按路径(快速通道/复杂路径)统计 P50/P95
- 独立基准化路由/检索耗时, 解释瓶颈(不与总耗时简单相加, 因复杂路径二者并行)
"""
import json, sys, os, time
sys.path.insert(0, r'D:\aiproject')
os.chdir(r'D:\aiproject')
import numpy as np

# ── 固定选取问题: 每个意图取前4条, 保证覆盖均衡且可复现 ──
with open(r'D:\aiproject\data\eval\finance_qa_golden.jsonl', 'r', encoding='utf-8') as f:
    golden = [json.loads(l) for l in f if l.strip()]

by_intent = {}
for g in golden:
    by_intent.setdefault(g['intent'], []).append(g)
selected = []
for intent, items in by_intent.items():
    selected.extend(items[:4])
print(f'固定测试集: {len(selected)} 条, 意图分布: ' +
      ', '.join(f'{k}:{len(v[:4])}' for k, v in by_intent.items()))

from src.graph.cs_graph import run
from src.retrieval.retriever import HybridRetriever
from src.agents.router_agent import RouterAgent

retriever = HybridRetriever()
router = RouterAgent()

# ── 预热: 跑3条让 bge/LLM/Milvus 全部就绪, 不计入数据 ──
print('\n预热中(排除冷启动)...')
for g in selected[:3]:
    run(g['question'], user_id='latency_warmup')
    retriever.hybrid_retrieve(g['question'])
    router.route(g['question'])
print('预热完成\n')

# ── 正式测量 ──
records = []
for i, g in enumerate(selected):
    q = g['question']

    # 端到端
    t0 = time.perf_counter()
    result = run(q, user_id='latency_test')
    t_total = time.perf_counter() - t0

    # 独立基准: 检索
    t0 = time.perf_counter()
    retriever.hybrid_retrieve(q)
    t_retrieve = time.perf_counter() - t0

    # 独立基准: 路由
    t0 = time.perf_counter()
    router.route(q)
    t_route = time.perf_counter() - t0

    rec = {
        'id': g['id'], 'intent': g['intent'], 'question': q[:36],
        'stage': result.get('stage', ''), 'routed_intent': result.get('intent', ''),
        'total_s': round(t_total, 2), 'retrieve_s': round(t_retrieve, 2),
        'route_s': round(t_route, 2),
    }
    records.append(rec)
    print(f"  [{i+1}/{len(selected)}] {rec['stage']:11s} 总{rec['total_s']:5.2f}s "
          f"检索{rec['retrieve_s']:4.2f}s 路由{rec['route_s']:4.2f}s | {rec['question']}")

# ── 统计 ──
def pct(vals, p):
    return float(np.percentile(np.array(vals), p))

def stats(vals):
    vals = sorted(vals)
    return {
        'n': len(vals), 'mean': round(float(np.mean(vals)), 2),
        'p50': round(pct(vals, 50), 2), 'p90': round(pct(vals, 90), 2),
        'p95': round(pct(vals, 95), 2), 'min': round(vals[0], 2),
        'max': round(vals[-1], 2),
    }

totals = [r['total_s'] for r in records]
fast = [r['total_s'] for r in records if r['stage'] in ('greeting', 'simple_fact')]
complex_ = [r['total_s'] for r in records if r['stage'] not in ('greeting', 'simple_fact')]
retrieves = [r['retrieve_s'] for r in records]
routes = [r['route_s'] for r in records]

summary = {
    'overall_end_to_end': stats(totals),
    'fast_path': stats(fast),
    'complex_path': stats(complex_),
    'retrieval_standalone': stats(retrieves),
    'routing_standalone': stats(routes),
}

print('\n' + '='*64)
print('延迟专项测试结果 (单位: 秒)')
print('='*64)
for name, s in summary.items():
    print(f"\n● {name}")
    print(f"   n={s['n']}  平均={s['mean']}  P50={s['p50']}  P90={s['p90']}  "
          f"P95={s['p95']}  min={s['min']}  max={s['max']}")

# 路径分布
from collections import Counter
dist = Counter(r['stage'] for r in records)
print(f"\n路径分布: {dict(dist)}")

out = r'D:\aiproject\data\eval\latency_benchmark.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump({'summary': summary, 'records': records,
               'path_distribution': dict(dist)}, f, ensure_ascii=False, indent=2)
print(f'\n明细已保存: {out}')
