# 归档脚本（不要运行）

这些是 2026-09 知识库扩充那次的一次性脚本，已被 `scripts/sync_kb.py` 取代。
保留只为追溯，**不要直接运行**。

| 文件 | 为什么归档 |
|---|---|
| `expand_kb.py` | **危险**。用 `"a"` append 模式直接往产物 `chunks_processed.jsonl` 里追加内容，不是幂等的 —— 再跑一次会产生重复 chunk。它的内容已全部迁进 `data/finance_kb/*.md` 源文件，重新生成请走 `sync_kb.py`。 |
| `rebuild_faiss_only.py` | 只重建 FAISS，绕过完整流水线（不更新 BM25 / Milvus / 图谱 / 三元组），会造成几路索引不一致。用 `sync_kb.py`。 |
| `verify_kb.py` | 硬编码"新增的13条chunks"（取 `chunks[-13:]`），依赖当时的追加顺序，早已失真。用 `scripts/check_duplicates.py`。 |
| `test_retrieval.py` | 8 条查询的目视冒烟（只 print 无断言）。要验证检索请跑 `scripts/eval_finance.py`。 |

## 扩知识库的正确姿势

往 `data/finance_kb/` 里丢 `.md` 文件（`## ` 小节会被切成 chunk），然后：

```bash
.venv/Scripts/python.exe scripts/sync_kb.py
```

它会比对源指纹 → 重建 chunks/FAISS/BM25/Milvus → 从 catalog 派生产品文档与图谱三元组
→ 跑回归 → 报告本次新增/修改/删除了哪些知识块。
