# 理财智能客服系统

> 面向金融 / 银行理财产品场景的智能客服：产品咨询、收益口径、存款保险、适当性合规、购买门控与人工协同。
> 基于 **Gemini（免费档）+ 双路检索（Milvus 向量 + BM25）+ 知识图谱多跳（Neo4j / 内存降级）**
> 的 LangGraph 多 Agent 状态机，资金动账走确定性门控、全程留痕。

---

## 一、能力亮点

- **理财产品域客服**：11 款演示产品（自营理财 R1~R4、存款类、代销基金、代销保险）来自单一事实源
  `data/finance_kb/catalog.jsonl`，另有 11 篇制度类 + 11 篇产品说明书 Markdown 组成检索知识库。
- **12 类理财意图**（LLM few-shot + 规则兜底）：产品咨询 / 对比 / 适当性 / 收益口径 / 存款保险 /
  购买流程 / 持有赎回 / 费用 / 风险举报 / 投诉 / 闲聊 / 未知。
- **确定性资金门控（不赌 LLM 自觉）**：申购走 `subscribe_node` —— 归属校验 → 适当性匹配
  （测评等级 < 产品风险等级即拦截/转人工）→ 起购金额 → 短信验证码二次确认 → 写持仓并审计留痕。
  资金动账**不注册为 Agent 可自由调用的工具**。
- **知识图谱多跳（GraphRAG 局部思路）**：90 条金融三元组（存款保险覆盖/不保、代销边界、保险犹豫期、
  适当性、征信链），检索时对关系/多实体问题注入 1~3 条带 `doc` 溯源的推理路径；Neo4j 可用走 Neo4j，
  否则内存邻接自动降级（链路不断）。前端溯源以 🕸️ 角标展示图谱路径。
- **金融合规**：出站强制 PII 掩码（身份证/卡号/手机号）、广告法与金融禁语（稳赚/零风险/承诺保本）、
  非存款产品涉收益自动附风险提示；用户私有/动账类问答不进共享语义缓存（防跨用户串答）。
- **反幻觉**：QuickCheck 正则预检（数字单位含元/万/%/年化等）+ LLM 审查 + ≤2 次重试。
- **人工协同闭环**：疑似诈骗/投诉确定性转人工；坐席工作台（接单/回复/AI 建议草稿/关闭）；全程审计。
- **可观测**：多 Agent 时间线、SLI（QPS/成功率/语义缓存命中/P50~P90）、审计日志、理财意图分布。

## 二、技术栈

- **LLM**：Gemini 免费档（`LLM_PROVIDER=gemini`，可切任意 OpenAI 兼容端点）；嵌入本地 `bge-base-zh-v1.5`（768 维，离线）。
- **编排**：LangGraph 状态机 + MemorySaver checkpointer（`src/graph/cs_graph.py`），双后端 `/chat/sync` 与 `/chat/stream`(SSE)。
- **检索**：Milvus 向量（FAISS 降级）与 BM25 两路，RRF 融合 + MMR 重排；图谱多跳补充注入。
- **知识图谱**：Docker Neo4j（`docker compose up -d neo4j`）+ 内存邻接降级；三元组 `data/finance_kb/kg_triples.jsonl`。
- **安全/合规**：工具安全拦截器（越权审计）、PII 掩码、风险提示、资金操作留痕（Redis `security:events`）。
- **人机协同**：转人工工单（open→take→reply→close）+ 用户端实时轮询 + AI 建议草稿。
- **服务化**：MCP（stdio）暴露只读理财工具（产品/适当性/持仓/知识库）。
- **前端**：React 18 + TS + Vite + AntD5 + Zustand；聊天时间线 + 理财意图分布 + SLI 面板 + 坐席工作台。

## 三、关键目录

```
src/
  graph/cs_graph.py         # LangGraph 状态机(分类→记忆→检索→槽位→编排→门控/领域→审核→格式化)
  agents/                   # router_agent(意图) / agentic_agent(工具循环) / reviewer_agent(反幻觉) / prompts.py
  retrieval/retriever.py    # HybridRetriever(Milvus/FAISS + BM25 + 图谱多跳注入)
  knowledge_graph/          # 三元组加载/内存邻接/Neo4j 适配/GraphRetriever
  business/                 # finance_services(查产品/适当性/申购门控/持仓) + mock_finance
  core/                     # compliance(PII/风险提示) / slot_filler(理财槽位) / answer_cache
  security/interceptor.py   # 工具权限审计
  cache/                    # gateway_cache(语义缓存) / privacy(缓存隐私守卫)
  mcp/server.py             # FastMCP 只读工具
  api/routes/               # chat(stream/handoff/feedback) / monitor(SLI/审计) / auth / knowledge
data/
  finance_kb/               # catalog.jsonl + 制度/产品说明 Markdown + kg_triples.jsonl
  eval/finance_qa_golden.jsonl  # 40 条 Golden QA
  _archive_customer_20260903/   # 历史电商示例(归档, 未删除)
scripts/
  build_finance_kb.py       # KB→chunks→FAISS/BM25(+Milvus 同步)
  load_kg_to_neo4j.py       # 三元组→Neo4j 灌库(幂等)
  eval_finance.py           # 金融 Golden QA 评测(离线:证据覆盖/路径一致性; --with-answers:意图/合规)
tests/                      # 97 passed / 1 skipped(Gemini 环境用例除外)
```

## 四、启动

前置：`.env` 配好 `LLM_PROVIDER=gemini` + `GEMINI_API_KEY`；Redis 与 Milvus（可选，FAISS 自动降级）；
Neo4j（可选，内存图谱降级）。

```bash
# 依赖容器(按需): redis / milvus 栈 / neo4j
docker compose up -d redis etcd minio milvus      # Milvus 向量 + Redis
docker compose up -d neo4j                        # 知识图谱(Docker, 镜像源已配 proxy.vvvv.ee 可拉)

# 构建知识库索引(chunks → FAISS/BM25, Milvus 可用则同步重灌向量)
.venv/Scripts/python.exe scripts/build_finance_kb.py

# 灌知识图谱三元组(Neo4j 起后)
.venv/Scripts/python.exe scripts/load_kg_to_neo4j.py

# 启动
python run.py                # 前台
python run.py --daemon       # 后台
python run.py --stop / --status / --dev
```

浏览器 http://localhost:8001 （单端口托管前端 + API）；Swagger /docs。

## 五、评测

```bash
# 离线评测(无需 LLM): 证据覆盖 / 多跳路径结构一致 / 图谱触发
.venv/Scripts/python.exe scripts/eval_finance.py
# 在线评测(需 Gemini): 意图准确率 / 合规通过率 / (可选 Judge)
.venv/Scripts/python.exe scripts/eval_finance.py --with-answers --limit 10
```

Golden 40 条：简单事实 + 多跳(带 expected_path) + 合规哨兵(risk_required)。

## 六、已知说明

- 资金动账只走确定性门控节点并留痕；MCP 仅暴露只读工具。
- 图谱生产后端为 Neo4j（`scripts/load_kg_to_neo4j.py` 已灌 63 节点 / 78 关系验证）；未起时自动用内存邻接（功能等价）。
- 全量单测：`./.venv/Scripts/python.exe -m pytest tests -q`。
