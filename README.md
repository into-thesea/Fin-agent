# 理财智能客服系统

> 面向银行 / 金融理财产品场景的智能客服：产品咨询、收益口径、存款保险、适当性合规、申购意向预检与人工协同。
> 基于 **DeepSeek（deepseek-chat）+ 双路混合检索（Milvus 向量 + BM25，RRF 融合）+ 知识图谱多跳（Neo4j / 内存降级）+ 父子文档分块**
> 的 LangGraph 多 Agent 状态机，资金动账走确定性预检节点、全程合规留痕。

---

## 一、能力亮点

- **理财产品域客服**：24 款产品（自营理财 R1~R4、存款类、代销基金、代销保险）来自单一事实源
  `data/finance_kb/catalog.jsonl`，另有 22 篇制度类 + 产品说明书 Markdown 组成检索知识库，
  经父子文档分块后共 465 个检索块（213 父块 + 252 子块）。
- **12 类理财意图**（LLM few-shot + CoT 推理 + 规则兜底）：产品咨询 / 对比 / 适当性 / 收益口径 / 存款保险 /
  购买流程 / 持有赎回 / 费用 / 风险举报 / 投诉 / 闲聊 / 未知。
- **续轮意图继承**：多轮对话中，验证码、确认词、纯金额、短问法等续轮输入通过确定性规则自动继承上一轮意图，
  跳过路由 LLM（零延迟），续轮准确率 100%；22 词话题切换检测防止错误传染。
- **申购意向合规预检（通用规则，不执行真实交易）**：用户表达购买意向时走 `subscribe_node` ——
  产品名校验 → 起购金额校验 → R4/R5 高风险产品统一转人工 → 验证码流程引导（首次提示、输入任意 6 位数字视为确认）
  → 意向确认话术（明确声明"不构成真实交易指令"）。不写持仓、不动账、不依赖用户 mock 数据。
  资金动账**不注册为 Agent 可自由调用的工具**。
- **知识图谱多跳（GraphRAG 局部思路）**：180 条金融三元组（存款保险覆盖/不保、代销边界、保险犹豫期、
  适当性、征信链），检索时对关系/多实体问题注入 1~3 条带 `doc` 溯源的推理路径；Neo4j 可用走 Neo4j，
  否则内存邻接自动降级（链路不断）。前端溯源以 🕸️ 角标展示图谱路径。
- **快速通道五道闸**：高频简单问题经置信度闸、知识库命中闸等五道确定性判断后直接返回，
  跳过完整 Agent 链路，P50 延迟降至 6.80s。
- **金融合规**：出站强制 PII 掩码（身份证/卡号/手机号）、广告法与金融禁语（稳赚/零风险/承诺保本）、
  非存款产品涉收益自动附风险提示；用户私有/动账类问答不进共享语义缓存（防跨用户串答）。
- **反幻觉**：QuickCheck 正则预检（数字单位含元/万/%/年化等）+ LLM 审查 + ≤2 次重试。
- **流式输出**：回答通过 SSE 逐字返回，首字延迟显著低于整包返回。
- **人工协同闭环**：疑似诈骗/投诉/R4+高风险申购确定性转人工；坐席工作台（接单/回复/AI 建议草稿/关闭）；全程审计。
- **可观测**：多 Agent 时间线、SLI（QPS/成功率/语义缓存命中/P50~P95）、审计日志、理财意图分布。

## 二、技术栈

- **LLM**：DeepSeek 官方接口（`deepseek-chat`）；嵌入本地 `bge-base-zh-v1.5`（768 维，离线，不依赖外网）。
- **编排**：LangGraph 状态机 + MemorySaver checkpointer（`src/graph/cs_graph.py`），双后端 `/chat/sync` 与 `/chat/stream`(SSE)。
- **检索**：Milvus 向量与 BM25 两路，RRF 融合 + MMR 重排；父子文档分块（子块检索、父块拼接上下文）；图谱多跳补充注入。
- **知识图谱**：Docker Neo4j + 内存邻接降级；三元组 `data/finance_kb/kg_triples.jsonl`（180 条）。
- **意图识别**：路由 LLM（CoT + 动态 few-shot）+ 规则兜底 + 续轮意图继承（`src/core/intent_inheritance.py`）。
- **安全/合规**：工具安全拦截器（越权审计）、PII 掩码、风险提示、资金操作留痕（Redis `security:events`）。
- **人机协同**：转人工工单（open→take→reply→close）+ 用户端实时轮询 + AI 建议草稿。
- **服务化**：MCP（stdio）暴露只读理财工具（产品/适当性/持仓/知识库）。
- **前端**：React 18 + TS + Vite + AntD5 + Zustand；聊天时间线 + 理财意图分布 + SLI 面板 + 坐席工作台。

## 三、关键目录

```
src/
  graph/cs_graph.py         # LangGraph 状态机(分类→记忆→检索→槽位→编排→预检/领域→审核→格式化)
  agents/                   # router_agent(意图, CoT+动态few-shot) / agentic_agent(工具循环) / reviewer_agent(反幻觉) / prompts.py
  core/
    intent_inheritance.py   # 续轮意图继承(验证码/确认词/纯金额/短问法四条规则 + 话题切换检测)
    fast_path.py            # 快速通道五道闸(置信度/知识库命中等确定性判断)
    slot_filler.py          # 理财槽位抽取(产品名/金额/验证码) + is_executable_order
    compliance.py           # PII掩码/风险提示/金融禁语
    answer_cache.py         # 语义缓存
  retrieval/retriever.py    # HybridRetriever(Milvus + BM25 + 图谱多跳注入 + 父子分块)
  knowledge_graph/          # 三元组加载/内存邻接/Neo4j 适配/GraphRetriever
  business/
    finance_services.py     # 查产品/适当性/申购意向预检(通用规则)/持仓查询
    mock_finance.py         # 银行核心系统对接预留接口的 Mock 实现(仅只读查询工具用, 主流程不依赖)
  security/interceptor.py   # 工具权限审计
  cache/                    # gateway_cache(语义缓存) / privacy(缓存隐私守卫)
  mcp/server.py             # FastMCP 只读工具
  api/routes/               # chat(stream/handoff/feedback) / monitor(SLI/审计) / auth / knowledge
data/
  finance_kb/               # catalog.jsonl(24款) + 制度/产品说明 Markdown(22篇) + kg_triples.jsonl(180条)
  eval/
    finance_qa_golden.jsonl # 105 条 Golden QA(10类意图, 含对抗/超纲/模糊样本)
    finance_sessions.jsonl  # 多轮会话评测集(8会话19轮)
scripts/
  build_finance_kb.py       # KB→父子分块→Milvus 建索引 + BM25
  load_kg_to_neo4j.py       # 三元组→Neo4j 灌库(幂等)
  eval_finance.py           # 单轮 Golden QA 评测(离线:证据覆盖/chunk命中/路径一致; --with-answers:意图/合规)
  eval_sessions.py          # 多轮会话评测(多轮意图/续轮/指代消解/合规)
tests/                      # 373 passed / 1 skipped
```

## 四、启动

前置：`.env` 配好 `DEEPSEEK_API_KEY`（`LLM_PROVIDER=deepseek`）；Redis 与 Milvus；
Neo4j（可选，内存图谱降级）。

```bash
# 依赖容器(按需): redis / milvus 栈 / neo4j
docker compose up -d redis etcd minio milvus      # Milvus 向量 + Redis
docker compose up -d neo4j                        # 知识图谱(Docker)

# 构建知识库索引(父子分块 → Milvus 向量 + BM25)
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
# 离线评测(无需 LLM): 证据覆盖 / chunk命中 / 多跳路径结构一致 / 图谱触发
.venv/Scripts/python.exe scripts/eval_finance.py
# 在线评测(需 DeepSeek): 意图准确率 / 合规通过率
.venv/Scripts/python.exe scripts/eval_finance.py --with-answers
# 多轮会话评测: 多轮意图 / 续轮继承 / 指代消解 / 合规
.venv/Scripts/python.exe scripts/eval_sessions.py
```

### 最新指标（2026-10-05）

| 类别 | 指标 | 当前值 |
|------|------|--------|
| 单轮（105 条） | 全链路意图准确率 | 89.5% |
| 单轮 | 合规通过率 | 94.3% |
| 检索 | 证据覆盖率 | 99.0% |
| 检索 | chunk 信息命中率 | 98.0% |
| 检索 | 图谱路径结构一致率 | 83.3% |
| 多轮（8 会话 19 轮） | 多轮意图准确率 | 84.2% |
| 多轮 | 续轮意图准确率 | 100% |
| 多轮 | 指代消解成功率 | 100% |
| 性能（40 条） | 端到端延迟 P50 / P95 | 7.10s / 11.40s |
| 性能 | 快速通道 P50 / P95 | 6.80s / 7.48s |
| 工程 | 单元/接口测试 | 373 passed |

详细评测口径与归因见 `docs/优化后评测指标报告.md`。

## 六、已知说明

- 申购意向预检为通用规则演示流程，不执行真实交易、不写持仓、不动账；实际申购需通过手机银行或柜台办理。
- `mock_finance.py` 仅用于只读查询工具（持仓/账户/交易等）的演示数据，主流程节点不依赖 mock。
- 图谱生产后端为 Neo4j；未起时自动用内存邻接（功能等价）。
- 全量单测：`./.venv/Scripts/python.exe -m pytest tests -q`。
