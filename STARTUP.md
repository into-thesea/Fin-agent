# 智能客服系统 启动指南（Fin-Agent 改造版）

> 详细技术栈与架构见 [README.md](README.md)。本文档只讲怎么跑起来。

## 系统架构（当前）

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  React 前端   │ ──▶ │  FastAPI 后端 │ ──▶ │  Gemini LLM   │
│  (Vite, 单端口)│    │  localhost:8001│    │  +降级链       │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                    ┌───────┴────────┐
                    │ 路由 → Agentic │
                    │ Agent 工具循环  │
                    │ (FunctionCall) │
                    └───────┬────────┘
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          ┌────────┐  ┌──────────┐  ┌──────────┐
          │ Milvus │  │  BM25    │  │  Neo4j    │
          │ Docker │  │ +RRF/MMR │  │ Docker    │
          │ 向量检索│  │ 稀疏检索 │  │ 图谱检索  │
          └────────┘  └──────────┘  └──────────┘
```

## 前置条件

### 1. 基础设施（二选一）

**Docker 方式（推荐）**：

```bash
# Redis
docker run -d --name fin-redis -p 6379:6379 redis:7-alpine
# Neo4j (密码 finagent123, 与 .env 一致)
docker run -d --name fin-neo4j \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/finagent123 \
  neo4j:5
```

**WSL/本地方式**：

```bash
wsl redis-server --daemonize yes --port 6379
# Neo4j 需另行安装/启动 (或用 Docker 的 fin-neo4j)
```

### 2. Python 依赖

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install neo4j                 # 额外依赖
```

### 3. API Key

`.env` 中配置（至少一个）：

```env
# Gemini（当前主选）
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-3.5-flash,gemini-2.5-pro,gemini-3.1-flash-lite

# 或 DeepSeek（备用）
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=sk-your-key
```

### 4. 图谱数据（可选）

Neo4j 若为空库，先灌入图谱数据：

```bash
.venv/Scripts/python.exe scripts/load_graph_to_neo4j.py      # 实体JSONL → Neo4j
.venv/Scripts/python.exe -m src.graph.entity_merger_simple   # 核心车企别名消歧
```

---

## 快速启动

```bash
cd D:/aiproject
python run.py --daemon --celery     # 后台运行 + Celery Worker (推荐)
```

访问 **http://localhost:8001**（前后端一体，单端口）。Swagger：**http://localhost:8001/docs**

| 命令 | 作用 |
|---|---|
| `python run.py` | 前台运行 |
| `python run.py --daemon` | 后台运行 |
| `python run.py --daemon --celery` | 后台 + Celery Worker |
| `python run.py --dev` | 开发模式（前端 3000 热更新） |
| `python run.py --stop` / `--restart` | 停止 / 重启 |
| `python run.py --status` | 查看状态 |

---

## 验证

```bash
# 全部应为 ok
curl http://localhost:8001/api/v1/system/status
# 图谱应走 Neo4j (source: neo4j)
curl http://localhost:8001/api/v1/graph/stats
```

---

## API 接口

### 同步聊天 `POST /api/v1/chat/sync`

```bash
curl -X POST http://localhost:8001/api/v1/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "存款保险保多少钱？"}'
```

### 流式聊天 `POST /api/v1/chat/stream`（SSE）

```
data: {"type": "stage", "stage": "routing", "message": "🔍 正在分析问题意图..."}
data: {"type": "stage", "stage": "retrieving", "message": "📚 正在检索金融知识库..."}
data: {"type": "result", "answer": "...", "sources": [...], "review": {...}}
data: {"type": "done"}
```

### 图谱查询

| 接口 | 说明 |
|---|---|
| `GET /api/v1/graph/stats` | 图谱统计 |
| `GET /api/v1/graph/entity/{名称}` | 实体 1/2-hop 邻接 |
| `GET /api/v1/graph/search?q=` | 全文搜索实体 |
| `GET /api/v1/graph/visualize?entity=` | 力导向图数据 |

---

## 性能说明

- 首请求：加载 bge-base-zh-v1.5 模型 ~3-5s，总计 ~60s（含模型）
- 后续请求：~6-13s（Router 1-3s + 检索 1-2s + Agent 3-5s + Reviewer 1-3s）
- 缓存命中：~2s
- 对比类问题：>40s（可能触发 504，属已知限制）

## 故障排除

| 症状 | 处理 |
|---|---|
| 端口被占 | `netstat -ano \| grep :8001` → `taskkill //F //PID <PID>` → `python run.py --daemon` |
| 服务 degraded | 逐个查：`wsl redis-cli ping` / `docker ps`（fin-neo4j）/ Celery |
| Neo4j 慢连/超时 | 确认 `.env` 用 `bolt://127.0.0.1:7687`（非 localhost，IPv6 坑） |
| 402 余额不足 | DeepSeek/Gemini 充值或换 Key |
| 前端白屏 | `frontend/vite.config.ts` proxy target 是否指向 8001 |
| 缓存脏 | `run.py --stop` → 删 `src/data/semantic_cache.db` → 重启 |
