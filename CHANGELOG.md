# Changelog

All notable changes to the Fin-Agent project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-08-14

### Added
- QA 评测体系: `generate_qa.py` / `eval_retrieval.py` / `enrich_golden.py` (Recall@K + MRR, 80 条 QA)
- CrossEncoder (bge-reranker-base) 两阶段重排 + SparseBoost rerank (`src/retrieval/reranker.py`)
- 图谱灌库脚本 `scripts/load_graph_to_neo4j.py` (chunk 内 id→name 解析 + 名称聚合, 防跨文档撞 id 连错边)

### Changed
- Function Calling 全量重构: `AgenticAgent` 工具循环 (retrieve_knowledge / web_search / query_graph)
- 统一提示词边界块 (`src/agents/prompts.py`); Router unknown 门控 (UNKNOWN 不检索, 礼貌拒答)
- Reviewer quick_check 增强 (来源有效性 / 未引用来源 / 检索有货却答未找到)
- Gemini REST 直连修复 thought_signature (genai SDK pydantic 层剥离)
- NEO4J_URI 强制 IPv4 (`bolt://127.0.0.1:7687`), 根治 localhost→::1 导致 Bolt 握手 10s+ 超时
- `graph_search` 支持别名匹配 (name OR aliases)
- 嵌入模型: bge-m3 → bge-base-zh-v1.5 (768-dim); FAISS 索引重建至 1497 向量

### Fixed
- graph_local `PROJECT_ROOT` 少一层 dirname, 指向空的 `src/data/output_analysis` → 修到仓库根
- `.venv` 缺 `neo4j` 驱动导致 `NEO4J_AVAILABLE=False` (需 `pip install neo4j`)
- 实体别名消歧幂等化: 短名始终作为全称节点 alias (`entity_merger_simple.py`)
- 检索索引 3 个 bug: BM25 `chunk_id` 读取 / FAISS metadata 缺 `content` / bge 模型离线复用

## [4.0.0] - 2026-06-28

### Fixed
- Async concurrency: multiple POST requests no longer block each other
- RLock deadlock in dialog_state.update()
- Synchronous LLMClient replaced with async version

## [4.0.0-rc.4] - 2026-06-27

### Added
- L1+L2 semantic cache with synonym matching (e.g., 营收 and 营业收入 are treated as equivalent)
- 2-Worker thread pool with 120s async timeout
- SQLite persistence for semantic cache
- Enhanced rule-based routing

### Fixed
- Terminal popup on startup (`CREATE_NO_WINDOW` flag in run.py)
- API key updated
- Model name corrected (deepseek-v4-flash to deepseek-chat)

## [4.0.0-rc.3] - 2026-06-26

### Changed
- Embedding model: BGE-M3 replaced with bge-base-zh-v1.5 (768-dim, ~400MB smaller footprint)
- Semantic chunking: chapter-aware markdown splitting
- run.py: auto-clear ports, PID files, added --stop/--force/--daemon flags
- Cleaned up legacy files

## [4.0.0-rc.2] - 2026-06-25

### Added
- Web search tool with DuckDuckGo, Bing, and SerpAPI backends
- Token streaming for chat responses
- Model tiering (cheap model for routing, full model for analysis)
- BM25 sparse retrieval with RRF fusion
- MMR reranking (lambda=0.5)
- Intelligent text chunking

### Fixed
- WebSearch tool reliability improvements

## [4.0.0-rc.1] - 2026-06-23

### Fixed
- Full service recovery: Redis, Celery, Neo4j all operational
- Graph query performance improvements
- Chat performance fixes
- One-click startup script

### Added
- FAISS vector store rebuilt with 1125 vectors

## [4.0.0-beta] - 2026-06-21

### Added
- Graph analysis page
- Entity fusion pipeline
- Local graph fallback layer (when Neo4j is unavailable)
- WebSocket progress push for uploads

### Fixed
- Celery sync degradation fallback
- Upload progress polling robustness
- Service status indicators

## [3.0.0] - 2026-06-20

### Added
- FastAPI REST API with modular route splitting
- React 19 frontend with Vite 6, TypeScript, and Ant Design 5
- Celery async ETL pipeline with Redis broker/backend
- Redis three-tier cache (vector, graph, semantic)
- FAISS in-memory singleton with read-write lock
- Neo4j connection pool (50 connections)
- SHA256 content-addressed storage with version tracking
- 7-stage ETL pipeline with progress reporting and Pub/Sub
- WebSocket connection manager with Redis Pub/Sub integration
- Multi-agent pipeline: Router (6 intent classes), Domain Agents, Reviewer with confidence scoring
- Web search tool with multiple backends
- Knowledge graph model: 8 node types, 13 relationship types with evidence tracing
- Entity resolution engine (BGE-M3 embeddings + edit distance)
- Incremental graph merge with conflict detection and version tracking
- Template-based graph queries (competition, trend, event, entity)
- Leiden community detection with incremental updates
- Frontend with 6 pages: Login, Research Assistant, Knowledge Base, Dashboard, LLMOps, Admin
- GitHub Actions CI pipeline (lint, test, build, scan)
- 29 unit tests with Locust load testing scripts
- Prometheus + Grafana monitoring with alerting rules
- Docker multi-stage builds for web, API, and worker
- Complete deployment and operations manual

### Fixed
- Redis 50s blocking on Windows (lazy connect + automatic degradation)

## [2.0.0] - 2026-06-01

### Added
- Neo4j knowledge graph integration
- Multi-agent pipeline (Router, Domain Agents, Reviewer)
- FAISS vector search
- BM25 sparse retrieval
- 1125 entity relationship records injected into Neo4j
- Entity extraction pipeline
- Vectorization pipeline
- Full data injection pipeline
- Evidence traceability on all relationships
- Node label normalization (uppercase first letter)
- Relationship type cleanup

### Changed
- Data specification: enforced uppercase-first node labels and cleaned relationship types
