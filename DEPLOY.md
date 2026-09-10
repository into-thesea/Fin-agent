# Fin-Agent 4.0 — 部署与运维手册

> 工业级图增强金融智能体 · 企业级部署指南

---

## 目录

1. [环境要求](#1-环境要求)
2. [快速启动 (开发)](#2-快速启动-开发)
3. [生产部署 (Docker)](#3-生产部署-docker)
4. [Kubernetes 部署](#4-kubernetes-部署)
5. [配置说明](#5-配置说明)
6. [监控与告警](#6-监控与告警)
7. [性能调优](#7-性能调优)
8. [故障排查](#8-故障排查)
9. [运维脚本](#9-运维脚本)

---

## 1. 环境要求

### 最低配置 (开发/演示)

| 资源 | 要求 |
|------|------|
| CPU | 4 核 |
| 内存 | 8 GB |
| 磁盘 | 20 GB |
| Docker | 24.0+ |
| Docker Compose | 2.20+ |

### 推荐配置 (生产)

| 资源 | 要求 |
|------|------|
| CPU | 8 核 |
| 内存 | 32 GB |
| 磁盘 | 100 GB (SSD) |
| GPU | 可选 (加速向量推理) |
| 网络 | 100 Mbps+ |

### 依赖服务

| 服务 | 版本 | 用途 |
|------|------|------|
| Neo4j AuraDB | 5.x (云服务) | 知识图谱 |
| Redis | 7.x | 缓存/队列 |
| PostgreSQL | 16.x (可选) | 生产元数据库 |

---

## 2. 快速启动 (开发)

### 2.1 环境准备

```bash
# 1. 克隆项目
git clone <repo-url> fin-agent
cd fin-agent

# 2. Python 环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.template .env
# 编辑 .env 填入真实密钥

# 5. 检查配置
python scripts/env_manager.py check
```

### 2.2 本地启动 (无 Docker)

```bash
# 终端 1: 启动 Streamlit UI
streamlit run src/app.py --server.port 8501

# 终端 2: 启动 FastAPI (可选)
uvicorn src.api.main:app --port 8000

# 终端 3: 启动 Celery Worker (可选，需要 Redis)
celery -A src.celery_app worker --loglevel=info
```

访问: http://localhost:8501

### 2.3 前置数据准备

```bash
# 1. 解析 PDF → 语义分块
python src/main.py

# 2. 语义切分
python src/semantic_processor.py

# 3. 构建向量索引
python src/vector_indexer.py --rebuild

# 4. 实体提取 (需要 API Key)
python src/api_entity_extractor.py

# 5. 图谱注入
python src/graph_uploader.py

# 6. 社区检测与摘要
python src/community_detector.py
python src/community_summarizer.py
```

---

## 3. 生产部署 (Docker)

### 3.1 一键启动

```bash
# 完整栈启动
docker-compose up -d

# 仅启动核心服务 (无前端)
docker-compose up -d web api worker redis

# 查看日志
docker-compose logs -f api worker
```

### 3.2 服务访问

| 服务 | 地址 | 说明 |
|------|------|------|
| React SPA 前端 | http://localhost:80 | Nginx 承载 |
| Streamlit UI | http://localhost:8501 | 备用前端 |
| FastAPI 网关 | http://localhost:8000 | API 端点 |
| API 文档 | http://localhost:8000/docs | Swagger UI |

### 3.3 启动顺序

```
Redis (先) → API (依赖 Redis) → Worker (依赖 Redis) → Frontend (依赖 API)
```

### 3.4 优雅停止

```bash
docker-compose down

# 保留数据卷
docker-compose down -v  # 谨慎: 会删除 Redis 数据
```

### 3.5 扩缩容

```bash
# 增加 Worker 数量
docker-compose up -d --scale worker=4

# 增加 API 实例
docker-compose up -d --scale api=3
```

---

## 4. Kubernetes 部署

### 4.1 前置条件

- Kubernetes 1.28+
- Helm 3
- Ingress Controller (nginx-ingress)
- cert-manager (可选, HTTPS)

### 4.2 部署文件

```yaml
# deploy/k8s/deployment.yaml 示例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fin-agent-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fin-agent-api
  template:
    metadata:
      labels:
        app: fin-agent-api
    spec:
      containers:
        - name: api
          image: fin-agent:api-latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: fin-agent-secrets
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
```

### 4.3 Helm 部署 (推荐)

```bash
# 使用 Helm Chart 一键部署
helm install fin-agent ./deploy/charts/fin-agent \
  --set api.replicas=3 \
  --set worker.replicas=2 \
  --set redis.enabled=true \
  --set ingress.enabled=true \
  --set secrets.neo4jPassword=<password>
```

---

## 5. 配置说明

### 5.1 核心环境变量 (.env)

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | — | DeepSeek API 密钥 |
| `NEO4J_URI` | ✅ | — | Neo4j 连接地址 |
| `NEO4J_USERNAME` | ✅ | — | Neo4j 用户名 |
| `NEO4J_PASSWORD` | ✅ | — | Neo4j 密码 |
| `REDIS_HOST` | ❌ | localhost | Redis 主机 |
| `REDIS_PORT` | ❌ | 6379 | Redis 端口 |
| `LOG_LEVEL` | ❌ | INFO | 日志级别 |
| `LOG_FORMAT` | ❌ | json | 日志格式 (json/text) |
| `MODEL_CACHE_DIR` | ❌ | ./models | 模型缓存目录 |

### 5.2 Docker Secrets (生产)

```bash
# 使用 Docker Secrets 替代环境变量
echo "my-neo4j-password" | docker secret create neo4j_password -

# docker-compose 中引用
secrets:
  neo4j_password:
    external: true

services:
  api:
    secrets:
      - neo4j_password
```

---

## 6. 监控与告警

### 6.1 启动监控栈

```bash
docker-compose \
  -f docker-compose.yml \
  -f deploy/monitoring/docker-compose.monitoring.yml \
  up -d
```

### 6.2 监控端点

| 服务 | 地址 | 默认凭据 |
|------|------|---------|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / finadmin |

### 6.3 关键指标

| 指标 | 告警阈值 | 严重度 |
|------|---------|--------|
| API 可用性 | < 100% (1m) | Critical |
| P95 延迟 | > 5s (5m) | Warning |
| 5xx 错误率 | > 5% (5m) | Critical |
| Redis 连接数 | > 100 | Warning |
| 缓存命中率 | < 30% (10m) | Warning |
| 磁盘空间 | < 10% | Critical |

### 6.4 日志采集 (ELK)

```bash
# 启动 ELK Stack
docker-compose -f deploy/monitoring/docker-compose.elk.yml up -d

# 日志格式 (JSON)
{
  "timestamp": "2026-06-20T10:30:00.123Z",
  "level": "INFO",
  "logger": "src.retriever",
  "trace_id": "req_abc123",
  "message": "三路召回: 比亚迪2024年营收...",
  "latency_ms": 2340
}
```

---

## 7. 性能调优

### 7.1 数据库优化

```cypher
// Neo4j 索引 (自动创建)
CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.name);
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
  FOR (n:Company|Metric|Segment) ON EACH [n.name, n.aliases];
```

### 7.2 缓存策略

| 缓存类型 | TTL | 失效条件 |
|---------|-----|---------|
| 向量检索 | 5 min | 新文档入库 |
| 图谱查询 | 10 min | 图谱更新 |
| 语义缓存 | 1 hour | 新知识入库 |
| ETL 状态 | 1 hour | 任务完成 |

### 7.3 连接池配置

| 组件 | 参数 | 推荐值 |
|------|------|--------|
| Neo4j | `max_connection_pool_size` | 50 |
| Redis | `socket_timeout` | 3s |
| Celery | `worker_concurrency` | CPU 核数 × 2 |

### 7.4 索引重建

```bash
# 每周自动重建 (cron)
0 3 * * 1 cd /app && python src/vector_indexer.py --rebuild

# 手动触发
curl -X POST http://localhost:8000/api/v1/admin/rebuild-index
```

---

## 8. 故障排查

### 8.1 常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| API 返回 503 | Neo4j 或 Redis 不可用 | `docker-compose ps` 检查服务状态 |
| 上传卡住 | Celery Worker 未运行 | `docker-compose logs worker` 检查 |
| ETL 失败 | API Key 过期 | `python scripts/env_manager.py check` |
| 响应慢 | FAISS 索引未加载 | 检查 `logs/fin-agent.log` 的 FAISS 日志 |
| 缓存不生效 | Redis 连接失败 | `docker-compose logs redis` 检查 |

### 8.2 健康检查

```bash
# API 健康
curl http://localhost:8000/health

# Redis 就绪
docker-compose exec redis redis-cli ping

# Celery Worker 状态
celery -A src.celery_app status

# 日志实时查看
docker-compose logs -f --tail=100 api
```

### 8.3 数据恢复

```bash
# 1. 保留数据卷
docker-compose down
# 2. 重新启动
docker-compose up -d
# 3. 从备份恢复
python scripts/restore.py --backup ./backups/20260620/
```

---

## 9. 运维脚本

| 脚本 | 用途 |
|------|------|
| `scripts/env_manager.py` | 环境变量检查/加密 |
| `scripts/cache_stats.py` | 缓存统计与清理 (待实现) |
| `scripts/health_report.py` | 全量健康报告 (待实现) |
| `scripts/backup.py` | 数据备份 (待实现) |
| `scripts/migrate_schema.py` | 数据库迁移 (待实现) |

### 备份策略

```bash
# 数据持久化目录
./data_reports/        # PDF 源文件
./output_analysis/     # 分块/实体/索引
./models/              # ML 模型
./logs/                # 日志

# 推荐备份方案
tar -czf backup_$(date +%Y%m%d).tar.gz \
  data_reports/ output_analysis/ \
  deploy/monitoring/prometheus.yml \
  fin_agent_metadata.db
```

---

> **最后更新**: 2026-06-20
> **项目状态**: ✅ Phase 1-5 完成, 企业级全栈部署就绪
