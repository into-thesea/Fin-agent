# ============================================================================
# 智能客服系统 Dockerfile (单端口: FastAPI 托管 API + 前端静态)
#
# 构建:  docker build -t kefu-app .
# 运行API: docker run -d --name kefu-api --env-file .env.prod \
#            -v /opt/kefu-data:/app/data -p 127.0.0.1:8001:8001 kefu-app
# 运行Worker: docker run -d --name kefu-worker --env-file .env.prod \
#            --network host kefu-app celery -A src.celery_app worker --loglevel=info
# ============================================================================

# ── 阶段 1: 构建前端 ──────────────────────────
FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ── 阶段 2: Python 运行时 ─────────────────────
FROM python:3.12-slim AS runtime
WORKDIR /app

# 系统依赖 (PDF 解析, FAISS, ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run.py .

# 前端静态 (由 FastAPI 托管)
COPY --from=frontend /app/frontend/dist ./frontend/dist

# 本地 bge 模型 (离线加载, 需随镜像或挂载)
COPY models/ ./models/

# 数据目录 (生产建议用 -v 挂载持久化, 不打包数据)
RUN mkdir -p data/output_analysis data/data_reports data/customer_faq logs

EXPOSE 8001

# 默认启动 API (单端口托管 API + 前端); Worker 用 docker run 覆盖 CMD
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
