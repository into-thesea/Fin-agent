"""
Fin-Agent 路径配置中心

所有数据目录路径集中管理，便于目录结构调整。
"""

import os

# 计算项目根目录：从 src/infra/paths.py 向上 3 级 → 项目根
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 数据子目录
DATA_REPORTS_DIR = os.path.join(DATA_DIR, "data_reports")
OUTPUT_ANALYSIS_DIR = os.path.join(DATA_DIR, "output_analysis")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# 关键文件路径
METADATA_DB_PATH = os.path.join(OUTPUT_ANALYSIS_DIR, "fin_agent_metadata.db")
CHUNKS_PROCESSED_PATH = os.path.join(OUTPUT_ANALYSIS_DIR, "chunks_processed.jsonl")
BM25_INDEX_PATH = os.path.join(OUTPUT_ANALYSIS_DIR, "bm25_index.pkl")

# 确保目录存在
for d in [DATA_DIR, DATA_REPORTS_DIR, OUTPUT_ANALYSIS_DIR, VECTOR_DB_DIR]:
    os.makedirs(d, exist_ok=True)
