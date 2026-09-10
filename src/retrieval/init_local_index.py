import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 配置
VERSION = "v1_base"
RAW_DATA_PATH = "output_analysis/chunks_processed.jsonl" # 这里根据你本地实际路径调整
INDEX_DIR = f"vector_db/{VERSION}"
os.makedirs(INDEX_DIR, exist_ok=True)

def init_vector_index():
    print(f"🚀 正在为 {VERSION} 构建本地初始化向量索引...")
    
    # 修改：指向本地已有的模型路径
    local_model_path = "/mnt/d/aiproject/models/models--BAAI--bge-base-zh-v1.5/snapshots/"
    model_name = "BAAI/bge-base-zh-v1.5"
    # 自动检测快照目录
    snap_dir = "/mnt/d/aiproject/models/models--BAAI--bge-base-zh-v1.5/snapshots"
    if os.path.exists(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        if snaps:
            local_model_path = os.path.join(snap_dir, snaps[-1])
    if os.path.exists(local_model_path):
        print(f"📦 发现本地模型，正在离线加载: {local_model_path}")
        model = SentenceTransformer(local_model_path)
    else:
        print("🌐 未发现本地模型，尝试从网络下载...")
        model = SentenceTransformer(model_name)
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ 错误: 找不到原始分块文件 {RAW_DATA_PATH}")
        return

    # 加载已有的分块 (可以只加载前 N 个作为热启动测试)
    texts = []
    metadata = []
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            texts.append(data["content"])
            metadata.append({
                "chunk_id": data["chunk_id"],
                "file_name": data.get("file_name", "unknown"),
                "page_num": data.get("page_num", 0),
                "content": data["content"][:200] # 摘要存储
            })
            if i >= 429: # 对应你目前跑完的进度
                break
    
    print(f"📐 正在向量化 {len(texts)} 个文本块...")
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    
    # 写入 FAISS
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    
    # 写入元数据
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    
    print(f"✅ 向量索引构建完成！存储于: {INDEX_DIR}")

if __name__ == "__main__":
    init_vector_index()
