
import os

# 尝试多个 PDF 解析库
try:
    import pdfplumber  # noqa: F401
    _PDF_ENGINE = "pdfplumber"
except ImportError:
    try:
        import fitz  # PyMuPDF  # noqa: F401
        _PDF_ENGINE = "fitz"
    except ImportError:
        _PDF_ENGINE = None

# --- 核心：自动定位项目根目录 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.infra.paths import OUTPUT_ANALYSIS_DIR
    OUTPUT_DIR = OUTPUT_ANALYSIS_DIR
except ImportError:
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_analysis")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def extract_chunks_from_pdf(pdf_path):
    """从单个 PDF 提取文本分块并保存元数据"""
    file_name = os.path.basename(pdf_path)
    chunks = []

    if not _PDF_ENGINE:
        print("❌ 无可用 PDF 解析库 (请安装 pdfplumber 或 PyMuPDF)")
        return []

    print(f"解析中 [{_PDF_ENGINE}]: {file_name}...")
    try:
        if _PDF_ENGINE == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 50:
                        continue
                    chunk_id = f"{file_name}_{i:04d}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source": file_name,
                        "page": i + 1,
                        "content": text.strip(),
                    })
        else:
            # PyMuPDF 回退
            import fitz
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                text = page.get_text()
                if not text or len(text.strip()) < 50:
                    continue
                chunk_id = f"{file_name}_{i:04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": file_name,
                    "page": i + 1,
                    "content": text.strip(),
                })
            doc.close()

        # 记录到全量 chunks 文件
        chunk_file = os.path.join(OUTPUT_DIR, "chunks_processed.jsonl")
        from src.infra.jsonl_io import atomic_append
        for c in chunks:
            atomic_append(chunk_file, c)

        print(f"解析完成，生成 {len(chunks)} 个分块。")
        return chunks
    except Exception as e:
        print(f"解析失败 {file_name}: {e}")
        return []

if __name__ == "__main__":
    # 保持向后兼容的 CLI 运行模式
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data_reports")
    import glob
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    for f in pdf_files:
        extract_chunks_from_pdf(f)
