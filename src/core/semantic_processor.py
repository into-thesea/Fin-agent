import glob
import os

from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.infra.jsonl_io import atomic_append

# --- 核心：自动定位项目根目录 ---
# __file__ 是 src/semantic_processor.py，其父目录的父目录即为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "output_analysis")
OUTPUT_PATH = os.path.join(INPUT_DIR, "chunks_processed.jsonl")

def process_semantic_chunks(md_file_path):
    if not os.path.exists(md_file_path):
        return []

    with open(md_file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 提取第一行作为报告标题
    first_line = md_content.split('\n')[0].replace('# 解析结果:', '').strip()
    report_name = first_line if first_line else os.path.basename(md_file_path)

    # 定义我们要提取的标题层级
    headers_to_split_on = [
        ("##", "PageNumber"),
        ("###", "SectionType"),
    ]

    # 初始化 LangChain 的 Markdown 标题切分器
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(md_content)

    processed_chunks = []
    for i, split in enumerate(md_header_splits):
        metadata = split.metadata
        page_num = metadata.get("PageNumber", "Unknown")
        section_type = metadata.get("SectionType", "Text")

        context_prefix = f"【{report_name}】 [{page_num} | {section_type}] "
        enhanced_content = context_prefix + split.page_content.strip()

        chunk_data = {
            "chunk_id": f"{report_name}_{i:04d}",
            "source": report_name,
            "page": page_num,
            "section": section_type,
            "content": enhanced_content,
            "raw_content": split.page_content.strip()
        }
        processed_chunks.append(chunk_data)

    return processed_chunks

if __name__ == "__main__":
    md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
    all_chunks = []

    print(f"在 {INPUT_DIR} 发现 {len(md_files)} 个 Markdown 文件，准备开始全量语义切分...")

    for md_file in md_files:
        if "chunks_processed" in md_file:
            continue
        print(f"正在切分: {os.path.basename(md_file)} ...")
        chunks = process_semantic_chunks(md_file)
        all_chunks.extend(chunks)

    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))

    for chunk in all_chunks:
        atomic_append(OUTPUT_PATH, chunk)

    print("-" * 30)
    print(f"全量切分完成！汇总生成 {len(all_chunks)} 个语义区块。")
    print(f"结果已保存至: {OUTPUT_PATH}")
