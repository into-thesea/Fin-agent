"""
智能文档分块 (Smart Chunking) — 章节感知语义分块

改进策略:
  1. 章节标题检测 — 识别中文/英文财务报告的层级标题 (一、二 → 1.1 → (1))
  2. 章节树构建 — 按标题层级确定每节覆盖范围
  3. 标题前缀注入 — 每个块内容前加上 "[大章节 >> 子章节]" 路径
  4. 自适应细切 — 长章节用 RecursiveCharacterTextSplitter 切到 300 字
  5. 无标题回退 — 检测不到标题时保持原滑动窗口分块

适配:
  - bge-base-zh-v1.5 最大 512 tokens, 中文约 1.4 token/字符
  - 默认分块 350 字符 (留 ~50 字符给标题前缀)

用法:
    from src.core.smart_chunk import smart_chunk_pdf
    chunks = smart_chunk_pdf("path/to/report.pdf")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# bge-base-zh-v1.5 最大 512 tokens, 中文约 1.4 token/字符 → 安全上限 350 字符
DEFAULT_CHUNK_SIZE = 350   # 每块目标总字符数 (含前缀)
DEFAULT_OVERLAP = 70       # 块间重叠字符数
MAX_CHUNK_CONTENT = 300    # 纯内容上限 (前缀 ~50 字符)

# ──────────────────────────────────────────────
# 章节标题检测模式
# ──────────────────────────────────────────────

# 被识别为标题后还需过滤的负向条件
HEADER_EXCLUDE = re.compile(
    r"[。，；！？…]"   # 句尾有标点 → 完整句子
    r"|^\d{4}[-/]\d{1,2}[-/]\d{1,2}"            # 日期行
    r"|^\d+$|^第\d+页$"                          # 纯数字 / 页码
)

# 中英文标题模式 (按层级优先级)
HEADER_RULES = [
    # Level 1 — 章节级
    (re.compile(r"^第[一二三四五六七八九十百千零\d]+[章节篇]"), 1),
    (re.compile(r"^[一二三四五六七八九十百千]+[、．]"), 1),
    (re.compile(r"^(重要提示|释义|公司简介|管理层讨论|"
                r"公司治理|核心竞争力|研发投入|"
                r"财务报告|审计报告|风险提示|"
                r"关联交易|股东信息|备查文件)"), 1),
    # Level 2 — 小节级
    (re.compile(r"^\d+\.\d+(\.[\d]+)?\s"), 2),
    (re.compile(r"^（[一二三四五六七八九十]+）"), 2),
    # Level 3 — 条项级
    (re.compile(r"^\(\d+\)"), 3),
]


@dataclass
class Section:
    """检测到的章节"""
    level: int                     # 1/2/3
    title: str                     # 标题文本
    start_pos: int                 # 在全文中的起始字符偏移
    end_pos: int = 0               # 终止偏移 (由后续标题或文末决定)
    children: list["Section"] = field(default_factory=list)
    parent: Optional["Section"] = None


# ──────────────────────────────────────────────
# 标题检测
# ──────────────────────────────────────────────

def _detect_headers(text: str) -> list[tuple[str, int, int]]:
    """
    扫描全文, 检测所有可能的章节标题.

    Returns:
        [(title_text, level, char_offset), ...]
    """
    headers: list[tuple[str, int, int]] = []
    lines = text.split("\n")
    offset = 0

    for line in lines:
        stripped = line.strip()
        line_len = len(stripped)
        if line_len < 2 or line_len > 80:
            offset += len(line) + 1
            continue

        # 负向排除
        if HEADER_EXCLUDE.search(stripped):
            offset += len(line) + 1
            continue

        for pattern, level in HEADER_RULES:
            if pattern.match(stripped):
                # level 2/3 标题过长 → 误判概率高
                if level >= 2 and len(stripped) > 20:
                    break
                headers.append((stripped, level, offset))
                break

        offset += len(line) + 1

    # 去重: 相同标题在 200 字符内重复 → 保留首次 (抑制目录页)
    deduped: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int]] = set()
    for title, level, pos in headers:
        key = (title, level)
        if key in seen:
            if deduped and abs(pos - deduped[-1][2]) < 200:
                continue  # 目录页重复, 跳过
        else:
            seen.add(key)
        deduped.append((title, level, pos))

    return deduped


# ──────────────────────────────────────────────
# 章节树
# ──────────────────────────────────────────────

def _build_section_tree(
    headers: list[tuple[str, int, int]], text_length: int
) -> list[Section]:
    """
    将扁平标题列表构建为章节树, 确定每节的覆盖范围.

    Args:
        headers: [(title, level, pos), ...]
        text_length: 全文总长度

    Returns:
        按出现顺序排列的顶层章节列表
    """
    roots: list[Section] = []
    stack: list[Section] = []

    for title, level, pos in headers:
        sec = Section(level=level, title=title, start_pos=pos)
        # 从栈顶弹出所有 level >= 当前 level 的章节
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            sec.parent = stack[-1]
            stack[-1].children.append(sec)
        else:
            roots.append(sec)
        stack.append(sec)

    # 确定终止位置: 下一个同级/上级章节 或 全文末尾
    def _set_end(secs: list[Section], next_start: int) -> None:
        for i, sec in enumerate(secs):
            sec.end_pos = secs[i + 1].start_pos if i + 1 < len(secs) else next_start
            _set_end(sec.children, sec.end_pos)

    _set_end(roots, text_length)
    return roots


def _section_path(sec: Section) -> str:
    """递归构建章节路径: 大章节 >> 子章节"""
    parts: list[str] = []
    cur: Optional[Section] = sec
    while cur:
        parts.append(cur.title)
        cur = cur.parent
    return " >> ".join(reversed(parts))


def _count_nodes(sections: list[Section]) -> int:
    """递归计算章节树节点总数"""
    total = 0
    for s in sections:
        total += 1 + _count_nodes(s.children)
    return total


# ──────────────────────────────────────────────
# 获取章节独占内容 (排除子章节)
# ──────────────────────────────────────────────

def _sec_own_range(sec: Section) -> tuple[int, int]:
    """
    返回该章节**独占**的文本范围 [start, end), 排除子章节区域.

    父章节只拥有第一个子章节开始之前的文本 (前言/概述),
    子章节内容由子章节自己负责切分.
    """
    start = sec.start_pos
    if sec.children:
        end = sec.children[0].start_pos
    else:
        end = sec.end_pos
    return start, end


# ──────────────────────────────────────────────
# 页 → 位置映射
# ──────────────────────────────────────────────

def _build_page_index(
    pages: list[dict],
) -> tuple[str, list[tuple[int, int, int]]]:
    """
    拼接全文 + 构建页索引.

    Returns:
        (full_text, [(start_pos, end_pos, page_num), ...])
    """
    parts: list[str] = []
    breaks: list[tuple[int, int, int]] = []
    pos = 0
    for p in pages:
        text: str = p["text"].strip()
        parts.append(text)
        breaks.append((pos, pos + len(text), p["page"]))
        pos += len(text) + 1  # +1 保留边界
    return "\n".join(parts), breaks


def _pos_to_page(pos: int, breaks: list[tuple[int, int, int]]) -> int:
    """根据字符偏移定位页码"""
    for start, end, page in breaks:
        if pos <= end:
            return page
    return breaks[-1][2] if breaks else 1


# ──────────────────────────────────────────────
# 核心: 章节感知分块
# ──────────────────────────────────────────────

def _chaptered_chunk(
    pages: list[dict],
    file_name: str,
    chunk_size: int,
    overlap: int,
    max_chunk_content: int = MAX_CHUNK_CONTENT,
) -> list[dict]:
    """
    章节感知语义分块.

    流程:
      1. 拼接全文 → 检测标题 → 构建章节树
      2. 逐章节提取独占内容
      3. 短章节 → 完整保留, 长章节 → RecursiveCharacterTextSplitter 细切
      4. 每块注入 "[章节路径]" 前缀

    Args:
        max_chunk_content: 单块纯内容字符上限 (不含 "[章节路径]" 前缀)。
            由可编辑配置传入; 不传时用模块常量, 与加这个形参之前的行为一致。
    """
    full_text, page_breaks = _build_page_index(pages)

    # 检测标题
    headers = _detect_headers(full_text)
    if len(headers) < 2:
        logger.info("未检测到章节标题 (%d 个), 回退到滑动窗口分块", len(headers))
        return []

    # 构建章节树
    sections = _build_section_tree(headers, len(full_text))
    logger.info(
        "章节检测: %d 个顶层章节, %d 个总节点",
        len(sections), _count_nodes(sections),
    )

    chunks: list[dict] = []
    chunk_counter = 0
    content_size = min(max_chunk_content, chunk_size - 50)  # 留 ~50 给前缀

    def _process_range(sec: Section, start: int, end: int) -> None:
        """对 [start, end) 范围内的文本做分块."""
        nonlocal chunk_counter
        text = full_text[start:end].strip()
        if len(text) < 30:
            return

        prefix = f"[{_section_path(sec)}]"

        if len(text) <= content_size:
            # 短内容 → 完整保留
            chunks.append({
                "chunk_id": f"{file_name}_{_pos_to_page(start, page_breaks):04d}_{chunk_counter:02d}",
                "source": file_name,
                "page": _pos_to_page(start, page_breaks),
                "section": _section_path(sec),
                "content": f"{prefix} {text}",
            })
            chunk_counter += 1
            return

        # 长内容 → 细切
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=content_size,
                chunk_overlap=overlap,
                separators=["\n", "。", "；", ",", " ", ""],
                keep_separator=False,
            )
            sub_texts = splitter.split_text(text)
        except ImportError:
            sub_texts = []
            pos = 0
            while pos < len(text):
                end_pos = min(pos + content_size, len(text))
                sub_texts.append(text[pos:end_pos])
                pos = end_pos - overlap if end_pos < len(text) else end_pos

        for t in sub_texts:
            t = t.strip()
            if len(t) < 30:
                continue
            chunks.append({
                "chunk_id": f"{file_name}_{_pos_to_page(start, page_breaks):04d}_{chunk_counter:02d}",
                "source": file_name,
                "page": _pos_to_page(start, page_breaks),
                "section": _section_path(sec),
                "content": f"{prefix} {t}",
            })
            chunk_counter += 1

    def _walk(secs: list[Section]) -> None:
        """遍历章节树, 对每个节点处理其独占内容 + 递归子节点."""
        for sec in secs:
            own_start, own_end = _sec_own_range(sec)
            if own_end > own_start:
                _process_range(sec, own_start, own_end)
            _walk(sec.children)

    _walk(sections)

    logger.info(
        "语义分块完成: %s -> %d 块 (%d 章节, size=%d)",
        file_name, len(chunks), _count_nodes(sections), chunk_size,
    )
    return chunks


# ──────────────────────────────────────────────
# 原始滑动窗口回退 (保留但更新默认参数)
# ──────────────────────────────────────────────

def _sliding_window_chunks(
    pages: list[dict],
    file_name: str,
    chunk_size: int,
    overlap: int,
    no_fallback: bool = False,
) -> list[dict]:
    """
    跨页滑动窗口分块 (原始算法).

    当章节检测不到或 no_fallback=False 时使用.
    """
    # 拼接所有页面
    page_breaks: list[tuple[int, int, int]] = []
    full_text_parts: list[str] = []
    pos = 0
    for p in pages:
        t = p["text"]
        full_text_parts.append(t)
        page_breaks.append((pos, pos + len(t), p["page"]))
        pos += len(t) + 1
    full_text_str = "\n".join(full_text_parts)

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            keep_separator=False,
        )
        texts = splitter.split_text(full_text_str)
    except ImportError:
        texts = []
        start = 0
        while start < len(full_text_str):
            end = min(start + chunk_size, len(full_text_str))
            texts.append(full_text_str[start:end])
            start = end - overlap if end < len(full_text_str) else end

    chunks: list[dict] = []
    for idx, t in enumerate(texts):
        t = t.strip()
        if len(t) < 30:
            continue
        pos_in_full = full_text_str.find(t[:50])
        page_num = _pos_to_page(pos_in_full, page_breaks)
        chunks.append({
            "chunk_id": f"{file_name}_{page_num:04d}_{idx:02d}",
            "source": file_name,
            "page": page_num,
            "content": t,
        })
    return chunks


# ──────────────────────────────────────────────
# 公共入口
# ──────────────────────────────────────────────

def smart_chunk_pdf(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_chunk_content: int = MAX_CHUNK_CONTENT,
) -> list[dict]:
    """
    智能 PDF 分块 — 章节感知语义分块 + 滑动窗口回退.

    优先尝试章节检测分块:
      - 注入 "[章节路径]" 前缀提升 embedding 区分度
      - 长章节按句/段细切到 ~300 字
    检测不到标题时自动回退到跨页滑动窗口.

    三个参数都来自可编辑配置(data/kb_settings.json 的 chunking 段),
    由 pipeline_manager 传入; 不传时用模块常量, 行为与加形参之前一致。

    Args:
        pdf_path: PDF 文件路径
        chunk_size: 每块总字符数上限 (含前缀, 默认 350)
        overlap: 块间重叠字符数 (默认 70)
        max_chunk_content: 单块纯内容字符上限 (不含前缀, 默认 300)

    Returns:
        [{"chunk_id", "source", "page", "content"[, "section"]}, ...]
    """

    file_name = os.path.basename(pdf_path)
    pages = _extract_pages(pdf_path)
    if not pages:
        return []

    # 1) 先试章节感知分块
    chunks = _chaptered_chunk(pages, file_name, chunk_size, overlap, max_chunk_content)

    # 2) 检测不到章节 → 回退到滑动窗口
    if not chunks:
        logger.info("章节检测不到位, 回退到滑动窗口分块: %s", file_name)
        chunks = _sliding_window_chunks(pages, file_name, chunk_size, overlap)

    logger.info(
        "智能分块完成: %s -> %d 块 (size=%d, overlap=%d, content<=%d)",
        file_name, len(chunks), chunk_size, overlap, max_chunk_content,
    )
    return chunks


# ──────────────────────────────────────────────
# PDF 文本提取
# ──────────────────────────────────────────────

def _extract_pages(pdf_path: str) -> list[dict]:
    """提取 PDF 所有页面的文本, 返回 [{"page": int, "text": str}, ...]"""
    pages: list[dict] = []

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) >= 30:
                    pages.append({"page": i + 1, "text": text.strip()})
        if pages:
            return pages
    except Exception as e:
        logger.warning("pdfplumber 提取失败, 尝试 fitz 回退: %s", e)

    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and len(text.strip()) >= 30:
                pages.append({"page": i + 1, "text": text.strip()})
        doc.close()
    except Exception as e:
        logger.debug("fitz 提取失败 (可忽略): %s", e)

    if not pages:
        logger.warning("无法提取 PDF 文本: %s", pdf_path)
    return pages


# ──────────────────────────────────────────────
# 旧块转换工具
# ──────────────────────────────────────────────

def convert_legacy_chunks(
    legacy_chunks: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    将旧格式的每页一块转换为新的智能分块格式.

    用于在 ETL 流水线中对已解析的旧块做重分块.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            keep_separator=False,
        )
    except ImportError:
        logger.error("convert_legacy_chunks 需要 langchain 库")
        return legacy_chunks

    new_chunks: list[dict] = []
    for chunk in legacy_chunks:
        texts = splitter.split_text(chunk.get("content", ""))
        for idx, t in enumerate(texts):
            t = t.strip()
            if len(t) < 30:
                continue
            new_chunks.append({
                "chunk_id": f"{chunk['chunk_id']}_{idx:02d}",
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "content": t,
            })
    return new_chunks
