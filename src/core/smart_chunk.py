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

# 子块目标字符数 —— 入索引的就是子块, 所以这才是"单块内容上限"那个旋钮的真身。
CHILD_CHUNK_SIZE = 150
CHILD_OVERLAP = 30         # 子块间重叠字符数
# 设置页上的 max_chunk_content 就是它, 不再各留一个数 (两处必然漂移)
MAX_CHUNK_CONTENT = CHILD_CHUNK_SIZE

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
                r"关联交易|股东信息|备查文件|"
                # 理财产品说明书常用章节标题
                r"产品概述|产品说明|风险揭示|风险提示书|"
                r"信息披露|信息批露|投资范围|产品投向|"
                r"费用说明|产品相关费用|提前终止|收益测算|"
                r"产品收益|申购条件|认购条件|特别提示|"
                r"投资者权益|适合的投资者|内部风险评级|"
                r"产品不成立|业绩比较基准|产品净值|"
                r"理财产品托管|风险示例|计算示例)"), 1),
    # Level 2 — 小节级
    (re.compile(r"^\d+\.\d+(\.[\d]+)?\s"), 2),
    (re.compile(r"^（[一二三四五六七八九十]+）"), 2),
    # Level 3 — 条项级
    (re.compile(r"^\(\d+\)"), 3),
]

# 标题长度上限: 原先 80 对表格布局 PDF 太严 (标题+内容同一行),
# 放宽到 120; 超过 120 的长行尝试按多空格拆分提取标题前缀。
MAX_HEADER_LINE_LEN = 120


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

    对表格布局 PDF (标题与内容同一行, 用多空格分隔), 尝试拆分提取标题前缀.
    例如 "产品名称  聚宝财富专享8号  人民币理财产品" → 提取 "产品名称" 作为标题.

    Returns:
        [(title_text, level, char_offset), ...]
    """
    headers: list[tuple[str, int, int]] = []
    lines = text.split("\n")
    offset = 0

    for line in lines:
        stripped = line.strip()
        line_len = len(stripped)
        if line_len < 2:
            offset += len(line) + 1
            continue

        # 候选标题列表: 原行 + 从长行中拆分出的前缀
        candidates: list[tuple[str, int]] = [(stripped, 0)]  # (text, offset_in_line)

        # 长行 (>80) 可能是表格布局 "标题  内容  内容", 尝试按多空格拆分
        if line_len > 80:
            parts = re.split(r'\s{2,}', stripped)
            if len(parts) >= 2 and 2 <= len(parts[0]) <= 15:
                # 第一部分像标题 (短、无句末标点), 加入候选
                first = parts[0].strip()
                if not HEADER_EXCLUDE.search(first):
                    candidates.append((first, 0))

        for cand_text, cand_offset in candidates:
            if len(cand_text) > MAX_HEADER_LINE_LEN:
                continue

            # 负向排除
            if HEADER_EXCLUDE.search(cand_text):
                continue

            for pattern, level in HEADER_RULES:
                if pattern.match(cand_text):
                    # level 2/3 标题过长 → 误判概率高
                    if level >= 2 and len(cand_text) > 20:
                        break
                    headers.append((cand_text, level, offset + cand_offset))
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

def _chaptered_chunk(pages: list[dict], file_name: str) -> list[dict]:
    """
    章节感知语义分块 —— 只产出**父块**, 一个章节一块, 不设长度上限。

    流程:
      1. 拼接全文 → 检测标题 → 构建章节树
      2. 逐章节提取独占内容, 注入 "[章节路径]" 前缀

    细切由 build_parent_child_chunks 统一做 (子块才入索引)。切分参数(chunk_size /
    overlap / max_chunk_content)都由那一层接, 这里收了也是空转 —— 曾经收着,
    于是设置页上的参数改了没反应。
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

    def _process_range(sec: Section, start: int, end: int) -> None:
        """对 [start, end) 范围内的文本做分块.

        章节感知只按章节切, 每个章节完整内容作为一个父块.
        长章节的细切由 build_parent_child_chunks 统一做父子分块,
        避免两层切分叠加导致重复 section.
        """
        nonlocal chunk_counter
        text = full_text[start:end].strip()
        if len(text) < 30:
            return

        prefix = f"[{_section_path(sec)}]"
        chunks.append({
            "chunk_id": f"{file_name}_{_pos_to_page(start, page_breaks):04d}_{chunk_counter:02d}",
            "source": file_name,
            "page": _pos_to_page(start, page_breaks),
            "section": _section_path(sec),
            "content": f"{prefix} {text}",
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
        "语义分块完成: %s -> %d 个父块 (%d 章节)",
        file_name, len(chunks), _count_nodes(sections),
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
# 父子文档分块 (Parent-Child Chunking)
# ──────────────────────────────────────────────
# 父块 = 章节完整内容 (不入检索索引, 只做上下文容器)
# 子块 = 句子级小chunk (入Milvus/BM25检索, 上限见顶部 CHILD_CHUNK_SIZE)
# 检索命中子块后, 通过 parent_id 返回父块完整上下文


def _split_sentences(text: str) -> list[str]:
    """按中文句子边界切分 (句号/问号/感叹号/分号/换行)"""
    parts = re.split(r'(?<=[。！？；\n])', text)
    return [p for p in parts if p.strip()]


def _content_hash(text: str) -> str:
    """内容哈希, 用于去重和变更检测"""
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def split_parent_into_children(
    parent: dict,
    max_chunk_content: int = CHILD_CHUNK_SIZE,
) -> list[dict]:
    """把一个父chunk按句子边界切分为多个子chunk

    子块保留父块的章节前缀, 保证检索时上下文完整。
    适配 PDF 章节感知分块的前缀格式: "[章节路径] 内容"

    Args:
        max_chunk_content: 子块目标字符数 —— 设置页上那个"单块内容上限"就接到这里。
            接线之前它是空转的: 形参一路传到 _chaptered_chunk 却没人用, 子块死守
            常量 150, 于是页面改 180 什么都不会发生。
    """
    parent_id = parent["chunk_id"]
    full_content = parent.get("content", "")

    # 提取章节前缀 (如 "[产品概述 >> 风险提示] ")
    prefix_match = re.match(r'^(\[[^\]]+\]\s*)', full_content)
    title_prefix = prefix_match.group(1) if prefix_match else ""
    body = full_content[len(title_prefix):] if title_prefix else full_content

    sentences = _split_sentences(body)
    if not sentences:
        return []

    # 整个父块内容不超过子块大小 → 只生成一个子块
    if len(full_content) <= max_chunk_content:
        return [{
            "chunk_id": f"{parent_id}_c1",
            "source": parent.get("source", ""),
            "page": parent.get("page", 1),
            "section": parent.get("section", ""),
            "content": full_content,
            "content_hash": _content_hash(full_content),
            "chunk_type": "child",
            "parent_id": parent_id,
        }]

    # 累积句子生成子块
    children = []
    current = ""
    child_idx = 0

    for i, sent in enumerate(sentences):
        if not current:
            current = sent
        elif len(title_prefix) + len(current) + len(sent) <= max_chunk_content:
            current += sent
        else:
            # 当前子块已满, 保存
            child_idx += 1
            child_content = title_prefix + current
            children.append({
                "chunk_id": f"{parent_id}_c{child_idx}",
                "source": parent.get("source", ""),
                "page": parent.get("page", 1),
                "section": parent.get("section", ""),
                "content": child_content,
                "content_hash": _content_hash(child_content),
                "chunk_type": "child",
                "parent_id": parent_id,
            })
            # overlap: 下一个子块以上一个子块的最后一句开头
            last_sent = sentences[i - 1] if i > 0 else ""
            current = last_sent + sent if last_sent and len(last_sent) < CHILD_OVERLAP else sent

    # 最后一个子块
    if current:
        child_idx += 1
        child_content = title_prefix + current
        children.append({
            "chunk_id": f"{parent_id}_c{child_idx}",
            "source": parent.get("source", ""),
            "page": parent.get("page", 1),
            "section": parent.get("section", ""),
            "content": child_content,
            "content_hash": _content_hash(child_content),
            "chunk_type": "child",
            "parent_id": parent_id,
        })

    return children


def build_parent_child_chunks(
    parent_chunks: list[dict],
    max_chunk_content: int = CHILD_CHUNK_SIZE,
) -> list[dict]:
    """把所有父chunk转换为 父子结构 的完整chunk列表

    返回: [父块1, 子块1a, 子块1b, 父块2, 子块2a, ...]
    父块标记 chunk_type=parent, 子块标记 chunk_type=child+parent_id
    """
    result = []
    for p in parent_chunks:
        parent_with_type = dict(p)
        parent_with_type["chunk_type"] = "parent"
        if "content_hash" not in parent_with_type:
            parent_with_type["content_hash"] = _content_hash(p.get("content", ""))
        result.append(parent_with_type)

        children = split_parent_into_children(p, max_chunk_content)
        result.extend(children)

    return result


def indexable_chunks(chunks: list[dict]) -> list[dict]:
    """挑出该进检索索引的块 —— 父块只作上下文, 不入索引。

    索引侧有 3 条路径, 原先各自硬编码这个判断: 全量向量重建 / BM25 重建 /
    增量 add_chunks_to_index。第三条漏了, 于是审核通过一份上传文档就把父块也
    灌进 Milvus —— 整章大块跟子块抢 top-k, 直到下次全量重建才被清掉。
    收敛到这一处, 以后再加索引路径也漏不掉。

    为什么父块不该进索引: 实测 417 个父块里 228 个(55%)与某个子块**逐字相同**
    (章节太短没切开), 进去就是成对的重复向量; 长父块最长 8744 字, 远超
    bge-base-zh-v1.5 的 512 token 上下文, 会被截断成"只看开头一段"的失真向量。
    父块内容在回答时由 retriever._load_parent_map() 从分块文件按 id 取, 不依赖索引。
    """
    return [c for c in chunks if c.get("chunk_type") != "parent"]


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
      - 章节 → 父块 (整章, 不入索引), 句子级 → 子块 (入索引)
    检测不到标题时自动回退到跨页滑动窗口.

    三个参数都来自可编辑配置(data/kb_settings.json 的 chunking 段), 由
    pipeline_manager 传入; 不传时用模块常量。注意**生效范围不同** ——
    章节路径上 chunk_size/overlap 用不到(父块整章、子块按 max_chunk_content),
    它们只作用于无章节时的滑动窗口回退。

    Args:
        pdf_path: PDF 文件路径
        chunk_size: 每块总字符数上限 (含前缀, 默认 350; 仅滑动窗口回退用)
        overlap: 块间重叠字符数 (默认 70; 仅滑动窗口回退用)
        max_chunk_content: 子块内容上限 (默认 150)

    Returns:
        [{"chunk_id", "source", "page", "content"[, "section"]}, ...]
    """

    file_name = os.path.basename(pdf_path)
    pages = _extract_pages(pdf_path)
    if not pages:
        return []

    # 1) 先试章节感知分块
    chunks = _chaptered_chunk(pages, file_name)

    # 2) 章节感知成功 → 父子文档分块 (父块=章节完整内容, 子块=句子级入检索)
    if chunks:
        n_parents = len(chunks)
        chunks = build_parent_child_chunks(chunks, max_chunk_content)
        n_children = len(chunks) - n_parents
        logger.info(
            "父子分块: %s -> %d 父块 + %d 子块 = %d 块",
            file_name, n_parents, n_children, len(chunks),
        )
    else:
        # 3) 检测不到章节 → 回退到滑动窗口 (不做父子切分, 无章节结构)
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
