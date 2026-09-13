"""
知识库管理台接口 —— /api/v1/kb/*

整组 admin-only, 与 /api/v1/knowledge 一致。
校验交给 kb_validate (纯函数), 读写交给 kb_settings, 本层只负责 HTTP 语义。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.routes.auth import require_roles
from src.core import kb_settings
from src.core.kb_validate import validate, ValidationError
# 模块级 import: 测试靠 patch 这个名字把分块文件指到 tmp, 放函数里 patch 不生效
from src.infra.paths import CHUNKS_PROCESSED_PATH
from src.infra.jsonl_io import read_lines

logger = logging.getLogger(__name__)

# 后端强制鉴权: 整组挂守卫, 新增端点默认继承, 不靠逐个端点记得加
router = APIRouter(prefix="/api/v1/kb", tags=["知识库管理台"],
                   dependencies=[Depends(require_roles("admin"))])


class StrategyPut(BaseModel):
    section: str
    values: dict


@router.get("/strategy")
async def get_strategy():
    """当前生效值 + 代码默认值 + 哪些偏离了默认"""
    cfg, source = kb_settings.get_all()
    out = {"source": source,
           "updated_at": cfg.get("updated_at", ""),
           "updated_by": cfg.get("updated_by", "")}
    for section in ("chunking", "retrieval"):
        current = cfg[section]
        default = kb_settings.DEFAULTS[section]
        out[section] = {
            "current": current,
            "default": default,
            "changed": sorted(k for k, v in current.items() if default.get(k) != v),
        }
    return out


@router.put("/strategy")
async def put_strategy(body: StrategyPut, user=Depends(require_roles("admin"))):
    """保存某个配置段。校验失败返回 400 并指明具体字段与规则。"""
    try:
        values = validate(body.section, body.values)
    except ValidationError as e:
        # 前端直接读 detail.field / detail.rule 定位输入框, 故原样返回
        raise HTTPException(status_code=400, detail=e.as_dict())

    before, _ = kb_settings.get_all()
    cfg = kb_settings.save(body.section, values, user=user.get("user_id", "admin"))
    after = cfg[body.section]

    changed = sorted(k for k in values if before[body.section].get(k) != after[k])
    return {
        "saved": True,
        "changed": changed,
        # 切分参数只在下次上传时由 smart_chunk_pdf 使用, 重建也不影响已有内容
        "applies_to": "next_request" if body.section == "retrieval" else "next_upload",
        "config": after,
    }


@router.get("/chunks")
async def list_chunks(
    source: str = Query("", description="按来源文件精确筛选"),
    section: str = Query("", description="按小节标题子串筛选"),
    q: str = Query("", description="内容关键词子串"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),   # 上限 100, 不设等于允许一次拉全库
):
    """分块浏览 (走文件, 不走向量——这是浏览不是检索)

    翻到越界页或筛完没结果都只返回空列表, 不报错 —— 这个页面是排查
    "检索为什么没命中" 的入口, 报错会被当成知识库坏了。
    """
    rows = read_lines(CHUNKS_PROCESSED_PATH)
    if source:
        rows = [r for r in rows if r.get("source") == source]
    if section:
        rows = [r for r in rows if section in (r.get("section") or "")]
    if q:
        rows = [r for r in rows if q in (r.get("content") or "")]

    start = (page - 1) * size
    items = [
        {
            "chunk_id": r.get("chunk_id", ""),
            "source": r.get("source", ""),
            "section": r.get("section", ""),
            "page": r.get("page", 1),
            "content": r.get("content", ""),
            "content_hash": r.get("content_hash", ""),
            "length": len(r.get("content") or ""),
        }
        for r in rows[start:start + size]
    ]
    return {"total": len(rows), "page": page, "size": size, "items": items}
