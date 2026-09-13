"""
知识库管理台接口 —— /api/v1/kb/*

整组 admin-only, 与 /api/v1/knowledge 一致。
校验交给 kb_validate (纯函数), 读写交给 kb_settings, 本层只负责 HTTP 语义。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.routes.auth import require_roles
from src.core import kb_settings
from src.core.kb_validate import validate, ValidationError

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
