"""
转人工工单闭环 — 坐席工作台 API (人机协同)

工单生命周期: open → taken → closed
  queue    拉取待处理工单 (含会话上下文)
  take     坐席接单
  reply    坐席回复 (写入 handoff_messages, 用户侧轮询可见)
  messages 拉取消息 (用户/坐席)
  close    关闭工单
  suggest  AI 辅助坐席: 基于会话上下文生成回复草稿
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.api.routes.auth import require_roles

logger = logging.getLogger(__name__)

# 后端强制鉴权: 坐席工单操作仅 坐席(analyst)/管理员(admin)
router = APIRouter(prefix="/api/v1/handoff", tags=["转人工"],
                   dependencies=[Depends(require_roles("admin", "analyst"))])


class TakeReq(BaseModel):
    agent_id: str = ""


class ReplyReq(BaseModel):
    agent_id: str = ""
    content: str


class CloseReq(BaseModel):
    agent_id: str = ""


def _session_context(session_id: str) -> list:
    """拉取会话最近对话 (dialog_state history), 供坐席查看上下文"""
    try:
        from src.core.dialog_state import get_state_manager
        state = get_state_manager().get_or_create(session_id)
        return [
            {"role": "user", "query": h.get("query", ""), "answer": (h.get("answer") or "")[:200]}
            for h in state.history[-6:]
        ]
    except Exception:
        return []


@router.get("/queue")
async def queue():
    """待处理工单队列 (open/taken), 附会话上下文"""
    from src.db.manager import db_manager
    tickets = db_manager.get_handoff_queue()
    for t in tickets:
        t["session_context"] = _session_context(t["session_id"])
    return {"tickets": tickets}


@router.post("/{ticket_id}/take")
async def take(ticket_id: int, req: TakeReq):
    """坐席接单 (open→taken)"""
    from src.db.manager import db_manager
    if not db_manager.take_handoff(ticket_id, req.agent_id):
        raise HTTPException(status_code=400, detail="工单不存在或已被接单")
    return {"ok": True}


@router.post("/{ticket_id}/reply")
async def reply(ticket_id: int, req: ReplyReq):
    """坐席回复 (写入消息表, 用户侧轮询可见)"""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="回复内容不能为空")
    from src.db.manager import db_manager
    mid = db_manager.add_handoff_reply(ticket_id, "agent", req.content.strip())
    return {"ok": True, "message_id": mid}


@router.post("/{ticket_id}/close")
async def close(ticket_id: int, req: CloseReq):
    """关闭工单 (→closed)"""
    from src.db.manager import db_manager
    db_manager.close_handoff(ticket_id)
    return {"ok": True}


@router.get("/{ticket_id}/messages")
async def messages(ticket_id: int):
    """拉取工单消息 (用户侧轮询坐席回复)"""
    from src.db.manager import db_manager
    return {"messages": db_manager.get_handoff_messages(ticket_id)}


@router.post("/{ticket_id}/suggest")
async def suggest(ticket_id: int):
    """AI 辅助坐席: 基于会话上下文生成回复草稿 (人机协同的"机辅人")"""
    from src.db.manager import db_manager
    from src.core.dialog_state import get_state_manager
    from src.llm.llm_client import create_client
    ticket = db_manager.get_handoff(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")
    state = get_state_manager().get_or_create(ticket["session_id"])
    history = "\n".join(
        f"用户: {h.get('query', '')}\n客服: {(h.get('answer') or '')[:200]}"
        for h in state.history[-5:]
    )
    prompt = (
        "你是客服坐席助手。用户转人工时, 基于会话上下文草拟一条礼貌、简洁的回复。\n"
        f"【会话上下文】\n{history or '(空)'}\n"
        f"【转人工原因】{ticket.get('reason', '')}\n"
        "只输出回复内容, 不要任何前缀或解释。"
    )
    try:
        draft = create_client(cheap=True).chat(
            [{"role": "system", "content": "你是一名客服坐席。"},
             {"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as e:
        logger.warning("AI 建议生成失败: %s", e)
        draft = "您好，很抱歉给您带来不便，我这边正在为您处理，请稍候。"
    return {"draft": draft}
