"""
Golden QA 集读写 — badcase 提升入口

badcase 提升为 golden 时, evidence 刻意留空: badcase 的成因常常正是
"检索错了", 自动填入当轮召回的 sources 会把 bug 固化成标准答案。
证据必须人工补齐。
"""

import json
import os
import re

GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "eval", "finance_qa_golden.jsonl",
)


def load_golden(golden_path: str = GOLDEN_PATH) -> list:
    if not os.path.exists(golden_path):
        return []
    with open(golden_path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def next_badcase_id(golden_path: str = GOLDEN_PATH) -> str:
    """扫 bc 前缀取最大数字 +1, 与手写的 g001 系列不冲突"""
    used = set()
    for r in load_golden(golden_path):
        m = re.fullmatch(r"bc(\d+)", str(r.get("id", "")))
        if m:
            used.add(int(m.group(1)))
    n = max(used) + 1 if used else 1
    return f"bc{n:03d}"


def promote_to_golden(trace: dict, note: str = "", golden_path: str = GOLDEN_PATH) -> dict:
    """把一条 badcase 提升为 golden 骨架 (evidence 留空, 等人补)

    Returns: {"id": str, "needs_evidence": bool, "already": bool}
    """
    trace_id = trace.get("trace_id", "")
    # 同一条 trace 重复提升不追加新行
    for r in load_golden(golden_path):
        if r.get("source_trace") and r["source_trace"] == trace_id:
            return {"id": r["id"], "needs_evidence": not r.get("evidence"), "already": True}

    row = {
        "id": next_badcase_id(golden_path),
        "question": trace.get("query", ""),
        "intent": trace.get("intent", ""),
        "complexity": "simple",
        "evidence": [],
        "note": f"badcase {trace_id}: {note}".strip().rstrip(":"),
        "source_trace": trace_id,
    }
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    with open(golden_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"id": row["id"], "needs_evidence": True, "already": False}
