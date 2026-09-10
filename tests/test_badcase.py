"""badcase 闭环: trace 采集 / 候选派生 / 反馈关联 / golden 提升"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.database import db_manager
from fastapi import HTTPException

TRACE_PREFIX = "test_trace_"


def _cleanup():
    cur = db_manager.conn.cursor()
    cur.execute("DELETE FROM turn_traces WHERE trace_id LIKE ?", (TRACE_PREFIX + "%",))
    cur.execute("DELETE FROM feedback WHERE trace_id LIKE ?", (TRACE_PREFIX + "%",))
    db_manager.conn.commit()


def _make_result(**over):
    r = {
        "answer": "稳盈添利30天业绩比较基准为2.80%-3.60%。",
        "sources": ["prod_p002.md"],
        "intent": "product_consult",
        "review": {"verdict": "pass", "score": 85, "claims": 2},
        "handoff": False,
    }
    r.update(over)
    return r


def test_log_trace_roundtrip():
    _cleanup()
    tid = TRACE_PREFIX + "001"
    rid = db_manager.log_trace(
        trace_id=tid, session_id="s1", query="稳盈添利30天收益多少",
        rewritten="稳盈添利30天 业绩比较基准", result=_make_result(),
        latency=1234.5, user_id="u001",
    )
    assert rid > 0
    row = db_manager.get_trace(tid)
    assert row["query"] == "稳盈添利30天收益多少"
    assert row["rewritten"] == "稳盈添利30天 业绩比较基准"
    assert row["intent"] == "product_consult"
    assert row["sources"] == ["prod_p002.md"]
    assert row["review_verdict"] == "pass"
    assert row["review_score"] == 85
    assert row["triage_status"] == "new"
    _cleanup()


def test_candidate_reasons():
    """造五种行, 断言只捞回该捞的且 reasons 正确"""
    _cleanup()
    # 正常行 — 不该被捞回
    db_manager.log_trace(
        trace_id=TRACE_PREFIX + "ok", session_id="s", query="正常问题",
        rewritten="正常问题",
        result=_make_result(intent="product_consult", sources=["a.md"],
                            _confidence=0.9),
    )
    # 五种候选
    db_manager.log_trace(
        trace_id=TRACE_PREFIX + "unknown", session_id="s", query="???",
        rewritten="???", result=_make_result(intent="unknown", _confidence=0.9),
    )
    db_manager.log_trace(
        trace_id=TRACE_PREFIX + "empty", session_id="s", query="空检索",
        rewritten="空检索", result=_make_result(sources=[], _confidence=0.9),
    )
    db_manager.log_trace(
        trace_id=TRACE_PREFIX + "reject", session_id="s", query="幻觉",
        rewritten="幻觉",
        result=_make_result(review={"verdict": "reject", "score": 10, "claims": 1},
                            _confidence=0.9),
    )
    db_manager.log_trace(
        trace_id=TRACE_PREFIX + "lowconf", session_id="s", query="低置信",
        rewritten="低置信", result=_make_result(_confidence=0.2),
    )
    # 点踩: 先落 trace 再补 feedback
    db_manager.log_trace(
        trace_id=TRACE_PREFIX + "voted", session_id="s", query="被踩",
        rewritten="被踩", result=_make_result(_confidence=0.9),
    )
    db_manager.add_feedback("s", rating=-1, trace_id=TRACE_PREFIX + "voted")

    got = {r["trace_id"]: r for r in db_manager.query_badcases(limit=200)
           if r["trace_id"].startswith(TRACE_PREFIX)}

    assert TRACE_PREFIX + "ok" not in got, "正常行不该进候选池"
    assert got[TRACE_PREFIX + "unknown"]["reasons"] == ["unknown_intent"]
    assert got[TRACE_PREFIX + "empty"]["reasons"] == ["empty_retrieval"]
    assert got[TRACE_PREFIX + "reject"]["reasons"] == ["reject"]
    assert got[TRACE_PREFIX + "lowconf"]["reasons"] == ["low_confidence"]
    assert got[TRACE_PREFIX + "voted"]["reasons"] == ["downvote"]
    assert got[TRACE_PREFIX + "empty"]["sources_n"] == 0
    _cleanup()


def test_detail_reasons_no_false_empty_retrieval():
    """get_trace 的行形态不能被误报 empty_retrieval

    get_trace 把 sources_json 解析成 sources 列表并 pop 掉原键。
    若 _badcase_reasons 只认 sources_json, 读到的就是 None,
    而 (None or '') in ('[]','','null') 恒为真 —— 任何有召回的轮次都会被误报。
    """
    _cleanup()
    tid = TRACE_PREFIX + "detail"
    db_manager.log_trace(
        trace_id=tid, session_id="s", query="有召回的问题",
        rewritten="有召回的问题",
        result=_make_result(sources=["a.md", "b.md"], intent="product_consult",
                            _confidence=0.9),
    )
    trace = db_manager.get_trace(tid)
    assert "sources_json" not in trace, "前提变了: get_trace 不再 pop sources_json"
    assert db_manager._badcase_reasons(trace) == [], "有召回却被误报为候选"

    # 列表接口与详情接口必须给出同一份 reasons (点踩在两边都可见)
    db_manager.add_feedback("s", rating=-1, trace_id=tid)
    listed = [r for r in db_manager.query_badcases(limit=200) if r["trace_id"] == tid][0]
    detail = db_manager.get_trace(tid)
    assert listed["reasons"] == ["downvote"]
    assert db_manager._badcase_reasons(detail) == ["downvote"], \
        "详情接口漏报点踩, 与列表接口不一致"
    _cleanup()


def test_post_process_writes_trace():
    """_do_post_process 应把 trace 落到库, 且用传入的 trace_id"""
    from src.api.routes import chat as chat_route

    _cleanup()
    tid = TRACE_PREFIX + "pp"
    result = _make_result()
    result["_confidence"] = 0.9

    class _FakeStateMgr:
        def update(self, **kw):
            pass

    chat_route._do_post_process(
        "改写后问题", result, "s_pp", _FakeStateMgr(),
        "改写后问题", "原始问题", 12.5, "product_consult",
        trace_id=tid, user_id="u_pp",
    )
    row = db_manager.get_trace(tid)
    assert row["query"] == "原始问题"
    assert row["rewritten"] == "改写后问题"
    assert row["user_id"] == "u_pp"
    _cleanup()


def test_graph_exposes_confidence():
    """run / run_graph_stream 必须把 confidence 带出来.

    图状态里有 confidence (cs_graph 路由节点写入), 但两条出边各自手工拼了一份
    返回 dict。任一处漏掉, 低置信候选规则就会静默失效 — 所以两处都要守。
    """
    import inspect
    from src.graph import cs_graph

    for fn in (cs_graph.run, cs_graph.run_graph_stream):
        src = inspect.getsource(fn)
        assert '"_confidence"' in src, (
            f"{fn.__name__} 未暴露 _confidence, 低置信候选规则会静默失效"
        )


def test_submit_feedback_links_trace():
    """带 trace_id 的反馈必须能被候选查询的 EXISTS 子查询命中"""
    import asyncio
    from src.api.routes.chat import FeedbackRequest, submit_feedback

    _cleanup()
    tid = TRACE_PREFIX + "fb"
    db_manager.log_trace(
        trace_id=tid, session_id="s_fb", query="被踩的问题",
        rewritten="被踩的问题", result=_make_result(_confidence=0.9),
    )
    asyncio.new_event_loop().run_until_complete(
        submit_feedback(FeedbackRequest(session_id="s_fb", rating=-1,
                                        comment="答非所问", trace_id=tid))
    )
    got = {r["trace_id"]: r for r in db_manager.query_badcases(limit=200)
           if r["trace_id"].startswith(TRACE_PREFIX)}
    assert tid in got, "带 trace_id 的反馈未能把该轮带进候选池"
    assert "downvote" in got[tid]["reasons"]
    assert db_manager.get_trace(tid)["feedbacks"][0]["comment"] == "答非所问"
    _cleanup()


def test_promote_writes_empty_evidence(tmp_path):
    """promote 出的 golden 行 evidence 必须为空, source_trace 必须正确"""
    import shutil
    from src.core.golden import promote_to_golden, GOLDEN_PATH

    gp = tmp_path / "golden.jsonl"
    shutil.copy(GOLDEN_PATH, gp)

    trace = {"trace_id": "test_trace_p1", "query": "稳盈添利30天能提前赎回吗",
             "intent": "product_consult", "sources": ["prod_p002.md", "prod_p009.md"]}
    out = promote_to_golden(trace, note="答案漏了赎回条款", golden_path=str(gp))

    assert out["needs_evidence"] is True
    assert out["already"] is False
    rows = [json.loads(l) for l in open(gp, encoding="utf-8") if l.strip()]
    row = [r for r in rows if r["id"] == out["id"]][0]
    assert row["evidence"] == [], "evidence 必须留空等人补, 不能填当轮召回的 sources"
    assert row["source_trace"] == "test_trace_p1"
    assert row["question"] == "稳盈添利30天能提前赎回吗"

    # 同一条 trace 重复 promote 不追加新行
    again = promote_to_golden(trace, golden_path=str(gp))
    assert again["already"] is True
    assert again["id"] == out["id"]
    rows2 = [json.loads(l) for l in open(gp, encoding="utf-8") if l.strip()]
    assert len(rows2) == len(rows)


def test_badcase_endpoints(monkeypatch):
    import asyncio
    from src.api.routes import monitor as mon

    _cleanup()
    tid = TRACE_PREFIX + "api"
    db_manager.log_trace(
        trace_id=tid, session_id="s_api", query="接口测试问题",
        rewritten="接口测试问题", result=_make_result(sources=[], _confidence=0.9),
    )

    loop = asyncio.new_event_loop()
    listed = loop.run_until_complete(mon.list_badcases(status="", limit=200, offset=0))
    assert any(r["trace_id"] == tid for r in listed["items"])

    detail = loop.run_until_complete(mon.get_badcase(tid))
    assert detail["query"] == "接口测试问题"
    assert detail["sources"] == []

    ok = loop.run_until_complete(
        mon.patch_badcase(tid, mon.TriagePatch(status="confirmed", note="已复现")))
    assert ok["ok"] is True
    assert db_manager.get_trace(tid)["triage_status"] == "confirmed"

    # 404 分支
    try:
        loop.run_until_complete(mon.get_badcase(TRACE_PREFIX + "nope"))
        assert False, "不存在的 trace_id 应抛 404"
    except HTTPException as e:
        assert e.status_code == 404

    # promote 端点的接线: 替换掉 golden 写入函数, 确保绝不触碰真实 golden 文件
    # (promote_to_golden 自身行为由 test_promote_writes_empty_evidence 覆盖)
    calls = {}

    def _fake_promote(trace, note="", golden_path=None):
        calls["trace_id"] = trace.get("trace_id")
        calls["note"] = note
        return {"id": "bc999", "needs_evidence": True, "already": False}

    import src.core.golden as g
    monkeypatch.setattr(g, "promote_to_golden", _fake_promote)
    out = loop.run_until_complete(mon.promote_badcase(tid, mon.PromoteBody(note="漏答")))
    assert out == {"id": "bc999", "needs_evidence": True, "already": False}
    assert calls["trace_id"] == tid
    assert calls["note"] == "漏答"
    # 提升后自动标为 confirmed
    assert db_manager.get_trace(tid)["triage_status"] == "confirmed"
    loop.close()
    _cleanup()
