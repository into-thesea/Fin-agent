"""反馈与转人工工单落库测试"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.database import db_manager


def _cleanup():
    cur = db_manager.conn.cursor()
    cur.execute("DELETE FROM feedback WHERE session_id LIKE 'test_%'")
    cur.execute("DELETE FROM handoff_tickets WHERE session_id LIKE 'test_%'")
    db_manager.conn.commit()


def test_add_feedback():
    _cleanup()
    fid = db_manager.add_feedback("test_sess_1", rating=1, comment="回答准确", user_id="u001")
    assert fid > 0
    cur = db_manager.conn.cursor()
    cur.execute("SELECT rating, comment FROM feedback WHERE id = ?", (fid,))
    row = cur.fetchone()
    assert row["rating"] == 1
    assert "回答准确" in row["comment"]


def test_add_handoff_ticket():
    _cleanup()
    tid = db_manager.add_handoff_ticket("test_sess_2", user_id="u002", reason="超期退款")
    assert tid > 0
    cur = db_manager.conn.cursor()
    cur.execute("SELECT status, reason FROM handoff_tickets WHERE id = ?", (tid,))
    row = cur.fetchone()
    assert row["status"] == "open"
    assert "超期退款" in row["reason"]
