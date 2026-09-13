"""
Fin-Agent 元数据库 (企业版)

功能:
  - 文档元数据管理 (含内容哈希去重、版本追踪)
  - 查询审计日志
  - 系统健康指标记录

Schema 扩展:
  - documents 表增加 content_hash / version / prev_version_id
  - 新增 task_logs 表 (异步 ETL 任务追踪)
"""

import os
import json
import time
import hashlib
import logging
import sqlite3
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)

from src.infra.paths import OUTPUT_ANALYSIS_DIR
DB_DIR = OUTPUT_ANALYSIS_DIR
DB_PATH = os.path.join(DB_DIR, "fin_agent_metadata.db")


class DBManager:
    """元数据库管理器 (单例模式)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        os.makedirs(DB_DIR, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        """初始化数据库表结构 (含旧表迁移)"""
        cursor = self.conn.cursor()

        # 检查是否需要迁移 (旧表有 filename UNIQUE 约束)
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'")
        table_def = cursor.fetchone()
        needs_migration = table_def and "UNIQUE" in table_def[0].upper() and "filename" in table_def[0]

        if needs_migration:
            logger.info("检测到旧版 documents 表，正在执行 schema 迁移...")
            # 1. 创建新表 (无 filename 单独 UNIQUE)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    upload_time TEXT,
                    chunks_count INTEGER DEFAULT 0,
                    content_hash TEXT,
                    file_size INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    prev_version_id INTEGER,
                    source_label TEXT DEFAULT 'upload',
                    FOREIGN KEY (prev_version_id) REFERENCES documents_new(id)
                )
            """)
            # 2. 复制旧数据
            cursor.execute("""
                INSERT INTO documents_new (id, filename, status, upload_time, chunks_count, content_hash, file_size, version, prev_version_id)
                SELECT id, filename, status, upload_time, chunks_count, NULL, 0, 1, NULL FROM documents
            """)
            # 3. 删除旧表
            cursor.execute("DROP TABLE documents")
            # 4. 重命名
            cursor.execute("ALTER TABLE documents_new RENAME TO documents")
            # 5. 重建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_filename ON documents(filename)")
            try:
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_filename_hash ON documents(filename, content_hash)")
            except sqlite3.OperationalError as e:
                logger.debug("索引创建跳过 (已存在): %s", e)
            self.conn.commit()
            logger.info("Schema 迁移完成!")

        # 标准建表 (首次使用)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                upload_time TEXT,
                chunks_count INTEGER DEFAULT 0,
                content_hash TEXT,
                file_size INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                prev_version_id INTEGER,
                source_label TEXT DEFAULT 'upload',
                FOREIGN KEY (prev_version_id) REFERENCES documents(id)
            )
        """)

        # 添加新列 (兼容旧表)
        for col_sql in [
            "ALTER TABLE documents ADD COLUMN content_hash TEXT",
            "ALTER TABLE documents ADD COLUMN version INTEGER DEFAULT 1",
            "ALTER TABLE documents ADD COLUMN prev_version_id INTEGER",
            "ALTER TABLE documents ADD COLUMN file_size INTEGER DEFAULT 0",
            "ALTER TABLE documents ADD COLUMN source_label TEXT DEFAULT 'upload'",
        ]:
            try:
                cursor.execute(col_sql)
            except sqlite3.OperationalError:
                pass  # 列已存在

        # 索引 (幂等)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_filename ON documents(filename)")
        try:
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_filename_hash ON documents(filename, content_hash)")
        except sqlite3.OperationalError as e:
            logger.debug("索引创建跳过 (已存在): %s", e)

        # ETL 任务日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                filename TEXT,
                status TEXT,
                progress INTEGER DEFAULT 0,
                stage TEXT,
                error TEXT,
                started_at TEXT,
                finished_at TEXT
            )
        """)

        # 审计日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                latency REAL,
                tokens_used INTEGER DEFAULT 0,
                cache_hit BOOLEAN DEFAULT 0,
                user_id TEXT DEFAULT 'anonymous',
                intent TEXT,
                confidence REAL,
                rule_result TEXT,
                timestamp TEXT
            )
        """)

        # 兼容旧表: 新增列 (幂等)
        existing_cols = [r[1] for r in cursor.execute("PRAGMA table_info(audit_logs)").fetchall()]
        if "intent" not in existing_cols:
            cursor.execute("ALTER TABLE audit_logs ADD COLUMN intent TEXT")
        if "confidence" not in existing_cols:
            cursor.execute("ALTER TABLE audit_logs ADD COLUMN confidence REAL")
        if "rule_result" not in existing_cols:
            cursor.execute("ALTER TABLE audit_logs ADD COLUMN rule_result TEXT")

        # 系统健康记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                recorded_at TEXT
            )
        """)

        # 客服反馈表 (Phase 3: 点赞/踩 + 评论)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT DEFAULT 'anonymous',
                rating INTEGER,          -- 1 赞 / -1 踩
                comment TEXT,
                intent TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")

        # badcase 闭环: 每轮对话 trace
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS turn_traces (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id       TEXT,
                session_id     TEXT,
                user_id        TEXT DEFAULT 'anonymous',
                query          TEXT,
                rewritten      TEXT,
                intent         TEXT,
                confidence     REAL,
                sources_json   TEXT,
                review_verdict TEXT,
                review_score   INTEGER,
                answer         TEXT,
                handoff        INTEGER DEFAULT 0,
                cache_hit      INTEGER DEFAULT 0,
                latency        REAL,
                triage_status  TEXT DEFAULT 'new',
                triage_note    TEXT,
                triaged_at     TEXT,
                created_at     TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_trace ON turn_traces(trace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_created ON turn_traces(created_at)")

        # 反馈关联到具体轮次 (幂等迁移)
        fb_cols = [r[1] for r in cursor.execute("PRAGMA table_info(feedback)").fetchall()]
        if "trace_id" not in fb_cols:
            cursor.execute("ALTER TABLE feedback ADD COLUMN trace_id TEXT")

        # 人工转接工单表 (Phase 3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS handoff_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT DEFAULT 'anonymous',
                reason TEXT,
                status TEXT DEFAULT 'open',   -- open / taken / closed
                agent_id TEXT,
                created_at TEXT,
                resolved_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_handoff_status ON handoff_tickets(status)")

        # 转人工工单消息表 (坐席回复, 用户侧轮询)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS handoff_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER,
                role TEXT,          -- agent / system
                content TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_handoff_msg_ticket ON handoff_messages(ticket_id)")

        self.conn.commit()

    # ──────────────────────────────────────────────
    # 文档内容寻址
    # ──────────────────────────────────────────────

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """计算文件的 SHA256 哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """计算文本内容的 SHA256 哈希"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def find_document_by_hash(self, content_hash: str) -> Optional[dict]:
        """通过内容哈希查找已有文档"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, filename, version FROM documents WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        row = cursor.fetchone()
        if row:
            return {"id": row["id"], "filename": row["filename"], "version": row["version"]}
        return None

    def find_document_by_filename(self, filename: str) -> Optional[dict]:
        """通过文件名查找最新版本"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, filename, version, content_hash FROM documents WHERE filename = ? ORDER BY version DESC LIMIT 1",
            (filename,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    # ──────────────────────────────────────────────
    # 文档 CRUD
    # ──────────────────────────────────────────────

    def add_document(
        self,
        filename: str,
        status: str = "done",
        count: int = 0,
        content_hash: str = None,
        file_size: int = 0,
        source_label: str = "upload",
    ) -> int:
        """
        添加文档记录 (自动版本管理)

        如果同一 filename + content_hash 已存在 → 跳过 (返回现有 ID)
        如果同一 filename 不同 hash → 版本递增 (version + 1)

        Returns: 文档 ID
        """
        cursor = self.conn.cursor()

        # 查找旧版本
        existing = self.find_document_by_filename(filename)

        if existing:
            # 同一内容 → 跳过
            if content_hash and existing.get("content_hash") == content_hash:
                logger.info(f"文档 {filename} 内容一致 (hash: {content_hash[:12]}...), 跳过入库")
                return existing["id"]

            # 不同内容 → 新版本
            new_version = existing["version"] + 1
            prev_id = existing["id"]
        else:
            new_version = 1
            prev_id = None

        cursor.execute(
            """INSERT INTO documents
               (filename, status, upload_time, chunks_count, content_hash, file_size, version, prev_version_id, source_label)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, status, time.ctime(), count, content_hash, file_size, new_version, prev_id, source_label),
        )
        self.conn.commit()

        doc_id = cursor.lastrowid
        action = "新版本" if prev_id else "首次入库"
        logger.info(f"文档 {filename} {action}: v{new_version} (ID: {doc_id})")
        return doc_id

    def update_document_status(self, doc_id: int, status: str, count: int = None) -> bool:
        """更新文档状态 / 分块数

        配合 ETL 的「先占位、后收尾」: 流水线一开始就以 status='processing' 登记,
        这样并发的第二条流水线在去重检查处就能看到记录并跳过, 不会重复写向量。
        """
        cursor = self.conn.cursor()
        if count is None:
            cursor.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
        else:
            cursor.execute(
                "UPDATE documents SET status = ?, chunks_count = ? WHERE id = ?",
                (status, count, doc_id),
            )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_document_by_id(self, doc_id: int) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_documents(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, filename, status, upload_time, chunks_count, version, content_hash "
            "FROM documents ORDER BY upload_time DESC"
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "filename": r[1],
                "status": r[2],
                "upload_time": r[3],
                "chunks_count": r[4],
                "version": r[5],
                "content_hash": r[6][:12] if r[6] else None,
            }
            for r in rows
        ]

    def delete_document(self, doc_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ──────────────────────────────────────────────
    # 查询审计
    # ──────────────────────────────────────────────

    def log_query(self, query: str, latency: float, tokens: int = 0, cache_hit: bool = False,
                   user_id: str = "anonymous", intent: str = None, confidence: float = None,
                   rule_result: str = None):
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO audit_logs
               (query, latency, tokens_used, cache_hit, user_id, intent, confidence, rule_result, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (query[:500], latency, tokens, int(cache_hit), user_id,
             intent, confidence, rule_result, time.ctime()),
        )
        self.conn.commit()

    # ──────────────────────────────────────────────
    # 客服反馈 / 人工转接 (Phase 3)
    # ──────────────────────────────────────────────

    def add_feedback(self, session_id: str, rating: int, comment: str = "",
                     user_id: str = "anonymous", intent: str = "", trace_id: str = ""):
        """记录用户对回答的反馈 (rating: 1 赞 / -1 踩)"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (session_id, user_id, rating, comment, intent, trace_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, user_id, rating, comment, intent, trace_id, time.ctime()),
        )
        self.conn.commit()
        return cursor.lastrowid

    # ──────────────────────────────────────────────
    # 对话 trace / badcase 闭环
    # ──────────────────────────────────────────────

    # 候选判定规则 — 唯一来源, 改规则只改这里, 不动历史数据
    BADCASE_WHERE = """
        t.intent IN ('unknown', '')
        OR t.sources_json IN ('[]', '', 'null') OR t.sources_json IS NULL
        OR t.review_verdict = 'reject'
        OR (t.confidence IS NOT NULL AND t.confidence < 0.5)
        OR EXISTS (SELECT 1 FROM feedback f2 WHERE f2.trace_id = t.trace_id AND f2.rating = -1)
    """

    def log_trace(self, trace_id: str, session_id: str, query: str, rewritten: str,
                  result: dict, latency: float = 0.0,
                  user_id: str = "anonymous", cache_hit: bool = False) -> int:
        """记录一轮对话的完整 trace (供 badcase 追溯)"""
        review = result.get("review") or {}
        sources = result.get("sources") or []
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO turn_traces
               (trace_id, session_id, user_id, query, rewritten, intent, confidence,
                sources_json, review_verdict, review_score, answer, handoff, cache_hit,
                latency, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trace_id, session_id, user_id,
             (query or "")[:500], (rewritten or "")[:500],
             result.get("intent") or result.get("_intent") or "",
             result.get("_confidence"),
             json.dumps(sources, ensure_ascii=False)[:4000],
             review.get("verdict", ""), review.get("score"),
             (result.get("answer") or "")[:5000],
             int(bool(result.get("handoff"))), int(cache_hit), latency, time.ctime()),
        )
        self.conn.commit()
        return cursor.lastrowid

    @staticmethod
    def _badcase_reasons(row: dict) -> list:
        """从已存字段派生候选原因 (必须与 BADCASE_WHERE 保持一致)

        两种行形态都要认: query_badcases 给的是 sources_json (原始字符串),
        get_trace 已把它解析成 sources (列表) 并 pop 掉了原键。只认前者的话,
        详情接口会把任何有召回的轮次都误报成 empty_retrieval。
        """
        reasons = []
        if row.get("rating") == -1:
            reasons.append("downvote")
        if (row.get("intent") or "") in ("unknown", ""):
            reasons.append("unknown_intent")
        srcs = row.get("sources")
        if srcs is not None:
            empty = len(srcs) == 0
        else:
            empty = (row.get("sources_json") or "") in ("[]", "", "null")
        if empty:
            reasons.append("empty_retrieval")
        if row.get("review_verdict") == "reject":
            reasons.append("reject")
        conf = row.get("confidence")
        if conf is not None and conf < 0.5:
            reasons.append("low_confidence")
        return reasons

    def query_badcases(self, status: str = "", limit: int = 100, offset: int = 0) -> list:
        """badcase 候选列表 (条件在查询侧派生, 不落盘)"""
        where = [f"({self.BADCASE_WHERE})"]
        params = []
        if status:
            where.append("t.triage_status = ?")
            params.append(status)
        sql = (
            "SELECT t.id, t.trace_id, t.session_id, t.query, t.intent, t.confidence, "
            "t.sources_json, t.review_verdict, t.review_score, t.handoff, t.cache_hit, "
            "t.triage_status, t.triage_note, t.created_at, "
            "(SELECT MIN(rating) FROM feedback f2 WHERE f2.trace_id = t.trace_id) AS rating "
            "FROM turn_traces t "
            f"WHERE {' AND '.join(where)} ORDER BY t.id DESC LIMIT ? OFFSET ?"
        )
        cursor = self.conn.cursor()
        cursor.execute(sql, (*params, limit, offset))
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            d["reasons"] = self._badcase_reasons(d)
            d["sources_n"] = len(json.loads(d["sources_json"] or "[]"))
            d.pop("sources_json", None)
            rows.append(d)
        return rows

    def get_trace(self, trace_id: str) -> dict:
        """单条 trace 全文 (badcase 详情)"""
        cursor = self.conn.cursor()
        # ORDER BY 取最新行: trace_id 正常每请求唯一, 但读路径不该依赖 SQLite 的默认行序
        # rating 与 query_badcases 同源, 保证两个接口给出一致的 reasons
        cursor.execute(
            "SELECT t.*, (SELECT MIN(rating) FROM feedback f2 WHERE f2.trace_id = t.trace_id) AS rating "
            "FROM turn_traces t WHERE t.trace_id = ? ORDER BY t.id DESC LIMIT 1",
            (trace_id,))
        row = cursor.fetchone()
        if not row:
            return {}
        d = dict(row)
        d["sources"] = json.loads(d.pop("sources_json") or "[]")
        cursor.execute(
            "SELECT rating, comment, user_id, timestamp FROM feedback "
            "WHERE trace_id = ? ORDER BY id DESC", (trace_id,))
        d["feedbacks"] = [dict(r) for r in cursor.fetchall()]
        return d

    def update_triage(self, trace_id: str, status: str = None, note: str = None) -> bool:
        """更新 triage 状态 / 备注"""
        sets, params = [], []
        if status is not None:
            sets.append("triage_status = ?")
            params.append(status)
        if note is not None:
            sets.append("triage_note = ?")
            params.append(note)
        if not sets:
            return False
        sets.append("triaged_at = ?")
        params.append(time.ctime())
        params.append(trace_id)
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE turn_traces SET {', '.join(sets)} WHERE trace_id = ?", params)
        self.conn.commit()
        return cursor.rowcount > 0

    def add_handoff_ticket(self, session_id: str, user_id: str = "anonymous",
                           reason: str = "") -> int:
        """创建人工转接工单 (status=open)"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO handoff_tickets (session_id, user_id, reason, status, created_at) "
            "VALUES (?, ?, ?, 'open', ?)",
            (session_id, user_id, reason, time.ctime()),
        )
        self.conn.commit()
        return cursor.lastrowid

    # ── 转人工工单闭环 (坐席接管) ─────────────────
    def get_handoff_queue(self, status: str = "open") -> list:
        """拉取待处理工单 (open/taken), 按创建倒序"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM handoff_tickets WHERE status IN (?, ?) ORDER BY id DESC LIMIT 50",
            (status, "taken"),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_handoff(self, ticket_id: int) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM handoff_tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def take_handoff(self, ticket_id: int, agent_id: str) -> bool:
        """坐席接单 (open→taken); 已被他人接单返回 False"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE handoff_tickets SET status='taken', agent_id=? WHERE id=? AND status='open'",
            (agent_id, ticket_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def close_handoff(self, ticket_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE handoff_tickets SET status='closed', resolved_at=? WHERE id=? AND status != 'closed'",
            (time.ctime(), ticket_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def add_handoff_reply(self, ticket_id: int, role: str, content: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO handoff_messages (ticket_id, role, content, created_at) VALUES (?,?,?,?)",
            (ticket_id, role, content, time.ctime()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_handoff_messages(self, ticket_id: int) -> list:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, role, content, created_at FROM handoff_messages WHERE ticket_id=? ORDER BY id",
            (ticket_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # ──────────────────────────────────────────────
    # 监控统计 (客服: 意图分布 / 转人工率 / 审计日志)
    # ──────────────────────────────────────────────

    def get_recent_audit_logs(self, limit: int = 20) -> list:
        """最近审计日志 (监控页用)"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, query, latency, tokens_used, cache_hit, user_id, intent, timestamp "
            "FROM audit_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_intent_stats(self) -> dict:
        """查询意图分布 {intent: count}"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT intent, COUNT(*) AS cnt FROM audit_logs "
            "WHERE intent IS NOT NULL AND intent != '' GROUP BY intent ORDER BY cnt DESC"
        )
        return {r["intent"]: r["cnt"] for r in cursor.fetchall()}

    def count_handoff_tickets(self, status: str = None) -> int:
        """转人工工单数量 (可按状态过滤)"""
        cursor = self.conn.cursor()
        if status:
            cursor.execute("SELECT COUNT(*) AS c FROM handoff_tickets WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) AS c FROM handoff_tickets")
        return cursor.fetchone()["c"]

    # ──────────────────────────────────────────────
    # ETL 任务追踪
    # ──────────────────────────────────────────────

    def create_task(self, task_id: str, filename: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO task_logs (task_id, filename, status, progress, started_at) VALUES (?, ?, 'pending', 0, ?)",
            (task_id, filename, time.ctime()),
        )
        self.conn.commit()

    def get_task(self, task_id: str) -> Optional[dict]:
        """按 task_id 查任务记录 (Redis TTL 过期后的历史兜底)"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT task_id, filename, status, progress, stage, error, started_at, finished_at "
            "FROM task_logs WHERE task_id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_task(self, task_id: str, status: str, progress: int = None, stage: str = None, error: str = None):
        cursor = self.conn.cursor()
        updates = []
        params = []

        if status:
            updates.append("status = ?")
            params.append(status)
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        if stage:
            updates.append("stage = ?")
            params.append(stage)
        if error:
            updates.append("error = ?")
            params.append(error)
        if status in ("completed", "failed"):
            updates.append("finished_at = ?")
            params.append(time.ctime())

        params.append(task_id)
        cursor.execute(
            f"UPDATE task_logs SET {', '.join(updates)} WHERE task_id = ?",
            params,
        )
        self.conn.commit()

    # ──────────────────────────────────────────────
    # 系统健康
    # ──────────────────────────────────────────────

    def record_health_metric(self, metric_name: str, metric_value: float):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO system_health (metric_name, metric_value, recorded_at) VALUES (?, ?, ?)",
            (metric_name, metric_value, time.ctime()),
        )
        self.conn.commit()

    def get_recent_metrics(self, metric_name: str, limit: int = 100) -> list:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT metric_value, recorded_at FROM system_health WHERE metric_name = ? ORDER BY id DESC LIMIT ?",
            (metric_name, limit),
        )
        return cursor.fetchall()

    # ──────────────────────────────────────────────
    # 统计
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) AS c FROM documents")
        doc_count = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) AS c FROM audit_logs")
        query_count = cursor.fetchone()["c"]
        cursor.execute("SELECT COALESCE(SUM(chunks_count), 0) AS c FROM documents")
        chunk_count = cursor.fetchone()["c"]
        return {
            "documents": doc_count,
            "queries": query_count,
            "chunks": chunk_count,
            "db_path": DB_PATH,
        }

    def close(self):
        self.conn.close()


# 全局单例
db_manager = DBManager()
