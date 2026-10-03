"""
对话状态追踪 (DST) — 多轮对话的实体/指标/时间追踪

数据结构:
    DialogState:
        - session_id: str              # 会话标识
        - history: list[dict]           # 最近 N 轮对话 [{role, query, answer}]
        - focal_company: str | None     # 当前焦点公司
        - focal_metric: str | None      # 当前焦点指标
        - focal_year: int | str | None  # 当前焦点年份
        - last_intent: str | None       # 上轮意图

存储:
    内存 dict (L1) + Redis (L2, 30 分钟 TTL)
    重启后从 Redis 恢复，适合多机部署
"""

import os
import json
import logging
import threading
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

MAX_HISTORY = 10  # 最多保留 10 轮
_DIALOG_REDIS_TTL = 1800  # 30 分钟


@dataclass
class DialogState:
    """单次会话的状态"""
    session_id: str = ""
    history: list = field(default_factory=list)       # [{"role": "user", "query": str, "answer": str}, ...]
    focal_company: Optional[str] = None                # 焦点公司 (客服场景一般不使用)
    focal_metric: Optional[str] = None                 # 焦点指标 (保留兼容)
    focal_year: Optional[str] = None                   # 焦点年份 (保留兼容)
    last_intent: Optional[str] = None                  # 上轮意图 (12类平面结构, 兼容旧代码)
    # ── 意图分层 (L0/L1/L2) ──
    current_l0: Optional[str] = None                   # 当前产品域, 取值见 slot_filler.L0_DOMAINS
    current_l1: Optional[str] = None                   # 当前问询类型: product_info/income/fee/risk/buy_process/redeem/compare/compliance
    current_l2: Optional[str] = None                   # LLM 动态提取的具体操作 (可选)
    # ── 槽位 ──
    slots: dict = field(default_factory=dict)          # 客服槽位 {product, product_type, amount, term, risk_level, ...}
    awaiting_slot: Optional[str] = None                # 当前待补充的槽位名
    # ── 多轮对话增强 ──
    topic_stack: list = field(default_factory=list)    # 话题栈: 切换产品域时旧话题入栈, 支持"回到刚才"
    negative_turns: int = 0                            # 连续负面情绪轮次 (情绪处理用)
    turn_count: int = 0                                # 当前话题轮次计数
    emotion: Optional[str] = None                      # 上轮检测到的情绪: normal/anxiety/complaint/confusion

    def push_topic(self):
        """当前话题入栈 (切换产品域前调用)"""
        if self.current_l0:
            self.topic_stack.append({
                "l0": self.current_l0,
                "l1": self.current_l1,
                "l2": self.current_l2,
                "slots": dict(self.slots),
                "intent": self.last_intent,
            })
            logger.debug("话题入栈: l0=%s, 栈深度=%d", self.current_l0, len(self.topic_stack))

    def pop_topic(self) -> Optional[dict]:
        """弹出上一个话题 (用户说"回到刚才"时调用)"""
        if self.topic_stack:
            topic = self.topic_stack.pop()
            self.current_l0 = topic.get("l0")
            self.current_l1 = topic.get("l1")
            self.current_l2 = topic.get("l2")
            self.slots = topic.get("slots", {})
            self.last_intent = topic.get("intent")
            logger.debug("话题出栈: l0=%s, 栈深度=%d", self.current_l0, len(self.topic_stack))
            return topic
        return None

    def reset_topic(self):
        """开启新话题时重置轮次计数"""
        self.turn_count = 0
        self.negative_turns = 0


class DialogStateManager:
    """对话状态管理器 (线程安全, 内存+Redis 双层)"""

    def __init__(self):
        self._sessions: dict[str, DialogState] = {}
        self._lock = threading.RLock()
        self._redis = None
        self._redis_ready = threading.Event()
        # 后台线程预热 Redis (避免 3s WSL ping 阻塞事件循环)
        threading.Thread(target=self._warm_redis, daemon=True).start()

    def _warm_redis(self):
        """在后台线程中预热 Redis 连接"""
        try:
            from src.cache.redis_client import RedisCache
            rc = RedisCache()
            if rc.enabled:
                # 触发惰性连接建立
                rc.ping()
                self._redis = rc
                self._redis_ready.set()
                logger.debug("对话状态 Redis 预热完成")
        except Exception as e:
            logger.debug("对话状态 Redis 预热失败, 仅内存模式: %s", e)

    def _get_redis(self):
        """获取 Redis 实例 (仅就绪时返回)"""
        if self._redis_ready.is_set():
            return self._redis
        return None

    def get_or_create(self, session_id: str) -> DialogState:
        """获取或创建会话 (L1 内存 → L2 Redis → 新建)"""
        with self._lock:
            # L1: 内存命中
            if session_id in self._sessions:
                return self._sessions[session_id]

            # L2: 从 Redis 恢复
            redis = self._get_redis()
            if redis:
                data = redis.get_dialog_state(session_id)
                if data:
                    state = self._deserialize(session_id, data)
                    self._sessions[session_id] = state
                    return state

            # 新建
            state = DialogState(session_id=session_id)
            self._sessions[session_id] = state
            return state

    def update(self, session_id: str, query: str, answer: str,
               entities: list = None, intent: str = None,
               l0: str = None, l1: str = None, l2: str = None,
               emotion: str = None) -> DialogState:
        """
        更新对话状态 (同步写入内存 + Redis)

        Args:
            session_id: 会话 ID
            query: 用户原始查询
            answer: 系统回答
            entities: 路由阶段提取的实体列表
            intent: 路由阶段识别的意图 (12类平面结构, 兼容旧代码)
            l0: 产品域 (deposit/wealth/fund/insurance/common/out_of_scope)
            l1: 问询类型
            l2: 具体操作
            emotion: 检测到的情绪
        """
        with self._lock:
            state = self.get_or_create(session_id)

            # 话题切换检测: L0 变化时旧话题入栈
            if l0 and state.current_l0 and l0 != state.current_l0:
                state.push_topic()
                state.reset_topic()

            # 更新意图分层
            if l0:
                state.current_l0 = l0
            if l1:
                state.current_l1 = l1
            if l2:
                state.current_l2 = l2
            # 负面轮次累计 (先算: 情绪判定要用到本轮累计后的值)
            from src.core.emotion import is_negative, detect_emotion
            if is_negative(query):
                state.negative_turns += 1
            else:
                state.negative_turns = 0
            state.turn_count += 1

            # 情绪检测: 显式传入优先, 否则按本轮问句 + 连续负面轮次判定
            # (此前 emotion 参数无人传, state.emotion 恒为 None, 下游拿不到情绪)
            state.emotion = emotion or detect_emotion(query, state.negative_turns)

            # 追加历史 (截断 answer 节省 Redis 空间)
            state.history.append({
                "role": "user",
                "query": query,
                "answer": answer[:500] if answer else "",
            })
            if len(state.history) > MAX_HISTORY:
                state.history = state.history[-MAX_HISTORY:]

            # 更新焦点实体
            if entities:
                for ent in entities:
                    etype = ent.get("type", "")
                    ename = ent.get("name", "")
                    if etype == "Company":
                        state.focal_company = ename
                    elif etype == "Metric":
                        state.focal_metric = ename

                # 从实体中提取年份
                for ent in entities:
                    etype = ent.get("type", "")
                    ename = str(ent.get("name", ""))
                    if etype == "Time":
                        import re
                        year_match = re.search(r'(20\d{2})', ename)
                        if year_match:
                            state.focal_year = year_match.group(1)

            # 从查询文本中提取年份（兜底）
            if not state.focal_year:
                import re
                year_match = re.search(r'(20\d{2})', query)
                if year_match:
                    state.focal_year = year_match.group(1)

            # 更新意图
            if intent:
                state.last_intent = intent

            # 客服槽位填充 (从查询文本 + 路由实体提取, 支持多轮累积)
            from src.core.slot_filler import extract_slots
            new_slots = extract_slots(query, entities)
            if new_slots:
                state.slots.update(new_slots)
                # 待补槽位已补齐 → 清除等待标记
                if state.awaiting_slot and state.slots.get(state.awaiting_slot):
                    state.awaiting_slot = None

            # 持久化到 Redis
            self._persist_to_redis(state)

            logger.debug("对话状态更新: l0=%s, l1=%s, intent=%s, neg_turns=%d, turn=%d",
                         state.current_l0, state.current_l1,
                         state.last_intent, state.negative_turns, state.turn_count)
            return state

    def _persist_to_redis(self, state: DialogState):
        """将状态写入 Redis"""
        redis = self._get_redis()
        if redis:
            try:
                data = self._serialize(state)
                redis.set_dialog_state(state.session_id, data, ttl=_DIALOG_REDIS_TTL)
            except Exception as e:
                logger.debug("对话状态 Redis 持久化失败: %s", e)

    @staticmethod
    def _serialize(state: DialogState) -> dict:
        """DialogState → dict (序列化用)"""
        return asdict(state)

    @staticmethod
    def _deserialize(session_id: str, data: dict) -> DialogState:
        """dict → DialogState (反序列化)"""
        return DialogState(
            session_id=session_id,
            history=data.get("history", []),
            focal_company=data.get("focal_company"),
            focal_metric=data.get("focal_metric"),
            focal_year=data.get("focal_year"),
            last_intent=data.get("last_intent"),
            current_l0=data.get("current_l0"),
            current_l1=data.get("current_l1"),
            current_l2=data.get("current_l2"),
            slots=data.get("slots", {}),
            awaiting_slot=data.get("awaiting_slot"),
            topic_stack=data.get("topic_stack", []),
            negative_turns=data.get("negative_turns", 0),
            turn_count=data.get("turn_count", 0),
            emotion=data.get("emotion"),
        )

    def close(self):
        """清理所有会话"""
        with self._lock:
            self._sessions.clear()


# 全局单例
_state_manager = None
_state_lock = threading.Lock()


def get_state_manager() -> DialogStateManager:
    """获取全局 DST 管理器单例"""
    global _state_manager
    if _state_manager is None:
        with _state_lock:
            if _state_manager is None:
                _state_manager = DialogStateManager()
    return _state_manager
