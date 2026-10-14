import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  timestamp: number;
  review?: { score: number | null; verdict: string | null };
  entities?: string | null;
  handoff?: boolean;
  traceId?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

interface ChatState {
  sessions: ChatSession[];
  currentSessionId: string | null;
  isStreaming: boolean;

  // 操作
  createSession: () => string;
  switchSession: (id: string) => void;
  addMessage: (msg: Message) => void;
  updateLastMessage: (content: string) => void;
  updateLastMessageMeta: (patch: Partial<Message>) => void;
  setStreaming: (v: boolean) => void;
  deleteSession: (id: string) => void;
  clearSessions: () => void;
}

// ── 会话持久化: 跨刷新恢复, 7 天内可接着聊 (与后端短期记忆 Redis TTL 对齐) ──
export const CHAT_STORAGE_KEY = 'fin-chat';
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 后端 mem:session:{sid} 滑动窗口 TTL = 7 天
const MAX_PERSIST_SESSIONS = 20;                // localStorage 容量保护: 最多保留会话数
const MAX_CONTENT_LEN = 4000;                   // 单条消息入本地存储前的截断长度

function lastSessionTs(s: ChatSession): number {
  return s.messages.length > 0 ? s.messages[s.messages.length - 1].timestamp : s.createdAt;
}

// 写入/读取 localStorage 前统一清洗:
// 1) 剔除最后活动距今超过 7 天的会话 (与后端记忆窗口一致, 过期后无法真正续上下文)
// 2) 截断超长消息, 避免长回答撑爆 localStorage (~5MB)
// 3) 按最后活动时间排序, 仅保留最近 MAX_PERSIST_SESSIONS 个
// 4) 若当前会话已被清理, 把 currentSessionId 置空 (由页面自动新建)
function sanitizePersisted(sessions: ChatSession[], currentSessionId: string | null) {
  const cutoff = Date.now() - SESSION_TTL_MS;
  const cleaned = sessions
    .filter((s) => lastSessionTs(s) >= cutoff)
    .map((s) => ({
      ...s,
      messages: s.messages.map((m) =>
        m.content.length > MAX_CONTENT_LEN
          ? { ...m, content: m.content.slice(0, MAX_CONTENT_LEN) + '…' }
          : m,
      ),
    }))
    .sort((a, b) => lastSessionTs(b) - lastSessionTs(a))
    .slice(0, MAX_PERSIST_SESSIONS);
  const nextCurrent = cleaned.some((s) => s.id === currentSessionId)
    ? currentSessionId
    : null;
  return { sessions: cleaned, currentSessionId: nextCurrent };
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSessionId: null,
      isStreaming: false,

      createSession: () => {
        const id = crypto.randomUUID();
        const session: ChatSession = {
          id,
          title: '新对话',
          messages: [],
          createdAt: Date.now(),
        };
        set((s) => ({
          sessions: [session, ...s.sessions],
          currentSessionId: id,
        }));
        return id;
      },

      switchSession: (id) => set({ currentSessionId: id }),

      addMessage: (msg) => {
        set((s) => {
          const sessions = s.sessions.map((ses) => {
            if (ses.id !== s.currentSessionId) return ses;
            const messages = [...ses.messages, msg];
            const title = ses.messages.length === 0 && msg.role === 'user'
              ? msg.content.slice(0, 30) : ses.title;
            return { ...ses, messages, title };
          });
          return { sessions };
        });
      },

      updateLastMessage: (content) => {
        set((s) => {
          const sessions = s.sessions.map((ses) => {
            if (ses.id !== s.currentSessionId) return ses;
            const messages = [...ses.messages];
            if (messages.length > 0) {
              const last = { ...messages[messages.length - 1] };
              last.content = content;
              messages[messages.length - 1] = last;
            }
            return { ...ses, messages };
          });
          return { sessions };
        });
      },

      // 合并元数据到最后一条消息 (sources/review/entities/handoff)
      updateLastMessageMeta: (patch) => {
        set((s) => {
          const sessions = s.sessions.map((ses) => {
            if (ses.id !== s.currentSessionId) return ses;
            const messages = [...ses.messages];
            if (messages.length > 0) {
              messages[messages.length - 1] = { ...messages[messages.length - 1], ...patch };
            }
            return { ...ses, messages };
          });
          return { sessions };
        });
      },

      setStreaming: (v) => set({ isStreaming: v }),

      deleteSession: (id) => {
        set((s) => {
          const sessions = s.sessions.filter((ses) => ses.id !== id);
          const currentSessionId = s.currentSessionId === id
            ? (sessions[0]?.id ?? null) : s.currentSessionId;
          return { sessions, currentSessionId };
        });
      },

      clearSessions: () => set({ sessions: [], currentSessionId: null }),
    }),
    {
      name: CHAT_STORAGE_KEY,
      version: 1,
      storage: createJSONStorage(() => localStorage),
      // 只持久化会话数据; isStreaming 是运行时状态, 刷新后必须复位为 false
      partialize: (s) => {
        const { sessions, currentSessionId } = sanitizePersisted(s.sessions, s.currentSessionId);
        return { sessions, currentSessionId };
      },
      // 读取时再做一次清洗, 覆盖"存了超过 7 天才再次打开"的场景; 并强制 isStreaming 复位
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<ChatState>;
        const { sessions, currentSessionId } = sanitizePersisted(
          p.sessions ?? [],
          p.currentSessionId ?? null,
        );
        return { ...current, sessions, currentSessionId, isStreaming: false };
      },
    },
  ),
);
