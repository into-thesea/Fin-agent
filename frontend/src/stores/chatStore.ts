import { create } from 'zustand';

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

export const useChatStore = create<ChatState>()((set, get) => ({
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
}));
