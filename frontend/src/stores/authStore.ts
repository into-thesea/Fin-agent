import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { CHAT_STORAGE_KEY } from './chatStore';

interface User {
  id: string;
  username: string;
  role: 'admin' | 'analyst' | 'viewer' | 'user';
  avatar?: string;
}

interface AuthState {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  setAuth: (token: string, user: User) => void;
  logout: () => void;
  hasRole: (roles: string[]) => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      isAuthenticated: false,

      setAuth: (token, user) => {
        localStorage.setItem('fin_token', token);
        set({ token, user, isAuthenticated: true });
      },

      logout: () => {
        localStorage.removeItem('fin_token');
        // 金融隐私: 退出时清掉本地缓存的聊天会话, 避免下一个账号看到上一个人的对话
        localStorage.removeItem(CHAT_STORAGE_KEY);
        set({ token: null, user: null, isAuthenticated: false });
        window.location.href = '/login';
      },

      hasRole: (roles) => {
        const user = get().user;
        return user ? roles.includes(user.role) : false;
      },
    }),
    { name: 'fin-auth' },
  ),
);
