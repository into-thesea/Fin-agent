import { createBrowserRouter, Navigate } from 'react-router-dom';
import React, { lazy, Suspense } from 'react';
import { Spin } from 'antd';
import { MainLayout } from './layouts/MainLayout';

// 路由级懒加载：每个页面独立 chunk
// 页面组件均为命名导出，需重映射为 default export 供 React.lazy 使用
const ChatPage = lazy(() => import('./pages/ChatPage').then(m => ({ default: m.ChatPage })));
const DashboardPage = lazy(() => import('./pages/DashboardPage').then(m => ({ default: m.DashboardPage })));
const LLMOpsPage = lazy(() => import('./pages/LLMOpsPage').then(m => ({ default: m.LLMOpsPage })));
const HandoffPage = lazy(() => import('./pages/HandoffPage').then(m => ({ default: m.HandoffPage })));
const BadcasePage = lazy(() => import('./pages/BadcasePage').then(m => ({ default: m.BadcasePage })));
const EvalPage = lazy(() => import('./pages/EvalPage').then(m => ({ default: m.EvalPage })));
const AdminPage = lazy(() => import('./pages/AdminPage').then(m => ({ default: m.AdminPage })));
const LoginPage = lazy(() => import('./pages/LoginPage').then(m => ({ default: m.LoginPage })));
const KnowledgeDocsPage = lazy(() => import('./pages/knowledge/DocsPage').then(m => ({ default: m.DocsPage })));
const KnowledgeChunksPage = lazy(() => import('./pages/knowledge/ChunksPage').then(m => ({ default: m.ChunksPage })));
const KnowledgeStrategyPage = lazy(() => import('./pages/knowledge/StrategyPage').then(m => ({ default: m.StrategyPage })));
const RetrievalTestPage = lazy(() => import('./pages/knowledge/RetrievalTestPage').then(m => ({ default: m.RetrievalTestPage })));

function LazyLoad({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div style={{
          display: 'flex', justifyContent: 'center',
          alignItems: 'center', height: '100%', minHeight: 300,
        }}>
          <Spin size="large" />
        </div>
      }
    >
      {children}
    </Suspense>
  );
}

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <LazyLoad><LoginPage /></LazyLoad>,
  },
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      { path: 'chat', element: <LazyLoad><ChatPage /></LazyLoad> },
      // 知识库分组: 直接访问 /knowledge 落到文档管理
      { path: 'knowledge', element: <Navigate to="/knowledge/docs" replace /> },
      { path: 'knowledge/docs', element: <LazyLoad><KnowledgeDocsPage /></LazyLoad> },
      { path: 'knowledge/chunks', element: <LazyLoad><KnowledgeChunksPage /></LazyLoad> },
      { path: 'knowledge/strategy', element: <LazyLoad><KnowledgeStrategyPage /></LazyLoad> },
      { path: 'knowledge/retrieval-test', element: <LazyLoad><RetrievalTestPage /></LazyLoad> },
      { path: 'dashboard', element: <LazyLoad><DashboardPage /></LazyLoad> },
      { path: 'llmops', element: <LazyLoad><LLMOpsPage /></LazyLoad> },
      { path: 'handoff', element: <LazyLoad><HandoffPage /></LazyLoad> },
      { path: 'badcase', element: <LazyLoad><BadcasePage /></LazyLoad> },
      { path: 'eval', element: <LazyLoad><EvalPage /></LazyLoad> },
      { path: 'admin', element: <LazyLoad><AdminPage /></LazyLoad> },
    ],
  },
]);
