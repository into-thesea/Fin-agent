import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { RouterProvider } from 'react-router-dom';
import { router } from './routes';
import { antdTheme } from './styles/theme';
import './styles/global.css';

// 原先这里挂了 react-query 的 Provider, 但全项目零 useQuery/useMutation —— 各页面
// 都是 useEffect + api.xxx。无用依赖已移除; 将来要接服务端状态缓存再装回来。
// 主题统一在 src/styles/theme.ts, 这里只做注入。
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN} theme={antdTheme}>
      <RouterProvider router={router} />
    </ConfigProvider>
  </React.StrictMode>,
);
