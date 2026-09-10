import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
      },
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        manualChunks: {
          // antd 组件库 (~2.4MB → 独立缓存，版本更新不波及业务)
          'vendor-antd': ['antd', '@ant-design/icons', '@ant-design/pro-layout'],
          // 图表库
          'vendor-echarts': ['echarts', 'echarts-for-react'],
          // React 运行时
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // 数据层
          'vendor-data': ['@tanstack/react-query', 'zustand', 'axios'],
        },
      },
    },
  },
});
