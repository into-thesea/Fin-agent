/**
 * 全局视觉主题 —— 单一配色来源
 *
 * 风格: 深色侧栏 + 浅色内容。
 * 任何地方要用颜色都从这里取, 不要在组件里写死十六进制值 ——
 * 否则改一次主题要满仓库找色号(这次换风格就是因为原先 6 个文件里散着 11 处 #1677ff)。
 *
 * 图表色板经过可访问性校验(浅色底): 明度带 / 色度下限 / 色盲相邻对区分度 /
 * 常视觉相邻对区分度 / 与底色的对比度。常视觉相邻对 ΔE 23.3(下限 15),
 * 色盲相邻对 ΔE 9.3(目标 8), 前五槽与底色对比度均 ≥3:1。
 * 第六槽 #0ea5e9 对比度 2.7 偏低, 只在系列超过 5 个时才会用到;
 * 用处(饼图)本身带可见标签, 满足"需可见标签或表格视图兜底"的要求。
 */

import type { ThemeConfig } from 'antd';

export const PALETTE = {
  // 侧栏 (深色)
  sidebar: '#1b2436',
  sidebarSelected: '#2d3b57',
  sidebarText: '#c9d2e3',
  sidebarDivider: '#26314a',

  // 内容区 (浅色)
  surface: '#f5f6f8',
  card: '#ffffff',
  border: '#e5e7eb',
  borderSoft: '#f0f2f5',
  mutedBg: '#f6f8fa',

  // 主色 / 强调
  primary: '#2563eb',
  primarySoft: '#eff4ff',
  accent: '#0ea5e9',

  // 文字
  text: '#1f2937',
  textSecondary: '#6b7280',
  textMuted: '#9ca3af',
} as const;

/** 图表分类色板 —— 按固定顺序取用, 不要循环复用 */
export const CHART_COLORS = [
  '#2563eb', // 主蓝
  '#b45309', // 琥珀
  '#059669', // 绿
  '#7c3aed', // 紫
  '#dc2626', // 红
  '#0ea5e9', // 强调青 (第 6 槽起才用到)
] as const;

/** 图表通用底样式: 网格与坐标轴保持克制, 不与数据争视线 */
export const CHART_BASE = {
  color: [...CHART_COLORS],
  textStyle: { color: PALETTE.textSecondary, fontSize: 12 },
  grid: { left: 48, right: 24, top: 32, bottom: 40, containLabel: true },
} as const;

export const antdTheme: ThemeConfig = {
  token: {
    colorPrimary: PALETTE.primary,
    colorInfo: PALETTE.primary,
    colorLink: PALETTE.primary,
    colorText: PALETTE.text,
    colorTextSecondary: PALETTE.textSecondary,
    colorBorder: PALETTE.border,
    colorBorderSecondary: PALETTE.borderSoft,
    borderRadius: 8,
    fontSize: 14,
  },
  components: {
    Layout: {
      bodyBg: PALETTE.surface,
      headerBg: PALETTE.card,
      siderBg: PALETTE.sidebar,
      headerHeight: 64,
      headerPadding: '0 24px',
    },
    Menu: {
      darkItemBg: PALETTE.sidebar,
      darkSubMenuItemBg: PALETTE.sidebar,
      darkItemColor: PALETTE.sidebarText,
      darkItemHoverBg: PALETTE.sidebarSelected,
      darkItemSelectedBg: PALETTE.sidebarSelected,
      darkItemSelectedColor: '#ffffff',
      itemBorderRadius: 6,
      itemMarginInline: 8,
    },
    Card: { borderRadiusLG: 10 },
    Table: {
      headerBg: '#f8fafc',
      headerColor: PALETTE.text,
      borderColor: PALETTE.borderSoft,
    },
  },
};
