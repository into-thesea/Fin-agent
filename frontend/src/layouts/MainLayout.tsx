import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation, Navigate } from 'react-router-dom';
import { Layout, Menu, Button, Avatar, Dropdown, Typography, theme } from 'antd';
import {
  RobotOutlined, FolderOpenOutlined, DashboardOutlined,
  SettingOutlined, BugOutlined, UserOutlined, PhoneOutlined,
  MenuFoldOutlined, MenuUnfoldOutlined, LogoutOutlined, AlertOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../stores/authStore';
import { PALETTE } from '../styles/theme';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/chat', icon: <RobotOutlined />, label: '理财咨询' },
  {
    key: '/knowledge',
    icon: <FolderOpenOutlined />,
    label: '知识库管理',
    children: [
      { key: '/knowledge/docs', label: '文档管理' },
      { key: '/knowledge/chunks', label: '片段管理' },
      { key: '/knowledge/strategy', label: '策略配置' },
      { key: '/knowledge/retrieval-test', label: '检索调试台' },
    ],
  },
  { key: '/dashboard', icon: <DashboardOutlined />, label: '数据看板' },
  { key: '/llmops', icon: <BugOutlined />, label: '服务监控' },
  { key: '/eval', icon: <ExperimentOutlined />, label: '评测看板' },
  { key: '/badcase', icon: <AlertOutlined />, label: '质量复盘' },
  { key: '/handoff', icon: <PhoneOutlined />, label: '坐席工作台' },
  { key: '/admin', icon: <SettingOutlined />, label: '系统管理' },
];

export const MainLayout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const { token: { colorBgContainer, colorBorderSecondary, borderRadiusLG } } = theme.useToken();

  // 如果没有登录，跳转登录页
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  const userMenu = {
    items: [
      { key: 'profile', icon: <UserOutlined />, label: `${user.username} (${user.role})` },
      { type: 'divider' as const },
      { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', danger: true },
    ],
    onClick: ({ key }: { key: string }) => {
      if (key === 'logout') logout();
    },
  };

  // 双端菜单: admin=全部 / analyst=客服+监控+坐席+评测 / user=仅客服
  const roleAllowed = (key: string) =>
    user.role === 'admin' ||
    (user.role === 'analyst'
      ? ['/chat', '/llmops', '/handoff', '/eval'].includes(key)
      : key === '/chat');

  // 带子项的分组: 子项全被过滤掉时父项也必须隐藏 —— 否则留一个点了没反应的死菜单
  const visibleMenu = menuItems
    .map((m: any) => (m.children
      ? { ...m, children: m.children.filter((c: any) => roleAllowed(c.key)) }
      : m))
    .filter((m: any) => (m.children ? m.children.length > 0 : roleAllowed(m.key)));

  // 守卫用前缀匹配: /knowledge/chunks 属于 /knowledge 组。
  // 父键也进列表 —— 谁能看见该分组, 谁就能访问它的父路径(用于 /knowledge → /knowledge/docs 的重定向)。
  const allowedPrefixes = visibleMenu.flatMap((m: any) =>
    m.children ? [...m.children.map((c: any) => c.key), m.key] : [m.key]);
  const allowed = allowedPrefixes.some(
    (p) => location.pathname === p || location.pathname.startsWith(p + '/'),
  );
  if (!allowed) {
    return <Navigate to="/chat" replace />;
  }

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider trigger={null} collapsible collapsed={collapsed} theme="dark" width={216}>
        <div
          className="fa-sider-logo"
          style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          <Text strong style={{ fontSize: collapsed ? 16 : 18, color: '#ffffff', letterSpacing: 1 }}>
            {collapsed ? '客服' : '理财智能客服'}
          </Text>
        </div>
        <Menu
          mode="inline"
          theme="dark"
          selectedKeys={[location.pathname]}
          defaultOpenKeys={location.pathname.startsWith('/knowledge') ? ['/knowledge'] : []}
          items={visibleMenu}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 0, marginTop: 8 }}
        />
      </Sider>
      <Layout>
        <Header style={{
          padding: '0 24px', background: colorBgContainer,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          borderBottom: `1px solid ${colorBorderSecondary}`, height: 64,
        }}>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
          />
          <Dropdown menu={userMenu} placement="bottomRight">
            <Avatar
              icon={<UserOutlined />}
              style={{ cursor: 'pointer', backgroundColor: PALETTE.primary }}
            />
          </Dropdown>
        </Header>
        <Content style={{ margin: 16, overflow: 'auto' }}>
          <div style={{
            padding: 16, minHeight: '100%',
            background: colorBgContainer, borderRadius: borderRadiusLG,
            boxShadow: '0 1px 2px rgba(16, 24, 40, 0.04)',
          }}>
            <Outlet />
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};
