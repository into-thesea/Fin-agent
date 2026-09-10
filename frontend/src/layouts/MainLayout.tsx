import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation, Navigate } from 'react-router-dom';
import { Layout, Menu, Button, Avatar, Dropdown, Typography, theme } from 'antd';
import {
  RobotOutlined, FolderOpenOutlined, DashboardOutlined,
  SettingOutlined, BugOutlined, UserOutlined, PhoneOutlined,
  MenuFoldOutlined, MenuUnfoldOutlined, LogoutOutlined, AlertOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../stores/authStore';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/chat', icon: <RobotOutlined />, label: '理财咨询' },
  { key: '/knowledge', icon: <FolderOpenOutlined />, label: '知识库管理' },
  { key: '/dashboard', icon: <DashboardOutlined />, label: '数据看板' },
  { key: '/llmops', icon: <BugOutlined />, label: '服务监控' },
  { key: '/badcase', icon: <AlertOutlined />, label: '质量复盘' },
  { key: '/handoff', icon: <PhoneOutlined />, label: '坐席工作台' },
  { key: '/admin', icon: <SettingOutlined />, label: '系统管理' },
];

export const MainLayout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const { token: { colorBgContainer, borderRadiusLG } } = theme.useToken();

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

  // 双端菜单: admin=全部 / analyst=客服+监控+坐席 / user=仅客服
  const visibleMenu = user.role === 'admin' ? menuItems
    : user.role === 'analyst'
      ? menuItems.filter((m) => ['/chat', '/llmops', '/handoff'].includes(m.key))
      : [menuItems[0]];

  // 路由守卫: 当前路径不在本角色允许范围内 → 强制回 /chat
  const allowedPaths = new Set(visibleMenu.map((m) => m.key));
  if (!allowedPaths.has(location.pathname)) {
    return <Navigate to="/chat" replace />;
  }

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        theme="light"
        style={{ borderRight: '1px solid #f0f0f0' }}
      >
        <div style={{
          height: 64, display: 'flex', alignItems: 'center',
          justifyContent: 'center', borderBottom: '1px solid #f0f0f0',
        }}>
          <Text strong style={{ fontSize: collapsed ? 16 : 18, color: '#1677ff' }}>
            {collapsed ? '客服' : '理财智能客服'}
          </Text>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={visibleMenu}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 0 }}
        />
      </Sider>
      <Layout>
        <Header style={{
          padding: '0 24px', background: colorBgContainer,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          borderBottom: '1px solid #f0f0f0', height: 64,
        }}>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
          />
          <Dropdown menu={userMenu} placement="bottomRight">
            <Avatar icon={<UserOutlined />} style={{ cursor: 'pointer', backgroundColor: '#1677ff' }} />
          </Dropdown>
        </Header>
        <Content style={{ margin: 16, overflow: 'auto' }}>
          <div style={{
            padding: 16, minHeight: '100%',
            background: colorBgContainer, borderRadius: borderRadiusLG,
          }}>
            <Outlet />
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};
