import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Form, Input, Button, Typography, message, Space } from 'antd';
import { UserOutlined, LockOutlined, SafetyOutlined } from '@ant-design/icons';
import { api } from '../services/api';
import { useAuthStore } from '../stores/authStore';
import { PALETTE } from '../styles/theme';

const { Title, Text } = Typography;

export const LoginPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { setAuth } = useAuthStore();

  const onFinish = async (values: { username: string; password: string }) => {
    setLoading(true);
    try {
      const res = await api.login(values.username, values.password);
      setAuth(res.token, res.user);
      message.success(`登录成功 (${res.user.role})`);
      navigate('/chat', { replace: true });
    } catch {
      message.error('用户名或密码错误，或使用下方演示入口');
    } finally {
      setLoading(false);
    }
  };

  // 双端演示入口: 管理员 / 用户
  const demoLogin = (role: 'admin' | 'user') => {
    setAuth(`demo-${role}`, {
      id: 'demo', username: role === 'admin' ? '管理员演示' : '用户演示',
      role,
    });
    message.success(`演示模式登录成功 (${role === 'admin' ? '管理员端' : '用户端'})`);
    navigate('/chat', { replace: true });
  };

  return (
    <div className="login-bg" style={{
      height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center',
    }}>
      <Card style={{ width: 400, borderRadius: 12, boxShadow: '0 8px 24px rgba(0,0,0,0.15)' }}>
        <Space direction="vertical" size="large" style={{ width: '100%', textAlign: 'center' }}>
          <SafetyOutlined style={{ fontSize: 48, color: PALETTE.primary }} />
          <div>
            <Title level={3} style={{ margin: 0 }}>理财智能客服系统</Title>
            <Text type="secondary">理财产品 · 存款保险 · 基金保险 · 适当性合规</Text>
          </div>
        </Space>
        <Form
          name="login"
          onFinish={onFinish}
          layout="vertical"
          style={{ marginTop: 24 }}
          size="large"
        >
          <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
            <Input prefix={<UserOutlined />} placeholder="用户名" />
          </Form.Item>
          <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
            <Input.Password prefix={<LockOutlined />} placeholder="密码" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} block>
              登录 / 演示模式
            </Button>
          </Form.Item>
        </Form>
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <Button block icon={<SafetyOutlined />} onClick={() => demoLogin('admin')}>👨‍💼 管理员端演示</Button>
          <Button block icon={<UserOutlined />} onClick={() => demoLogin('user')}>👤 用户端演示</Button>
          <Text type="secondary" style={{ display: 'block', textAlign: 'center', fontSize: 12 }}>
            真实账号: 管理员 admin/admin123 · 用户 user/user123
          </Text>
        </Space>
      </Card>
    </div>
  );
};
