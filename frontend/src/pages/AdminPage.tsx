import React, { useState } from 'react';
import { Card, Table, Tabs, Tag, Button, Space, Typography, Form, Input, Modal, message, Switch, Descriptions } from 'antd';
import { UserOutlined, SettingOutlined, SafetyOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons';
import { useAuthStore } from '../stores/authStore';

const { Title, Text } = Typography;

// 演示用户数据
const DEMO_USERS = [
  { id: '1', username: 'admin', role: 'admin', email: 'admin@example.com', status: 'active', created: '2026-01-15' },
  { id: '2', username: 'service', role: 'analyst', email: 'service@example.com', status: 'active', created: '2026-03-20' },
  { id: '3', username: 'viewer', role: 'viewer', email: 'viewer@example.com', status: 'active', created: '2026-05-10' },
];

const PERMISSION_MATRIX = [
  { role: 'admin', actions: ['知识库读写', '转人工管理', '用户管理', '审计查看', '系统配置'], color: 'red' },
  { role: 'analyst', actions: ['知识库读写', '客服对话', '反馈查看'], color: 'blue' },
  { role: 'viewer', actions: ['知识库只读', '客服对话'], color: 'green' },
];

export const AdminPage: React.FC = () => {
  const { user } = useAuthStore();
  const [users] = useState(DEMO_USERS);
  const [addModalOpen, setAddModalOpen] = useState(false);

  if (user?.role !== 'admin') {
    return (
      <Card>
        <Title level={4}>权限不足</Title>
        <Text>需要管理员权限才能访问系统管理页面。</Text>
      </Card>
    );
  }

  const userColumns = [
    { title: '用户名', dataIndex: 'username', key: 'username' },
    {
      title: '角色', dataIndex: 'role', key: 'role',
      render: (r: string) => (
        <Tag color={r === 'admin' ? 'red' : r === 'analyst' ? 'blue' : 'green'}>{r}</Tag>
      ),
    },
    { title: '邮箱', dataIndex: 'email', key: 'email' },
    {
      title: '状态', dataIndex: 'status', key: 'status',
      render: (s: string) => <Tag color={s === 'active' ? 'success' : 'default'}>{s}</Tag>,
    },
    { title: '创建时间', dataIndex: 'created', key: 'created' },
    {
      title: '操作', key: 'action',
      render: (_: any, record: any) => (
        <Space>
          <Button type="link" size="small">编辑</Button>
          {record.username !== 'admin' && (
            <Button type="link" size="small" danger>禁用</Button>
          )}
        </Space>
      ),
    },
  ];

  const permColumns = [
    { title: '角色', dataIndex: 'role', key: 'role',
      render: (r: string) => <Tag color={PERMISSION_MATRIX.find((p) => p.role === r)?.color}>{r}</Tag>,
    },
    {
      title: '权限', dataIndex: 'actions', key: 'actions',
      render: (actions: string[]) => (
        <Space wrap>
          {actions.map((a) => <Tag key={a}>{a}</Tag>)}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>
        <SettingOutlined /> 系统管理
      </Title>

      <Tabs defaultActiveKey="users" items={[
        {
          key: 'users',
          label: <span><UserOutlined /> 用户管理</span>,
          children: (
            <Card size="small" extra={
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setAddModalOpen(true)}>
                添加用户
              </Button>
            }>
              <Table dataSource={users} columns={userColumns} rowKey="id" size="small" pagination={false} />
            </Card>
          ),
        },
        {
          key: 'permissions',
          label: <span><SafetyOutlined /> 权限矩阵</span>,
          children: (
            <Card size="small">
              <Table dataSource={PERMISSION_MATRIX} columns={permColumns} rowKey="role" size="small" pagination={false} />
            </Card>
          ),
        },
        {
          key: 'config',
          label: <span><SettingOutlined /> 系统配置</span>,
          children: (
            <Card size="small">
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="LLM 模型">按 .env LLM_PROVIDER 配置（Gemini 免费档）</Descriptions.Item>
                <Descriptions.Item label="向量模型">BAAI/bge-base-zh-v1.5 · Milvus(FAISS 降级)</Descriptions.Item>
                <Descriptions.Item label="图谱检索">已接入（Neo4j / 内存邻接多跳）</Descriptions.Item>
                <Descriptions.Item label="缓存">语义缓存 SQLite · Redis (可选)</Descriptions.Item>
                <Descriptions.Item label="日志级别">INFO</Descriptions.Item>
                <Descriptions.Item label="最大上传">50MB</Descriptions.Item>
                <Descriptions.Item label="ETL 并发">Celery</Descriptions.Item>
                <Descriptions.Item label="API 版本">v4.0.0</Descriptions.Item>
              </Descriptions>
            </Card>
          ),
        },
      ]} />

      <Modal
        title="添加用户"
        open={addModalOpen}
        onCancel={() => setAddModalOpen(false)}
        onOk={() => { message.success('用户已添加 (演示)'); setAddModalOpen(false); }}
      >
        <Form layout="vertical">
          <Form.Item label="用户名" required><Input /></Form.Item>
          <Form.Item label="邮箱" required><Input type="email" /></Form.Item>
          <Form.Item label="角色" required>
            <select style={{ width: '100%', padding: 4 }}>
              <option value="admin">管理员</option>
              <option value="analyst">分析师</option>
              <option value="viewer">观察者</option>
            </select>
          </Form.Item>
          <Form.Item label="密码" required><Input.Password /></Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
