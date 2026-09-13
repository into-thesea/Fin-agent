/**
 * 片段管理 —— 浏览知识库实际切出来的分块
 *
 * 这是"分块到底切成了什么样"的唯一可视入口: 排查检索问题时,
 * 先来这里看目标内容是否真的独立成块、小节标题是否正确。
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Card, Table, Tag, Input, Space, Typography, Drawer, Button, Alert } from 'antd';
import { ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { api } from '../../services/api';
import { PALETTE } from '../../styles/theme';
import type { ChunkItem } from '../../types/api';

const { Text, Paragraph } = Typography;

export const ChunksPage: React.FC = () => {
  const [items, setItems] = useState<ChunkItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [size, setSize] = useState(20);
  const [source, setSource] = useState('');
  const [q, setQ] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<ChunkItem | null>(null);
  // 快速改筛选会连发多个请求, 先发的后到就会拿旧结果盖掉新结果 —— 只认最后一次
  const seqRef = useRef(0);

  const load = useCallback(async () => {
    const seq = ++seqRef.current;
    setLoading(true);
    try {
      const res = await api.listChunks({ page, size, source, q });
      if (seq !== seqRef.current) return;  // 过期响应: 丢弃, loading 交给新请求收尾
      setItems(res.items);
      setTotal(res.total);
      setError(null);
    } catch (err: any) {
      if (seq !== seqRef.current) return;
      // 不静默: 请求失败时表格空着会让人以为"知识库是空的"
      setError(`片段加载失败：${err?.response?.data?.detail || err?.message || '未知错误'}`);
    }
    setLoading(false);
  }, [page, size, source, q]);

  useEffect(() => { load(); }, [load]);

  const columns = [
    { title: '分块 ID', dataIndex: 'chunk_id', key: 'chunk_id', width: 220, ellipsis: true },
    {
      title: '来源', dataIndex: 'source', key: 'source', width: 180, ellipsis: true,
      render: (v: string) => <Tag>{v}</Tag>,
    },
    { title: '小节', dataIndex: 'section', key: 'section', width: 160, ellipsis: true },
    { title: '字数', dataIndex: 'length', key: 'length', width: 80 },
    {
      title: '内容', dataIndex: 'content', key: 'content', ellipsis: true,
      render: (v: string, r: ChunkItem) => (
        <a onClick={() => setDetail(r)}>{v.slice(0, 60)}{v.length > 60 ? '…' : ''}</a>
      ),
    },
  ];

  return (
    <div>
      {error && (
        <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }}
               action={<Button size="small" onClick={load}>重试</Button>} />
      )}

      <Card
        title={`🧩 知识库片段（${total}）`}
        size="small"
        extra={
          <Space>
            <Input
              placeholder="按来源筛选，如 prod_p002.md"
              allowClear style={{ width: 220 }}
              onChange={(e) => { setPage(1); setSource(e.target.value.trim()); }}
            />
            <Input
              placeholder="按内容关键词搜索"
              allowClear prefix={<SearchOutlined />} style={{ width: 220 }}
              onChange={(e) => { setPage(1); setQ(e.target.value.trim()); }}
            />
            <Button icon={<ReloadOutlined />} onClick={load}>刷新</Button>
          </Space>
        }
      >
        <Table
          dataSource={items}
          columns={columns}
          rowKey="chunk_id"
          size="small"
          loading={loading}
          pagination={{
            current: page, pageSize: size, total,
            showSizeChanger: true,
            pageSizeOptions: ['10', '20', '50', '100'],
            onChange: (p, s) => { setPage(p); setSize(s); },
          }}
          locale={{
            // 请求失败时表格必然为空: 提示语也跟着改, 免得被读成"知识库一条分块都没有"
            emptyText: error ? '加载失败，请重试' : '没有匹配的片段',
          }}
        />
      </Card>

      <Drawer
        title="分块详情"
        width={720}
        open={!!detail}
        onClose={() => setDetail(null)}
      >
        {detail && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text><Text strong>分块 ID：</Text><Text code>{detail.chunk_id}</Text></Text>
            <Text><Text strong>来源：</Text>{detail.source}</Text>
            <Text><Text strong>小节：</Text>{detail.section || '（无）'}</Text>
            <Text>
              <Text strong>内容指纹：</Text>
              <Text code style={{ color: PALETTE.textSecondary }}>{detail.content_hash.slice(0, 16)}…</Text>
              <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                内容变化时该值变化，分块 ID 不变
              </Text>
            </Text>
            <Paragraph style={{
              whiteSpace: 'pre-wrap', background: PALETTE.mutedBg,
              padding: 12, borderRadius: 6, maxHeight: '60vh', overflow: 'auto',
            }}>
              {detail.content}
            </Paragraph>
          </Space>
        )}
      </Drawer>
    </div>
  );
};
