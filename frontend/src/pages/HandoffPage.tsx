import React, { useEffect, useState, useRef, useCallback } from 'react';
import { List, Card, Button, Input, Tag, Typography, Space, Empty, message, Spin, Badge } from 'antd';
import { PhoneOutlined, CheckOutlined, CloseOutlined, RobotOutlined, ReloadOutlined } from '@ant-design/icons';
import { api } from '../services/api';
import type { HandoffTicket, HandoffMessage } from '../types/api';
import { PALETTE } from '../styles/theme';

const { Text, Paragraph } = Typography;
const { TextArea } = Input;

const STATUS_COLOR: Record<string, string> = { open: 'orange', taken: 'blue', closed: 'green' };

export const HandoffPage: React.FC = () => {
  const [tickets, setTickets] = useState<HandoffTicket[]>([]);
  const [selected, setSelected] = useState<HandoffTicket | null>(null);
  const [messages, setMessages] = useState<HandoffMessage[]>([]);
  const [reply, setReply] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const agentId = localStorage.getItem('fin_agent') || '坐席01';
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadQueue = useCallback(async () => {
    try {
      const { tickets: ts } = await api.getHandoffQueue();
      setTickets(ts);
      // 选中项状态刷新
      setSelected((cur) => cur ? ts.find((t) => t.id === cur.id) || null : cur);
    } catch {
      // 静默
    }
  }, []);

  const loadMessages = useCallback(async (id: number) => {
    try {
      const { messages: ms } = await api.getHandoffMessages(id);
      setMessages(ms);
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    loadQueue();
    timerRef.current = setInterval(loadQueue, 5000); // 轮询队列
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [loadQueue]);

  const select = async (t: HandoffTicket) => {
    setSelected(t);
    setMessages([]);
    setReply('');
    await loadMessages(t.id);
  };

  const handleTake = async (t: HandoffTicket) => {
    try {
      await api.takeHandoff(t.id, agentId);
      message.success('已接单');
      await loadQueue();
      await loadMessages(t.id);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '接单失败');
    }
  };

  const handleSuggest = async () => {
    if (!selected) return;
    setSuggesting(true);
    try {
      const { draft } = await api.suggestHandoff(selected.id);
      setReply(draft);
    } catch (e: any) {
      message.error('AI 建议生成失败');
    } finally {
      setSuggesting(false);
    }
  };

  const handleReply = async () => {
    if (!selected || !reply.trim()) return;
    setLoading(true);
    try {
      await api.replyHandoff(selected.id, reply.trim());
      message.success('已回复用户');
      setReply('');
      await loadMessages(selected.id);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '回复失败');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = async () => {
    if (!selected) return;
    try {
      await api.closeHandoff(selected.id);
      message.success('工单已关闭');
      setSelected(null);
      await loadQueue();
    } catch {
      message.error('关闭失败');
    }
  };

  return (
    <div style={{ display: 'flex', gap: 16, height: 'calc(100vh - 120px)' }}>
      {/* 左: 工单队列 */}
      <Card
        title={
          <Space>
            <Badge count={tickets.filter((t) => t.status !== 'closed').length} overflowCount={99} />
            <Text strong>📞 转人工工单队列</Text>
          </Space>
        }
        size="small" style={{ width: 340, overflow: 'auto' }}
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={loadQueue} />}
      >
        {tickets.length === 0 ? (
          <Empty description="暂无待处理工单" />
        ) : (
          <List
            dataSource={tickets}
            renderItem={(t) => (
              <List.Item
                onClick={() => select(t)}
                style={{ cursor: 'pointer', background: selected?.id === t.id ? PALETTE.primarySoft : undefined, padding: '0 8px' }}
                actions={[
                  <Tag color={STATUS_COLOR[t.status] || 'default'} key="s">{t.status}</Tag>,
                  t.status === 'open' ? (
                    <Button key="t" size="small" type="primary" onClick={(e) => { e.stopPropagation(); handleTake(t); }}>
                      <CheckOutlined />接单
                    </Button>
                  ) : null,
                ]}
              >
                <Space direction="vertical" size={0}>
                  <Text>#{t.id} · {t.reason?.slice(0, 30) || '人工咨询'}</Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>{t.created_at} · 用户 {t.user_id}</Text>
                </Space>
              </List.Item>
            )}
          />
        )}
      </Card>

      {/* 右: 工单详情 */}
      <Card size="small" style={{ flex: 1, overflow: 'auto' }} title={selected ? `工单 #${selected.id}` : '工单详情'}>
        {!selected ? (
          <Empty description="选择左侧工单查看详情" style={{ padding: 80 }} />
        ) : (
          <div>
            {/* 工单信息 (点击队列项后的明确反馈) */}
            <div style={{ background: PALETTE.primarySoft, borderRadius: 8, padding: '8px 12px', marginBottom: 16 }}>
              <Paragraph style={{ marginBottom: 4 }}>
                <Tag color={STATUS_COLOR[selected.status]}>{selected.status}</Tag>
                <Text strong> 用户: {selected.user_id}</Text>
                <Text type="secondary"> · 创建于 {selected.created_at}</Text>
              </Paragraph>
              <Paragraph style={{ marginBottom: 0 }}>
                <Text type="secondary">转人工原因:</Text> {selected.reason || '(未记录)'}
              </Paragraph>
            </div>

            {/* 会话上下文 */}
            <Paragraph>
              <Text strong>👤 用户会话上下文：</Text>
            </Paragraph>
            {(!selected.session_context || selected.session_context.length === 0) ? (
              <Text type="secondary">（无历史上下文）</Text>
            ) : (
              <div style={{ background: PALETTE.mutedBg, borderRadius: 8, padding: '8px 12px', marginBottom: 16 }}>
                {selected.session_context.map((c, i) => (
                  <Paragraph key={i} style={{ marginBottom: 6, fontSize: 13 }}>
                    <Text type="secondary">用户:</Text> {c.query}<br />
                    <Text type="secondary">客服:</Text> {(c.answer || '').slice(0, 120)}
                  </Paragraph>
                ))}
              </div>
            )}

            {/* 消息记录 */}
            <Paragraph><Text strong>💬 会话记录：</Text></Paragraph>
            <div style={{ background: PALETTE.mutedBg, borderRadius: 8, padding: '8px 12px', marginBottom: 16, minHeight: 60 }}>
              {messages.length === 0 && <Text type="secondary">暂无坐席回复</Text>}
              {messages.map((m) => (
                <Paragraph key={m.id} style={{ marginBottom: 6, fontSize: 13 }}>
                  <Tag color={m.role === 'agent' ? 'green' : 'default'}>{m.role === 'agent' ? '坐席' : m.role}</Tag>
                  {m.content} <Text type="secondary" style={{ fontSize: 11 }}>{m.created_at}</Text>
                </Paragraph>
              ))}
            </div>

            {/* 回复区 */}
            <Space.Compact style={{ width: '100%' }}>
              <TextArea
                rows={3} value={reply} onChange={(e) => setReply(e.target.value)}
                placeholder="输入坐席回复，或点击 AI 建议生成草稿..."
                disabled={selected.status === 'closed'}
              />
            </Space.Compact>
            <Space style={{ marginTop: 8 }}>
              <Button icon={<RobotOutlined />} loading={suggesting} onClick={handleSuggest} disabled={selected.status === 'closed'}>
                🤖 AI 建议
              </Button>
              <Button type="primary" icon={<CheckOutlined />} loading={loading} onClick={handleReply} disabled={selected.status === 'closed' || !reply.trim()}>
                回复用户
              </Button>
              <Button danger icon={<CloseOutlined />} onClick={handleClose} disabled={selected.status === 'closed'}>
                关闭工单
              </Button>
            </Space>
            {selected.status === 'closed' && <Text type="secondary">（工单已关闭）</Text>}
          </div>
        )}
      </Card>
    </div>
  );
};
