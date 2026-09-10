import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Input, Button, List, Tag, Typography, Space, Spin, Empty, message } from 'antd';
import { SendOutlined, DeleteOutlined, PlusOutlined, RobotOutlined, ReloadOutlined, LikeOutlined, DislikeOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { useChatStore } from '../stores/chatStore';
import { api } from '../services/api';

const { TextArea } = Input;
const { Text, Title } = Typography;

// 加载阶段消息（当流式/同步 API 处理时展示）
const STAGE_MESSAGES = [
  '🔍 正在理解您的问题...',
  '📚 正在检索理财知识库...',
  '🧠 正在生成回答...',
  '✅ 正在核验回答准确性...',
];

// P9: LangGraph 节点 → 进度时间线标签 (客户可见, 不出现技术术语)
const STAGE_LABELS: Record<string, string> = {
  rewriting: '上下文理解',
  classify_node: '意图识别',
  memory_node: '记忆读取',
  route_retrieve_node: '知识检索',
  slot_check_node: '槽位检查',
  orchestrator_node: '组织回答',
  finance_node: '理财问答',
  subscribe_node: '申购门控',
  handoff_node: '转人工坐席',
  chitchat_node: '闲聊',
  general_node: '通用问答',
  reviewer_node: '回答核验',
  regenerate_node: '重试修正',
  formatter_node: '格式化',
  greeting_node: '问候',
  simple_fact_node: '快速问答',
  unknown_node: '处理中',
};

// SSE 超时时间 (毫秒)
const STREAM_TIMEOUT = 30_000;

// 同步回退超时
const SYNC_TIMEOUT = 180_000;

export const ChatPage: React.FC = () => {
  const {
    sessions, currentSessionId, isStreaming,
    createSession, switchSession, addMessage, updateLastMessage, updateLastMessageMeta,
    setStreaming, deleteSession, clearSessions,
  } = useChatStore();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<any>(null);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const mountedRef = useRef(true);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null); // 转人工坐席回复轮询

  // 本地 UI 状态：当前阶段提示 / 错误信息
  const [stageMessage, setStageMessage] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  // P9: 多 Agent 协同时间线 (节点流转序列)
  const [agentFlow, setAgentFlow] = useState<{ key: string; label: string }[]>([]);
  // 输入框受控值 (发送后清空)
  const [inputValue, setInputValue] = useState('');

  const currentSession = sessions.find((s) => s.id === currentSessionId);
  const messages = currentSession?.messages ?? [];

  // 注册/清理定时器的工具函数
  const addTimer = useCallback((timer: ReturnType<typeof setTimeout>) => {
    timersRef.current.push(timer);
  }, []);

  const clearAllTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  // 组件挂载状态追踪
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearAllTimers();
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [clearAllTimers]);

  // ─── 转人工: 轮询坐席回复 (人机协同闭环) ──────
  const pollHandoff = useCallback((ticketId: number) => {
    if (!mountedRef.current) return;
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    let lastId = 0;
    pollTimerRef.current = setInterval(async () => {
      if (!mountedRef.current) return;
      try {
        const { messages } = await api.getHandoffMessages(ticketId);
        const newMsgs = messages.filter((m) => m.id > lastId);
        if (newMsgs.length > 0) {
          lastId = messages[messages.length - 1].id;
          newMsgs.forEach((m) => {
            if (m.role === 'agent') {
              addMessage({
                id: crypto.randomUUID(), role: 'assistant' as const,
                content: `🧑‍💼 **人工坐席**：${m.content}`,
                timestamp: Date.now(),
              });
            }
          });
        }
      } catch {
        // 网络/服务异常静默, 下轮重试
      }
    }, 4000);
  }, [addMessage]);

  // 自动创建首个会话
  useEffect(() => {
    if (!currentSessionId && sessions.length === 0) {
      createSession();
    }
  }, [currentSessionId, sessions.length, createSession]);

  // 自动滚动到最新消息
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ─── 异步重试：同步 API 回退 ──────────────────
  const trySyncFallback = useCallback(async (query: string) => {
    if (!mountedRef.current) return;

    updateLastMessage('⏳ 正在生成分析，请稍候...');

    let stageIdx = 0;
    const stageTimer = setInterval(() => {
      if (stageIdx < STAGE_MESSAGES.length && mountedRef.current) {
        updateLastMessage(STAGE_MESSAGES[stageIdx]);
        stageIdx++;
      }
    }, 4000);
    addTimer(stageTimer);

    try {
      const result = await api.chat(query, 60000, currentSessionId || '');
      if (mountedRef.current) {
        updateLastMessage(result.answer);
        updateLastMessageMeta({
          sources: result.sources,
          review: result.review,
          entities: result.entities,
          handoff: !!result.handoff,
          traceId: result.trace_id,
        });
        setStageMessage(result.handoff ? '🛟 已为您转接人工客服，请稍候...' : '');
        setErrorMessage('');
      }
    } catch (err: any) {
      if (!mountedRef.current) return;
      const errMsg = err?.response?.data?.detail || err?.message || '服务暂不可用';
      const display = `❌ **请求失败**\n\n> ${errMsg}\n\n请检查后端服务是否正常运行，或稍后重试。`;
      updateLastMessage(display);
      setErrorMessage(errMsg);
      setStageMessage('');
    } finally {
      clearInterval(stageTimer);
      if (mountedRef.current) {
        setStreaming(false);
      }
    }
  }, [addTimer, updateLastMessage, updateLastMessageMeta, setStreaming, currentSessionId]);

  // ─── 发送消息 ─────────────────────────────────
  const handleSend = useCallback(async () => {
    const value = inputValue;
    if (!value?.trim() || isStreaming) return;

    let sessionId = currentSessionId;
    if (!sessionId) {
      sessionId = createSession();
    }

    const query = value.trim();

    // 添加用户消息
    const userMsg = {
      id: crypto.randomUUID(),
      role: 'user' as const,
      content: query,
      timestamp: Date.now(),
    };
    addMessage(userMsg);

    // 清空输入框 (受控组件)
    setInputValue('');

    // 添加空的助手消息占位
    const assistantId = crypto.randomUUID();
    const assistantMsg = {
      id: assistantId,
      role: 'assistant' as const,
      content: '',
      timestamp: Date.now(),
    };
    addMessage(assistantMsg);

    setStreaming(true);
    setStageMessage('🔍 正在连接服务...');
    setErrorMessage('');
    setAgentFlow([]); // P9: 每次发送重置 Agent 时间线
    if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; }

    let resolved = false;

    // ── 路径 A: SSE 流式 ──────────────────────
    const controller = api.chatStream(
      query,
      // onResult: 收到最终结果
      (result) => {
        if (!mountedRef.current || resolved) return;
        resolved = true;
        clearAllTimers();
        updateLastMessage(result.answer);
        updateLastMessageMeta({
          sources: result.sources,
          review: result.review,
          entities: result.entities,
          handoff: !!result.handoff,
          traceId: result.trace_id,
        });
        setStageMessage(result.handoff ? '🛟 已为您转接人工客服，请稍候...' : '');
        setStreaming(false);
      },
      // onStage: 阶段更新 → 累计 Agent 时间线 (P9)
      (stage, stageMsg) => {
        if (!mountedRef.current || resolved) return;
        setStageMessage(stageMsg);
        const label = STAGE_LABELS[stage] || stage;
        setAgentFlow((prev) =>
          prev.some((s) => s.label === label) ? prev : [...prev, { key: `${stage}-${prev.length}`, label }],
        );
        updateLastMessage(stageMsg + '\n\n_⏳ 正在处理，请稍候..._');
      },
      // onError: 流式出错 → 立即切同步回退
      (error) => {
        if (!mountedRef.current || resolved) return;
        console.warn('SSE 流错误，切换到同步回退:', error);
        resolved = true;
        clearAllTimers();
        controller.abort();
        setStageMessage('⏳ 流式响应异常，切换至同步模式...');
        trySyncFallback(query);
      },
      sessionId,             // 会话 ID (多轮上下文)
      (ticketId) => {        // onHandoff: 转人工 → 轮询坐席回复
        if (!mountedRef.current) return;
        setStageMessage('🛟 已为您转接人工客服，请稍候...');
        if (ticketId) pollHandoff(ticketId);
      },
      (sources) => {         // onMeta: 提前显示来源
        if (!mountedRef.current || resolved) return;
        updateLastMessageMeta({ sources });
      },
    );

    // ── 流式超时回退 ──────────────────────────
    const timeoutTimer = setTimeout(() => {
      if (resolved || !mountedRef.current) return;
      console.log('SSE 超时，切换到同步回退');
      controller.abort();
      clearAllTimers();
      trySyncFallback(query);
    }, STREAM_TIMEOUT);
    addTimer(timeoutTimer);

    // ── 初始阶段动画（SSE 可能延迟） ──────────
    let stageIdx = 0;
    const stageTimer = setInterval(() => {
      if (resolved || !mountedRef.current) return;
      if (stageIdx < STAGE_MESSAGES.length) {
        updateLastMessage(STAGE_MESSAGES[stageIdx] + '\n\n_⏳ 正在处理，请稍候..._');
        stageIdx++;
      }
    }, 4000);
    addTimer(stageTimer);
  }, [
    currentSessionId, isStreaming, createSession, addMessage,
    updateLastMessage, updateLastMessageMeta, setStreaming,
    trySyncFallback, clearAllTimers, addTimer, inputValue,
  ]);

  // ─── 重试 ─────────────────────────────────────
  const handleRetry = useCallback(() => {
    // 找到最后一条用户消息重新发送
    const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user');
    if (lastUserMsg) {
      setInputValue(lastUserMsg.content);
      // 删除最后几条消息（用户+失败的助手回复）
      const session = sessions.find((s) => s.id === currentSessionId);
      if (session) {
        const msgs = session.messages;
        const lastAssistantIdx = msgs.length - 1;
        if (msgs[lastAssistantIdx]?.role === 'assistant') {
          // 删除助手回复
          useChatStore.setState((s) => ({
            sessions: s.sessions.map((ses) => {
              if (ses.id !== currentSessionId) return ses;
              return {
                ...ses,
                messages: ses.messages.slice(0, lastAssistantIdx),
              };
            }),
          }));
        }
      }
      // 重新发送
      setTimeout(() => handleSend(), 100);
    }
  }, [messages, sessions, currentSessionId, handleSend]);

  // ─── 回答反馈 (👍/👎) ─────────────────────────
  const handleFeedback = useCallback(async (rating: number, traceId?: string) => {
    if (!currentSessionId) return;
    try {
      await api.sendFeedback(currentSessionId, rating, '', traceId || '');
      message.success(rating > 0 ? '感谢您的认可！' : '已记录，我们会持续改进');
    } catch {
      message.warning('反馈提交失败');
    }
  }, [currentSessionId]);

  // ─── 键盘快捷发送 ─────────────────────────────
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  return (
    <div style={{ display: 'flex', height: '100%', gap: 16 }}>
      {/* ─── 会话侧边栏 ─────────────────────────── */}
      <div style={{
        width: 240, flexShrink: 0,
        borderRight: '1px solid #f0f0f0', paddingRight: 16,
        display: 'flex', flexDirection: 'column',
      }}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          block
          onClick={() => createSession()}
          style={{ marginBottom: 12 }}
        >
          新对话
        </Button>
        <div style={{ flex: 1, overflow: 'auto' }}>
          {sessions.length === 0 ? (
            <Text type="secondary" style={{ display: 'block', textAlign: 'center', marginTop: 32 }}>
              暂无会话
            </Text>
          ) : (
            <List
              size="small"
              dataSource={sessions}
              renderItem={(item) => (
                <List.Item
                  onClick={() => switchSession(item.id)}
                  style={{
                    cursor: 'pointer',
                    background: item.id === currentSessionId ? '#e6f4ff' : undefined,
                    borderRadius: 4, padding: '4px 8px',
                    transition: 'background 0.2s',
                  }}
                  actions={[
                    <Button
                      key="del" type="text" size="small" danger
                      icon={<DeleteOutlined />}
                      onClick={(e) => { e.stopPropagation(); deleteSession(item.id); }}
                    />,
                  ]}
                >
                  <Text ellipsis style={{ maxWidth: 150 }}>{item.title}</Text>
                </List.Item>
              )}
            />
          )}
        </div>
      </div>

      {/* ─── 对话主区域 ─────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* 消息列表 */}
        <div style={{ flex: 1, overflow: 'auto', padding: '0 16px' }}>
          {messages.length === 0 ? (
            <div style={{ textAlign: 'center', marginTop: 120 }}>
              <RobotOutlined style={{ fontSize: 64, color: '#1677ff' }} />
              <Title level={4} style={{ marginTop: 16 }}>理财智能客服</Title>
              <Text type="secondary">理财咨询 · 存款保险 · 收益口径 · 基金保险 · 适当性</Text>
              <div style={{ marginTop: 32, textAlign: 'left', maxWidth: 440, margin: '32px auto 0' }}>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  <strong>💡 试试这些问题：</strong><br />
                  • 银行理财是存款吗？存款保险赔吗？<br />
                  • 稳盈添利30天保本吗？<br />
                  • 我测评R2能买平衡增利180天吗？<br />
                  • 大额存单和固收理财哪个适合我？<br />
                  • 增额终身寿买了能退吗？
                </Text>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={msg.id} style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <Tag
                    color={msg.role === 'user' ? 'blue' : 'green'}
                    style={{ marginTop: 4, flexShrink: 0 }}
                  >
                    {msg.role === 'user' ? '你' : 'AI'}
                  </Tag>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 14, lineHeight: 1.8 }}>
                      <ReactMarkdown>{msg.content || ''}</ReactMarkdown>
                      {/* 正在加载时显示闪烁光标 */}
                      {isStreaming && idx === messages.length - 1 && msg.role === 'assistant' && (
                        <span className="typing-cursor" />
                      )}
                    </div>
                    {/* 溯源证据链 */}
                    {msg.sources && msg.sources.length > 0 && (
                      <details style={{ marginTop: 8 }}>
                        <summary style={{ cursor: 'pointer', color: '#1677ff', fontSize: 13 }}>
                          溯源证据链 ({msg.sources.length})
                          {msg.sources.some((s) => String(s).startsWith('知识图谱')) ? ' · 🕸️ 含图谱推理路径' : ''}
                        </summary>
                        {msg.sources.map((s, i) => (
                          <div key={i} className="evidence-card">
                            {String(s).startsWith('知识图谱') ? '🕸️' : '📄'} {s}
                          </div>
                        ))}
                      </details>
                    )}
                    {/* 转人工提示 */}
                    {msg.handoff && (
                      <div style={{
                        marginTop: 8, padding: '8px 12px', background: '#fff7e6',
                        border: '1px solid #ffd591', borderRadius: 6,
                        color: '#ad6800', fontSize: 13,
                      }}>
                        🛟 正在为您转接人工客服，请稍候…
                      </div>
                    )}
                    {/* 错误消息显示重试按钮 */}
                    {msg.role === 'assistant' && msg.content.includes('❌') && !isStreaming && (
                      <Button
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={handleRetry}
                        style={{ marginTop: 8 }}
                      >
                        重试
                      </Button>
                    )}
                    {/* 点赞/踩反馈 */}
                    {msg.role === 'assistant' && msg.content && !isStreaming
                      && !msg.content.includes('❌') && !msg.content.includes('⏳') && (
                      <Space size={4} style={{ marginTop: 8 }}>
                        <Button size="small" type="text" icon={<LikeOutlined />} onClick={() => handleFeedback(1, msg.traceId)} />
                        <Button size="small" type="text" icon={<DislikeOutlined />} onClick={() => handleFeedback(-1, msg.traceId)} />
                      </Space>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* P9: 多 Agent 协同时间线 */}
        {agentFlow.length > 0 && (
          <div style={{
            padding: '6px 16px 0',
            fontSize: 12,
            color: '#888',
            textAlign: 'center',
          }}>
            <Space size={[4, 4]} wrap style={{ justifyContent: 'center' }}>
              {agentFlow.map((s, i) => (
                <React.Fragment key={s.key}>
                  {i > 0 && <span style={{ color: '#ccc' }}>→</span>}
                  <Tag color="blue" style={{ marginInlineEnd: 0 }}>{s.label}</Tag>
                </React.Fragment>
              ))}
            </Space>
          </div>
        )}

        {/* 阶段提示条 */}
        {stageMessage && (
          <div style={{
            padding: '6px 16px',
            background: '#f6f8fa',
            borderTop: '1px solid #f0f0f0',
            fontSize: 13,
            color: '#666',
            textAlign: 'center',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
          }}>
            <Spin size="small" />
            <span>{stageMessage}</span>
          </div>
        )}

        {/* 输入区域 */}
        <div style={{ borderTop: stageMessage ? undefined : '1px solid #f0f0f0', padding: '12px 0' }}>
          <TextArea
            ref={inputRef}
            rows={3}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="请输入您的问题（理财 / 存款 / 基金保险 / 收益口径 / 适当性）..."
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, alignItems: 'center' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Enter 发送 · Shift+Enter 换行
            </Text>
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSend}
              loading={isStreaming}
            >
              {isStreaming ? '处理中...' : '发送'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
