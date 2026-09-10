/**
 * 质量复盘 — 待处理的问题回答列表 + 状态跟进 + 沉淀为回归用例
 *
 * 候选范围由后端 SQL 派生 (用户点踩 / 类型未识别 / 未找到资料 / 审核未通过 / 把握不足),
 * 前端只负责展示、标状态、沉淀。界面文案面向业务人员, 不出现技术术语。
 */
import { useEffect, useState, useCallback } from 'react';
import {
  Table, Tag, Drawer, Button, Space, Select, Input, Descriptions,
  Typography, message, Alert, Empty,
} from 'antd';
import { api } from '../services/api';

const REASON_LABEL: Record<string, { text: string; color: string }> = {
  downvote: { text: '用户点踩', color: 'red' },
  unknown_intent: { text: '类型未识别', color: 'orange' },
  empty_retrieval: { text: '未找到资料', color: 'volcano' },
  reject: { text: '审核未通过', color: 'magenta' },
  low_confidence: { text: '把握不足', color: 'gold' },
};

const STATUS_LABEL: Record<string, { text: string; color: string }> = {
  new: { text: '待处理', color: 'default' },
  confirmed: { text: '已确认', color: 'blue' },
  fixed: { text: '已修复', color: 'green' },
  ignored: { text: '已忽略', color: 'default' },
};

interface Row {
  trace_id: string;
  query: string;
  intent: string;
  confidence: number | null;
  sources_n: number;
  review_verdict: string;
  triage_status: string;
  triage_note: string;
  created_at: string;
  reasons: string[];
}

export function BadcasePage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [detail, setDetail] = useState<any>(null);
  const [note, setNote] = useState('');
  const [open, setOpen] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listBadcases(status);
      setRows(data.items || []);
    } catch {
      message.error('加载 badcase 列表失败');
    } finally {
      setLoading(false);
    }
  }, [status]);

  useEffect(() => { load(); }, [load]);

  const openDetail = async (traceId: string) => {
    try {
      const d = await api.getBadcase(traceId);
      setDetail(d);
      setNote(d.triage_note || '');
      setOpen(true);
    } catch {
      message.error('加载详情失败');
    }
  };

  const setTriage = async (s: string) => {
    try {
      await api.patchBadcase(detail.trace_id, s, note);
      message.success('已更新');
      setOpen(false);
      load();
    } catch {
      message.error('更新失败');
    }
  };

  const promote = async () => {
    try {
      const out = await api.promoteBadcase(detail.trace_id, note);
      message.success(
        out.already ? `已在回归用例中 (${out.id})` : `已加入回归用例 ${out.id}，请补引用资料`,
      );
      setOpen(false);
      load();
    } catch {
      message.error('沉淀失败');
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <Space style={{ marginBottom: 12 }}>
        <Typography.Title level={4} style={{ margin: 0 }}>质量复盘</Typography.Title>
        <Select
          value={status}
          style={{ width: 140 }}
          onChange={setStatus}
          options={[
            { value: '', label: '全部状态' },
            { value: 'new', label: '待处理' },
            { value: 'confirmed', label: '已确认' },
            { value: 'fixed', label: '已修复' },
            { value: 'ignored', label: '已忽略' },
          ]}
        />
        <Button onClick={load}>刷新</Button>
      </Space>

      <Table<Row>
        rowKey="trace_id"
        loading={loading}
        dataSource={rows}
        size="small"
        locale={{ emptyText: <Empty description="暂无待复盘的问题" /> }}
        onRow={(r) => ({ onClick: () => openDetail(r.trace_id), style: { cursor: 'pointer' } })}
        columns={[
          { title: '时间', dataIndex: 'created_at', width: 180 },
          { title: '问题', dataIndex: 'query', ellipsis: true },
          { title: '问题类型', dataIndex: 'intent', width: 140 },
          { title: '引用资料', dataIndex: 'sources_n', width: 80 },
          {
            title: '审核结论', dataIndex: 'review_verdict', width: 90,
            render: (v: string) => (v === 'reject' ? <Tag color="magenta">未通过</Tag> : (v || '-')),
          },
          {
            title: '原因', dataIndex: 'reasons', width: 260,
            render: (rs: string[]) => (
              <Space size={4} wrap>
                {rs.map((r) => (
                  <Tag key={r} color={REASON_LABEL[r]?.color}>{REASON_LABEL[r]?.text || r}</Tag>
                ))}
              </Space>
            ),
          },
          {
            title: '状态', dataIndex: 'triage_status', width: 90,
            render: (s: string) => (
              <Tag color={STATUS_LABEL[s]?.color}>{STATUS_LABEL[s]?.text || s}</Tag>
            ),
          },
        ]}
      />

      <Drawer
        title={detail?.query}
        width={720}
        open={open}
        onClose={() => setOpen(false)}
        destroyOnClose
      >
        {detail && (
          <>
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 12 }}
              message="加入回归用例后「引用资料」为空，需要人工补齐"
              description="这类问题往往是当初就找错了资料，因此不会把当时的来源自动填进去——否则等于把错误答案定格成标准答案，以后反而测不出来。"
            />
            <Descriptions column={2} size="small" bordered>
              <Descriptions.Item label="记录编号">{detail.trace_id}</Descriptions.Item>
              <Descriptions.Item label="时间">{detail.created_at}</Descriptions.Item>
              <Descriptions.Item label="问题类型">{detail.intent || '-'}</Descriptions.Item>
              <Descriptions.Item label="把握程度">{detail.confidence ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="补全后问题">{detail.rewritten || '-'}</Descriptions.Item>
              <Descriptions.Item label="审核结论">
                {detail.review_verdict || '-'} / {detail.review_score ?? '-'}
              </Descriptions.Item>
              <Descriptions.Item label="引用资料" span={2}>
                {(detail.sources || []).join(', ') || '（空）'}
              </Descriptions.Item>
              <Descriptions.Item label="回答" span={2}>
                <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                  {detail.answer || '（空）'}
                </Typography.Paragraph>
              </Descriptions.Item>
              {detail.feedbacks?.length > 0 && (
                <Descriptions.Item label="用户反馈" span={2}>
                  {detail.feedbacks.map((f: any, i: number) => (
                    <div key={i}>{f.rating > 0 ? '👍' : '👎'} {f.comment || '（无评论）'}</div>
                  ))}
                </Descriptions.Item>
              )}
            </Descriptions>

            <Input.TextArea
              rows={3}
              style={{ marginTop: 12 }}
              placeholder="复盘备注"
              value={note}
              onChange={(e) => setNote(e.target.value)}
            />
            <Space style={{ marginTop: 12 }} wrap>
              <Button type="primary" onClick={promote}>加入回归用例</Button>
              <Button onClick={() => setTriage('confirmed')}>标为已确认</Button>
              <Button onClick={() => setTriage('fixed')}>标为已修复</Button>
              <Button onClick={() => setTriage('ignored')}>忽略</Button>
            </Space>
          </>
        )}
      </Drawer>
    </div>
  );
}
