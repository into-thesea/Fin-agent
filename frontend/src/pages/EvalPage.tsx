/**
 * 评测看板 —— 每次知识库同步后自动评测的结果
 *
 * 展示的是**离线可复现**的三个指标 (证据覆盖 / 路径一致 / 图谱触发):
 * 不消耗大模型调用、同样的知识库跑出来结果相同, 因此适合做版本趋势。
 * 答案质量类评测(需要大模型打分)目前未纳入, 页面会如实标出而不是假装有数据。
 *
 * 界面文案面向业务人员, 不出现技术术语。
 */
import { useEffect, useState, useCallback } from 'react';
import {
  Card, Row, Col, Statistic, Table, Tag, Typography, Space, Alert,
  Button, Empty, Spin, Progress, Tooltip,
} from 'antd';
import { ReloadOutlined, FileSearchOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { api } from '../services/api';
import { PALETTE, CHART_COLORS } from '../styles/theme';
import type { EvalSummary, EvalRow } from '../types/api';

const { Title, Text } = Typography;

// 用户问题的大类 (与后端意图标签对应, 展示成业务说法)
const INTENT_LABEL: Record<string, string> = {
  product_consult: '产品咨询',
  deposit_insurance: '存款保险',
  risk_suitability: '风险与适当性',
  income_question: '收益与计息',
  hold_redeem: '持有与赎回',
  buy_process: '购买流程',
  fee_rule: '费率规则',
  fraud_report: '诈骗举报',
  product_compare: '产品对比',
  chitchat: '闲聊',
};

// 指标低于此值就标黄 —— 提醒需要看具体是哪几条没覆盖
const WARN_THRESHOLD = 0.95;

export const EvalPage: React.FC = () => {
  const [data, setData] = useState<EvalSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.getEvalSummary()
      .then((res) => setData(res))
      .catch((err: any) => {
        const msg = err?.response?.data?.detail || err?.message || '未知错误';
        setError(`评测数据加载失败：${msg}`);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;
  }

  if (error) {
    return (
      <div style={{ marginTop: 48, textAlign: 'center' }}>
        <Alert
          type="error"
          message="评测数据加载异常"
          description={error}
          showIcon
          style={{ maxWidth: 520, margin: '0 auto 24px' }}
        />
        <Button icon={<ReloadOutlined />} onClick={load}>重新加载</Button>
      </div>
    );
  }

  if (!data || !data.available.report || !data.current) {
    return (
      <Card>
        <Empty
          image={<FileSearchOutlined style={{ fontSize: 56, color: '#bfbfbf' }} />}
          description={
            <Space direction="vertical" size={4}>
              <Text strong>还没有评测结果</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                知识库每次同步后会自动跑一次评测；也可以手动执行
                <Text code>scripts/sync_kb.py</Text> 生成。
              </Text>
            </Space>
          }
        >
          <Button icon={<ReloadOutlined />} onClick={load}>刷新</Button>
        </Empty>
      </Card>
    );
  }

  const cur = data.current;
  const metrics = cur.metrics;

  const metricValue = (key: string) => {
    const v = metrics[key];
    return typeof v === 'number' ? Math.round(v * 1000) / 10 : null;
  };

  // 版本趋势: 同一版本只保留最后一次快照 (后端已去重)
  const history = data.history;
  const trendOption = {
    color: [...CHART_COLORS],
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v: number) => (v == null ? '-' : `${v}%`),
    },
    legend: { data: ['证据覆盖', '路径一致', '图谱触发'], bottom: 0 },
    grid: { left: 48, right: 24, top: 24, bottom: 48 },
    xAxis: {
      type: 'category',
      data: history.map((h) => h.at?.slice(5, 16).replace('T', ' ') || h.version.slice(0, 6)),
      axisLabel: { rotate: 30, fontSize: 11 },
    },
    yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
    series: [
      {
        name: '证据覆盖', type: 'line', smooth: true, symbolSize: 6,
        data: history.map((h) => (h.evidence_coverage == null ? null : +(h.evidence_coverage * 100).toFixed(1))),
      },
      {
        name: '路径一致', type: 'line', smooth: true, symbolSize: 6,
        data: history.map((h) => (h.path_structure_consistency == null ? null : +(h.path_structure_consistency * 100).toFixed(1))),
      },
      {
        name: '图谱触发', type: 'line', smooth: true, symbolSize: 6,
        data: history.map((h) => (h.graph_trigger_ratio == null ? null : +(h.graph_trigger_ratio * 100).toFixed(1))),
      },
    ],
  };

  const columns = [
    { title: '编号', dataIndex: 'id', key: 'id', width: 70 },
    {
      title: '评测问题', dataIndex: 'question', key: 'question', ellipsis: true,
    },
    {
      title: '问题类型', dataIndex: 'intent', key: 'intent', width: 120,
      render: (v: string) => <Tag>{INTENT_LABEL[v] || v}</Tag>,
      filters: Object.entries(INTENT_LABEL).map(([value, text]) => ({ value, text })),
      onFilter: (value: any, record: EvalRow) => record.intent === value,
    },
    {
      title: '资料覆盖', dataIndex: 'evidence_covered', key: 'evidence_covered', width: 100,
      render: (ok: boolean) => ok
        ? <Tag color="success">已覆盖</Tag>
        : <Tag color="error">未覆盖</Tag>,
      filters: [{ value: true, text: '已覆盖' }, { value: false, text: '未覆盖' }],
      onFilter: (value: any, record: EvalRow) => record.evidence_covered === value,
    },
    {
      title: '推理路径', dataIndex: 'path_consistent', key: 'path_consistent', width: 100,
      render: (v: boolean | null) => v == null
        ? <Text type="secondary">不适用</Text>
        : v ? <Tag color="success">一致</Tag> : <Tag color="error">不一致</Tag>,
    },
    {
      title: '图谱命中', dataIndex: 'graph_hit', key: 'graph_hit', width: 100,
      render: (ok: boolean) => ok ? <Tag color="blue">命中</Tag> : <Text type="secondary">未命中</Text>,
    },
  ];

  const uncovered = cur.rows.filter((r) => !r.evidence_covered).length;

  return (
    <div>
      <Space style={{ marginBottom: 16 }} align="center">
        <Title level={4} style={{ margin: 0 }}>评测看板</Title>
        <Text type="secondary" style={{ fontSize: 12 }}>
          知识库版本 <Text code>{cur.version}</Text> · 共 {cur.golden_n} 条评测问题
        </Text>
        <Button size="small" icon={<ReloadOutlined />} onClick={load}>刷新</Button>
      </Space>

      {!data.available.online && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="下方为不需要大模型打分的离线指标"
          description="答案质量类评测（忠实度、相关性）需要调用大模型，当前未纳入本看板。"
        />
      )}

      {uncovered > 0 && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message={`有 ${uncovered} 条评测问题的答案资料未被检索覆盖`}
          description="在下方表格按「未覆盖」筛选，确认是知识库缺内容还是检索没找到。"
        />
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        {Object.entries(data.metric_labels).map(([key, label]) => {
          const v = metricValue(key);
          return (
            <Col xs={24} sm={8} key={key}>
              <Card>
                <Statistic
                  title={label}
                  value={v ?? 0}
                  suffix="%"
                  precision={1}
                  valueStyle={v != null && v < WARN_THRESHOLD * 100 ? { color: '#faad14' } : undefined}
                />
                <Progress
                  percent={v ?? 0}
                  showInfo={false}
                  size="small"
                  strokeColor={v != null && v < WARN_THRESHOLD * 100 ? '#faad14' : '#52c41a'}
                />
              </Card>
            </Col>
          );
        })}
      </Row>

      <Card title="📈 版本趋势" size="small" style={{ marginBottom: 16 }}>
        {history.length === 0 ? (
          <Empty description="暂无历史快照" />
        ) : (
          <>
            <ReactECharts option={trendOption} style={{ height: 300 }} />
            <Text type="secondary" style={{ fontSize: 12 }}>
              每知识库版本一个点（同一版本重复评测只保留最后一次）。
              合并前的原始记录共 {data.raw_history_count} 条。
            </Text>
          </>
        )}
      </Card>

      <Card
        title={`📋 逐条结果 (${cur.rows.length})`}
        size="small"
        extra={
          <Tooltip title="评测问题写在 data/eval/finance_qa_golden.jsonl，可人工增补">
            <Text type="secondary" style={{ fontSize: 12 }}>评测集 · {cur.golden_n} 条</Text>
          </Tooltip>
        }
      >
        <Table
          dataSource={cur.rows}
          columns={columns}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 15, showSizeChanger: false }}
          locale={{ emptyText: '暂无明细' }}
        />
      </Card>
    </div>
  );
};
