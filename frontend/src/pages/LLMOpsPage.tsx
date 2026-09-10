import React, { useEffect, useState, useRef } from 'react';
import { Card, Row, Col, Statistic, Table, Typography, Spin, Tag, Alert, Button, Empty } from 'antd';
import {
  ClockCircleOutlined, ThunderboltOutlined, BugOutlined,
  PhoneOutlined, PercentageOutlined, ReloadOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { api } from '../services/api';
import type { AuditLog, Metrics, SLI } from '../types/api';

const { Title } = Typography;

const LOADING_TIMEOUT_MS = 15000;

// 理财客服意图中文标签
const INTENT_LABELS: Record<string, string> = {
  product_consult: '产品咨询', product_compare: '产品对比', risk_suitability: '适当性',
  income_question: '收益口径', deposit_insurance: '存款保险', buy_process: '购买流程',
  hold_redeem: '持有/赎回', fee_rule: '费用', fraud_report: '风险举报',
  complaint: '投诉', chitchat: '闲聊', unknown: '未知',
  greeting: '问候', simple_fact: '简单FAQ', complex: '复杂',
};

export const LLMOpsPage: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [sli, setSli] = useState<SLI | null>(null); // P9: SLI 指标
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const loadData = () => {
    setLoading(true);
    setError(null);

    const timeoutId = setTimeout(() => {
      if (mountedRef.current) {
        setLoading(false);
        setError('数据加载超时，请检查后端服务');
      }
    }, LOADING_TIMEOUT_MS);

    Promise.all([api.getMetrics(), api.getAuditLogs(50), api.getSLI('1h')])
      .then(([metricsRes, logsRes, sliRes]) => {
        if (!mountedRef.current) return;
        setMetrics(metricsRes);
        setLogs(logsRes.logs || []);
        setSli(sliRes); // P9
        setError(null);
      })
      .catch((err: any) => {
        if (!mountedRef.current) return;
        setError(err?.response?.data?.detail || err?.message || '指标加载失败');
      })
      .finally(() => {
        clearTimeout(timeoutId);
        if (mountedRef.current) setLoading(false);
      });
  };

  useEffect(() => {
    mountedRef.current = true;
    loadData();
    return () => { mountedRef.current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;

  if (error) {
    return (
      <div style={{ marginTop: 48, textAlign: 'center' }}>
        <Alert type="warning" message="指标数据暂不可用" description={error} showIcon
          style={{ maxWidth: 500, margin: '0 auto 24px' }} />
        <Button icon={<ReloadOutlined />} onClick={loadData}>重试</Button>
      </div>
    );
  }

  // 真实平均延迟 (来自审计日志)
  const avgLatency = logs.length
    ? (logs.reduce((s, l) => s + (l.latency || 0), 0) / logs.length).toFixed(1)
    : '0.0';

  // 响应延迟趋势 (最近日志)
  const recentLogs = [...logs].sort((a, b) => a.id - b.id).slice(-20);
  const latencyOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      data: recentLogs.map((l) => l.timestamp?.slice(11, 16) || ''),
      axisLabel: { rotate: 45 },
    },
    yAxis: { type: 'value', name: '延迟 (秒)' },
    series: [{
      name: '响应延迟',
      type: 'bar',
      data: recentLogs.map((l) => l.latency || 0),
      itemStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: '#1677ff' },
            { offset: 1, color: '#91caff' },
          ],
        },
      },
    }],
  };

  // 客服意图分布
  const intentData = Object.entries(metrics?.intent_stats || {})
    .map(([k, v]) => ({ name: INTENT_LABELS[k] || k, value: v }))
    .filter((d) => d.value > 0);
  const intentOption = {
    tooltip: { trigger: 'item' },
    legend: { bottom: 0 },
    series: [{
      type: 'pie',
      radius: ['40%', '68%'],
      data: intentData,
      label: { show: true, formatter: '{b}: {c}' },
    }],
  };

  const logColumns = [
    { title: '时间', dataIndex: 'timestamp', key: 'timestamp', width: 170 },
    { title: '查询', dataIndex: 'query', key: 'query', ellipsis: true },
    {
      title: '意图', dataIndex: 'intent', key: 'intent', width: 110,
      render: (v: string) => v ? <Tag>{INTENT_LABELS[v] || v}</Tag> : '-',
    },
    {
      title: '延迟', dataIndex: 'latency', key: 'latency', width: 80,
      render: (v: number) => `${(v || 0).toFixed(1)}s`,
      sorter: (a: AuditLog, b: AuditLog) => a.latency - b.latency,
    },
    {
      title: '缓存命中', dataIndex: 'cache_hit', key: 'cache_hit', width: 90,
      render: (v: boolean) => <Tag color={v ? 'green' : 'default'}>{v ? '是' : '否'}</Tag>,
    },
  ];

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={12} lg={6}>
          <Card><Statistic title="缓存命中率" value={metrics?.cache_hit_ratio || 0} suffix="%" prefix={<ThunderboltOutlined />} /></Card>
        </Col>
        <Col xs={12} lg={6}>
          <Card><Statistic title="总查询次数" value={metrics?.queries_count || 0} prefix={<BugOutlined />} /></Card>
        </Col>
        <Col xs={12} lg={6}>
          <Card><Statistic title="平均延迟" value={avgLatency} suffix="s" prefix={<ClockCircleOutlined />} /></Card>
        </Col>
        <Col xs={12} lg={6}>
          <Card>
            <Statistic title="人工转接" value={metrics?.handoff_count || 0} prefix={<PhoneOutlined />}
              suffix={`(${(metrics?.handoff_rate || 0).toFixed(1)}%)`} />
          </Card>
        </Col>
      </Row>

      {/* P9: SLI 实时指标 (来自 /monitor/sli) */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={12} lg={6}>
          <Card><Statistic title="QPS" value={sli?.qps || 0} precision={2} prefix={<ThunderboltOutlined />} /></Card>
        </Col>
        <Col xs={12} lg={6}>
          <Card><Statistic title="成功率" value={(sli?.success_rate || 0) * 100} precision={1} suffix="%" prefix={<PercentageOutlined />} /></Card>
        </Col>
        <Col xs={12} lg={6}>
          <Card><Statistic title="语义缓存命中" value={(sli?.cache_hit_rate || 0) * 100} precision={1} suffix="%" prefix={<ThunderboltOutlined />} /></Card>
        </Col>
        <Col xs={12} lg={6}>
          <Card><Statistic title="P50 / P90" value={`${sli?.p50_ms ?? 0} / ${sli?.p90_ms ?? 0}`} suffix="ms" prefix={<ClockCircleOutlined />} /></Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="🎯 理财意图分布" size="small">
            {intentData.length === 0 ? (
              <Empty description="暂无意图数据" style={{ padding: 40 }} />
            ) : (
              <ReactECharts option={intentOption} style={{ height: 280 }} />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="⏱️ 响应延迟分布 (最近 20 条)" size="small">
            {recentLogs.length === 0 ? (
              <Empty description="暂无日志" style={{ padding: 40 }} />
            ) : (
              <ReactECharts option={latencyOption} style={{ height: 280 }} />
            )}
          </Card>
        </Col>
      </Row>

      <Card title="📝 审计日志" size="small">
        <Table
          dataSource={logs}
          columns={logColumns}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: '暂无日志' }}
        />
      </Card>
    </div>
  );
};
