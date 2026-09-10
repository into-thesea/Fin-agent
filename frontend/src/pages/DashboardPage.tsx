import React, { useEffect, useState, useRef } from 'react';
import { Card, Row, Col, Statistic, Typography, Spin, Table, Tag, Alert, Button } from 'antd';
import { FolderOpenOutlined, ApiOutlined, DatabaseOutlined, ThunderboltOutlined, ReloadOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { api } from '../services/api';
import type { Document, Metrics } from '../types/api';

const { Title } = Typography;

const LOADING_TIMEOUT_MS = 15000; // 15s 加载超时

export const DashboardPage: React.FC = () => {
  const [docs, setDocs] = useState<Document[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const loadData = () => {
    setLoading(true);
    setError(null);

    // 加载超时保护
    const timeoutId = setTimeout(() => {
      if (mountedRef.current) {
        setLoading(false);
        setError('数据加载超时，请检查后端服务是否正常运行');
      }
    }, LOADING_TIMEOUT_MS);

    Promise.all([
      api.getDocuments(),
      api.getMetrics(),
    ]).then(([docsRes, metricsRes]) => {
      if (!mountedRef.current) return;
      setDocs(docsRes.documents);
      setMetrics(metricsRes);
      setError(null);
    }).catch((err: any) => {
      if (!mountedRef.current) return;
      const msg = err?.response?.data?.detail || err?.message || '数据加载失败';
      setError(`加载失败: ${msg}`);
    }).finally(() => {
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
        <Alert
          type="error"
          message="数据加载异常"
          description={error}
          showIcon
          style={{ maxWidth: 500, margin: '0 auto 24px' }}
        />
        <Button icon={<ReloadOutlined />} onClick={loadData}>重新加载</Button>
      </div>
    );
  }

  // 知识库增长趋势图
  const trendData = docs
    .filter((d) => d.upload_time)
    .sort((a, b) => new Date(a.upload_time).getTime() - new Date(b.upload_time).getTime());

  const trendOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 40, right: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      data: trendData.map((d) => d.upload_time?.slice(5, 10) || ''),
      axisLabel: { rotate: 45 },
    },
    yAxis: { type: 'value', name: '累积分块数' },
    series: [{
      name: '知识库增长',
      type: 'line',
      smooth: true,
      areaStyle: { opacity: 0.3 },
      data: trendData.reduce((acc: number[], d) => {
        const last = acc.length > 0 ? acc[acc.length - 1] : 0;
        acc.push(last + (d.chunks_count || 0));
        return acc;
      }, []),
      markPoint: { data: [{ type: 'max', name: '最大值' }] },
    }],
  };

  // 缓存命中率仪表盘
  const gaugeOption = {
    series: [{
      type: 'gauge',
      startAngle: 200,
      endAngle: -20,
      min: 0,
      max: 100,
      detail: { formatter: '{value}%', fontSize: 20 },
      data: [{ value: metrics?.cache_hit_ratio || 0, name: '缓存命中率' }],
    }],
  };

  // 文档类型分布 (饼图)
  const docTypes = docs.reduce((acc: Record<string, number>, d) => {
    const ext = d.filename?.split('.').pop() || 'unknown';
    acc[ext] = (acc[ext] || 0) + 1;
    return acc;
  }, {});

  const pieOption = {
    tooltip: { trigger: 'item' },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      data: Object.entries(docTypes).map(([name, value]) => ({ name, value })),
      label: { show: true, formatter: '{b}: {c}' },
    }],
  };

  const columns = [
    { title: '文件名', dataIndex: 'filename', key: 'filename', ellipsis: true, width: 200 },
    { title: '版本', dataIndex: 'version', key: 'version', width: 60 },
    { title: '分块数', dataIndex: 'chunks_count', key: 'chunks_count', width: 80 },
    {
      title: '入库时间', dataIndex: 'upload_time', key: 'upload_time', width: 180,
    },
    {
      title: '哈希', dataIndex: 'content_hash', key: 'content_hash', width: 100,
      render: (h: string) => h ? <Tag>{h.slice(0, 8)}</Tag> : '-',
    },
  ];

  return (
    <div>
      {/* 指标卡片：大屏 4 列，中屏 2 列，小屏 1 列 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="文档总数" value={metrics?.documents_count || 0} prefix={<FolderOpenOutlined />} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="分块总数" value={metrics?.chunks_count || 0} prefix={<ApiOutlined />} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="查询次数" value={metrics?.queries_count || 0} prefix={<DatabaseOutlined />} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="缓存命中率" value={metrics?.cache_hit_ratio || 0} suffix="%" prefix={<ThunderboltOutlined />} /></Card>
        </Col>
      </Row>

      {/* 图表区域：大屏 6:3:3，中屏堆叠 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="📦 知识库增长趋势" size="small">
            {docs.length === 0 ? (
              <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>暂无数据</div>
            ) : (
              <ReactECharts option={trendOption} style={{ height: 300 }} />
            )}
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card title="🎯 缓存命中率" size="small">
            <ReactECharts option={gaugeOption} style={{ height: 300 }} />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card title="📊 文档类型分布" size="small">
            {Object.keys(docTypes).length === 0 ? (
              <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>暂无数据</div>
            ) : (
              <ReactECharts option={pieOption} style={{ height: 300 }} />
            )}
          </Card>
        </Col>
      </Row>

      <Card title="📄 文档列表" size="small">
        <Table
          dataSource={docs}
          columns={columns}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 5, size: 'small' }}
          locale={{ emptyText: '暂无文档' }}
        />
      </Card>
    </div>
  );
};
