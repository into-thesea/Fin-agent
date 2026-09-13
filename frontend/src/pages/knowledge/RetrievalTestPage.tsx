/**
 * 检索调试台 —— 输入一句话, 看它被哪条通路召回、排第几
 *
 * 「只看单通路」开关是核心: 两条通路都召回的条目通常没问题,
 * 靠单条通路捞回来的才是排查重点(另一条为什么没找到)。
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  Card, Input, Button, Space, Table, Tag, Alert, Typography, Switch, Descriptions, Empty,
} from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { api } from '../../services/api';
import { CHART_COLORS } from '../../styles/theme';
import type { RetrievalTestResult } from '../../types/api';

const { Text } = Typography;

const PATH_LABEL: Record<string, string> = { dense: '向量', sparse: '关键词' };

/** 参数名 → 业务叫法 (与「策略配置」页同一套措辞, 不把配置键直接摆给使用者看) */
const CONFIG_LABEL: Record<string, string> = {
  mmr_enabled: '结果去重',
  mmr_lambda: '去重权重',
  rrf_k: '融合平滑常数',
  rerank_enabled: '语义重排',
  rerank_model: '重排模型',
  rerank_pool: '重排候选数',
};

const fmtConfig = (v: any) => (typeof v === 'boolean' ? (v ? '开启' : '关闭') : String(v));

export const RetrievalTestPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [onlySingle, setOnlySingle] = useState(false);
  const [result, setResult] = useState<RetrievalTestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 返回条数跟随「策略配置」里的设置; 读不到就用默认值并在页面上说明(不静默)
  const [topK, setTopK] = useState(5);
  const [topKNote, setTopKNote] = useState<string | null>(null);
  // 开关一改就重查, 连点两下会连发两个请求; 先发的后到就会拿旧结果盖掉新结果 —— 只认最后一次
  const seqRef = useRef(0);

  useEffect(() => {
    api.getKbStrategy()
      .then((s) => {
        const v = s?.retrieval?.current?.top_k;
        if (typeof v === 'number' && v >= 1 && v <= 20) {
          setTopK(v);
          setTopKNote(null);
        } else {
          setTopKNote('策略配置里的返回条数不可用，本次按默认 5 条检索');
        }
      })
      .catch(() => setTopKNote('暂时读不到策略配置，本次按默认 5 条检索'));
  }, []);

  const run = async () => {
    if (!query.trim()) return;
    const seq = ++seqRef.current;
    setLoading(true);
    try {
      const res = await api.retrievalTest({ query: query.trim(), top_k: topK, only_single_path: onlySingle });
      if (seq !== seqRef.current) return;  // 过期响应: 丢弃, loading 交给新请求收尾
      setResult(res);
      setError(null);
    } catch (err: any) {
      if (seq !== seqRef.current) return;
      // 不静默: 失败时表格空着会被读成"知识库里没有相关内容"
      setError(`检索失败：${err?.response?.data?.detail || err?.message || '未知错误'}`);
    }
    setLoading(false);
  };

  // 开关是查询条件的一部分, 切了就该立刻看到过滤后的结果, 不该让人再点一次「调试」。
  // 首屏 query 为空, run() 自己会跳过, 不会白发请求。
  useEffect(() => { run(); }, [onlySingle]);

  const columns = [
    { title: '#', dataIndex: 'rank', key: 'rank', width: 50 },
    {
      title: '来源', dataIndex: 'source', key: 'source', width: 170, ellipsis: true,
      render: (v: string) => <Tag>{v}</Tag>,
    },
    { title: '小节', dataIndex: 'section', key: 'section', width: 150, ellipsis: true },
    {
      title: '召回通路', dataIndex: 'found_by', key: 'found_by', width: 160,
      render: (v: string[]) => (
        <Space size={4}>
          {v.map((p, i) => (
            <Tag key={p} color={i === 0 ? CHART_COLORS[0] : CHART_COLORS[1]}>
              {PATH_LABEL[p] || p}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '向量排名', dataIndex: 'dense_rank', key: 'dense_rank', width: 90,
      render: (v: number | null) => v == null ? <Text type="secondary">未召回</Text> : `第 ${v}`,
    },
    {
      title: '关键词排名', dataIndex: 'sparse_rank', key: 'sparse_rank', width: 100,
      render: (v: number | null) => v == null ? <Text type="secondary">未召回</Text> : `第 ${v}`,
    },
    {
      title: '融合分', dataIndex: 'rrf_score', key: 'rrf_score', width: 100,
      render: (v: number | null) => v == null ? '-' : v.toFixed(5),
    },
  ];

  return (
    <div>
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space style={{ width: '100%' }}>
          <Input
            placeholder="输入一句客户会问的话，例如：稳盈添利30天风险等级是多少"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onPressEnter={run}
            prefix={<SearchOutlined />}
            style={{ width: 520 }}
            allowClear
          />
          <Button type="primary" loading={loading} onClick={run}>调试</Button>
          <Space size={6}>
            <Switch size="small" checked={onlySingle} onChange={setOnlySingle} />
            <Text type="secondary" style={{ fontSize: 12 }}>只看单通路召回</Text>
          </Space>
        </Space>
      </Card>

      {error && (
        <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }}
               action={<Button size="small" onClick={run}>重试</Button>} />
      )}

      {topKNote && (
        <Alert type="warning" showIcon message={topKNote} style={{ marginBottom: 16 }} />
      )}

      {!result && !error && (
        <Card><Empty description="输入一个问题开始调试" /></Card>
      )}

      {result && (
        <>
          {result.sparse_empty && (
            <Alert
              type="warning" showIcon style={{ marginBottom: 16 }}
              message="关键词检索本次未返回任何结果"
              description="上面的召回通路只反映了向量这一路 —— 但这不代表「关键词没命中某一条」，而是整条关键词通路都是空的。这通常意味着关键词索引异常，可到「策略配置 · 索引维护」重建后再试。"
            />
          )}

          {!!result.config_used?.rerank_enabled && (
            <Alert
              type="info" showIcon style={{ marginBottom: 16 }}
              message="已启用语义重排，此处展示的是重排之前的排序"
              description="当前展示的是融合后的排序；已启用语义精排，线上最终顺序会在此基础上再调整。"
            />
          )}

          <Card size="small" title="本次生效的检索参数" style={{ marginBottom: 16 }}>
            <Descriptions size="small" column={4}>
              {Object.entries(result.config_used)
                .filter(([k]) => k !== 'top_k')
                .map(([k, v]) => (
                  <Descriptions.Item key={k} label={CONFIG_LABEL[k] || k}>
                    {fmtConfig(v)}
                  </Descriptions.Item>
                ))}
              {/* 显示的是**实际请求**的条数: 后端会把 config_used.top_k 覆盖成请求值 */}
              <Descriptions.Item label="返回条数">{result.config_used.top_k ?? topK}</Descriptions.Item>
              <Descriptions.Item label="耗时">{result.elapsed_ms} ms</Descriptions.Item>
            </Descriptions>
          </Card>

          {result.graph?.error && (
            <Alert type="warning" showIcon style={{ marginBottom: 16 }}
                   message="图谱检索未执行成功" description={result.graph.error} />
          )}

          <Card
            title={`检索结果（${result.fused.length}${onlySingle ? ` / 单通路 ${result.single_path_count}` : ''}）`}
            size="small"
          >
            <Table
              dataSource={result.fused}
              columns={columns as any}
              rowKey="chunk_id"
              size="small"
              pagination={false}
              locale={{ emptyText: '没有召回任何分块' }}
            />
            {onlySingle && result.fused.length === 0 && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                没有单通路召回的条目 —— 说明每条结果都被两条通路同时找到了。
              </Text>
            )}
          </Card>
        </>
      )}
    </div>
  );
};
