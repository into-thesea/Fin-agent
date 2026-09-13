/**
 * 策略配置 —— 切分与检索参数
 *
 * 两个 Tab 的"保存后什么时候生效"完全不同, 页面上必须说清楚:
 *   检索参数 → 下一个请求即生效, 免重启
 *   切分参数 → 只影响之后新上传的 PDF; 重建索引也不会回溯已有内容
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Card, Tabs, Form, InputNumber, Switch, Button, Space, Alert, Tag,
  Typography, message, Popconfirm, Input, Divider,
} from 'antd';
import { api } from '../../services/api';
import type { KbStrategy } from '../../types/api';

const { Text } = Typography;

/**
 * 把后端的错误说清楚 —— detail 有两种形态, 都要透出去:
 *   字符串  → 整句原因(如 503「ETL Worker 未运行…」), 不能换成 axios 的 "status code 503"
 *   对象    → 字段级 {field, value, rule}(400 校验失败), 拼成「哪一项：为什么」
 */
const detailText = (err: any): string => {
  const d = err?.response?.data?.detail;
  if (typeof d === 'string' && d) return d;
  if (d && typeof d === 'object' && d.field) return `${d.field}：${d.rule}`;
  return err?.message || '未知错误';
};

export const StrategyPage: React.FC = () => {
  const [cfg, setCfg] = useState<KbStrategy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  // 重建的当前状态: 提交后一直显示, 失败原因不会随 toast 消失
  const [rebuildState, setRebuildState] = useState<{ text: string; failed: boolean } | null>(null);
  const [chunkForm] = Form.useForm();
  const [retrForm] = Form.useForm();
  const timerRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);

  const load = useCallback(async () => {
    try {
      const res = await api.getKbStrategy();
      setCfg(res);
      chunkForm.setFieldsValue(res.chunking.current);
      retrForm.setFieldsValue(res.retrieval.current);
      setError(null);
    } catch (err: any) {
      setError(`策略加载失败：${detailText(err)}`);
    }
  }, [chunkForm, retrForm]);

  useEffect(() => { load(); }, [load]);

  // 离开页面就停止轮询, 否则定时器会一直打接口
  useEffect(() => () => {
    stoppedRef.current = true;
    if (timerRef.current !== null) clearInterval(timerRef.current);
  }, []);

  const save = async (section: 'chunking' | 'retrieval', form: any) => {
    setSaving(true);
    try {
      const values = await form.validateFields();
      const res = await api.putKbStrategy(section, values);
      if (!res.changed.length) {
        message.success('已保存，当前值与生效值一致');
      } else {
        message.success(
          res.applies_to === 'next_request'
            ? `已保存，${res.changed.length} 项变更立即生效`
            : `已保存，将于之后新上传的文档生效`,
        );
      }
      await load();
    } catch (err: any) {
      // 后端返回的是字段级 detail, 直接展示"哪一项、为什么"
      const d = err?.response?.data?.detail;
      message.error(d && typeof d === 'object' && d.field
        ? `${d.field}：${d.rule}`
        : `保存失败：${detailText(err)}`);
    }
    setSaving(false);
  };

  // 只把默认值填进表单, 不直接提交 —— 用户还能先改一处再一起保存
  const restoreDefaults = (section: 'chunking' | 'retrieval') => {
    const form = section === 'chunking' ? chunkForm : retrForm;
    form.setFieldsValue(cfg![section].default);
    message.info('已填入默认值，点「保存」后生效');
  };

  const doRebuild = async () => {
    setRebuilding(true);
    setRebuildState({ text: '已提交，正在重建…', failed: false });
    stoppedRef.current = false;

    const stop = (text: string, failed: boolean) => {
      stoppedRef.current = true;
      if (timerRef.current !== null) clearInterval(timerRef.current);
      timerRef.current = null;
      setRebuilding(false);
      setRebuildState({ text, failed });
    };

    try {
      const { task_id } = await api.rebuildKb();
      // 轮询复用任务状态接口, 不新增进度通道
      let fails = 0;
      timerRef.current = window.setInterval(async () => {
        if (stoppedRef.current) return;   // 已收尾/已卸载: 迟到的响应不再改状态
        try {
          const t = await api.getTaskStatus(task_id);
          fails = 0;
          if (t.status === 'completed') {
            stop('索引重建完成', false);
            message.success('索引重建完成');
          } else if (t.status === 'failed') {
            const why = t.error || '未知原因';
            stop(`重建失败：${why}`, true);
            message.error(`重建失败：${why}`);
          } else {
            setRebuildState({
              text: `重建中 ${t.progress ?? 0}%${t.stage ? ` · ${t.stage}` : ''}`,
              failed: false,
            });
          }
        } catch (err: any) {
          // 单次轮询失败不终止(服务重启/网络抖动), 但连续失败要收尾 ——
          // 否则按钮会永远转圈, 用户既不知道成没成, 也不知道该不该再点一次
          const why = detailText(err);
          if (++fails >= 5) {
            stop(`进度查询连续失败（${why}），已停止等待，请稍后重新查看`, true);
          }
        }
      }, 3000);
    } catch (err: any) {
      // 503 = 后台处理服务未运行, 后端已给出可读原因
      const why = detailText(err);
      stop(`提交失败：${why}`, true);
      message.error(`提交失败：${why}`);
    }
  };

  const label = (section: 'chunking' | 'retrieval', field: string, text: string) => (
    <Space size={6}>
      <span>{text}</span>
      {cfg?.[section].changed.includes(field) && <Tag color="orange">已改动</Tag>}
    </Space>
  );

  if (error) {
    return <Alert type="error" showIcon message={error} action={<Button size="small" onClick={load}>重试</Button>} />;
  }

  const chunking = (
    <Form form={chunkForm} layout="vertical" style={{ maxWidth: 520 }}>
      <Alert
        type="info" showIcon style={{ marginBottom: 16 }}
        message="切分参数只影响之后新上传的文档"
        description="已经入库的内容不会被重新切分 —— 这些参数在解析上传文档时使用，重建索引也不会追溯。要让某份文档按新参数重切，请删除后重新上传。"
      />
      <Form.Item name="chunk_size" label={label('chunking', 'chunk_size', '分块目标长度（字符）')}>
        <InputNumber min={150} max={800} style={{ width: '100%' }} />
      </Form.Item>
      <Form.Item
        name="overlap"
        label={label('chunking', 'overlap', '相邻分块重叠（字符）')}
        extra="必须小于分块目标长度"
      >
        <InputNumber min={0} max={400} style={{ width: '100%' }} />
      </Form.Item>
      <Form.Item
        name="max_chunk_content"
        label={label('chunking', 'max_chunk_content', '单块纯内容上限')}
        extra="不能大于分块目标长度"
      >
        <InputNumber min={100} max={1000} style={{ width: '100%' }} />
      </Form.Item>
      <Space>
        <Button type="primary" loading={saving} onClick={() => save('chunking', chunkForm)}>保存</Button>
        <Popconfirm title="填入代码默认值？" onConfirm={() => restoreDefaults('chunking')}>
          <Button>恢复默认</Button>
        </Popconfirm>
      </Space>
    </Form>
  );

  const retrieval = (
    <Form form={retrForm} layout="vertical" style={{ maxWidth: 520 }}>
      <Alert
        type="success" showIcon style={{ marginBottom: 16 }}
        message="检索参数保存后立即生效"
        description="无需重启服务，也无需重建索引 —— 下一次提问就会用上新参数。"
      />
      <Form.Item
        name="top_k"
        label={label('retrieval', 'top_k', '返回条数')}
        extra="影响问答检索的结果条数；智能助手在对话中主动查资料时固定取 5 条，不随这里变化"
      >
        <InputNumber min={1} max={20} style={{ width: '100%' }} />
      </Form.Item>
      <Form.Item name="mmr_enabled" label={label('retrieval', 'mmr_enabled', '结果多样性去重（MMR）')} valuePropName="checked">
        <Switch />
      </Form.Item>
      <Form.Item name="mmr_lambda" label={label('retrieval', 'mmr_lambda', 'MMR 相关性/多样性权重')}>
        <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
      </Form.Item>
      <Form.Item name="rrf_k" label={label('retrieval', 'rrf_k', '融合平滑常数（RRF k）')}>
        <InputNumber min={1} max={100} style={{ width: '100%' }} />
      </Form.Item>
      <Form.Item name="rerank_enabled" label={label('retrieval', 'rerank_enabled', '启用语义重排')} valuePropName="checked">
        <Switch />
      </Form.Item>
      <Form.Item
        name="rerank_pool"
        label={label('retrieval', 'rerank_pool', '重排候选池大小')}
        extra="必须大于返回条数，否则重排不生效"
      >
        <InputNumber min={5} max={100} style={{ width: '100%' }} />
      </Form.Item>
      {/* 重排模型当前只有一种可选, 做成只读展示; 要换模型得后端先支持 */}
      <Form.Item
        name="rerank_model"
        label={label('retrieval', 'rerank_model', '重排模型')}
        extra="当前仅此一种，不可修改"
      >
        <Input disabled style={{ width: '100%' }} />
      </Form.Item>
      <Space>
        <Button type="primary" loading={saving} onClick={() => save('retrieval', retrForm)}>保存</Button>
        <Popconfirm title="填入代码默认值？" onConfirm={() => restoreDefaults('retrieval')}>
          <Button>恢复默认</Button>
        </Popconfirm>
      </Space>
    </Form>
  );

  return (
    <div>
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space split={<Divider type="vertical" />}>
          {/* source 只说"不是从配置文件读的", 分不清"还没保存过"(正常初始态) 与"文件损坏",
              故不写"不可读" —— 那会把正常状态说成故障 */}
          <Text type="secondary">
            配置来源：{cfg?.source === 'file' ? '配置文件' : '代码默认值'}
          </Text>
          {cfg?.updated_at && (
            <Text type="secondary">最后修改：{cfg.updated_at} · {cfg.updated_by}</Text>
          )}
        </Space>
      </Card>

      <Card size="small" style={{ marginBottom: 16 }}>
        {/* forceRender: 两个表单都要在未激活时就能被 load() 填值,
            否则切到第二个 Tab 前 setFieldsValue 打在未挂载的表单上(开发期会警告) */}
        <Tabs items={[
          { key: 'chunking', label: '切分策略', children: chunking, forceRender: true },
          { key: 'retrieval', label: '检索策略', children: retrieval, forceRender: true },
        ]} />
      </Card>

      <Card title="🔧 索引维护" size="small">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            从内置知识库的源文件全量重建索引（检索内容 / 向量库 / 关键词索引 / 知识图谱）。
            这是维护操作，与切分参数无关 —— 不会把新的切分参数应用到已有内容。
            用于怀疑索引与资料本身对不上时。
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            重建的是<Text strong>内置知识库</Text>；已经上传的文档不受影响 —— 它们会被原样保留、不会丢失，
            但也不会按新的切分参数重新切一遍。要让某份上传文档按新参数重切，请删除后重新上传。
          </Text>
          <Space>
            <Popconfirm title="确定全量重建？期间检索结果可能短暂不完整" onConfirm={doRebuild}>
              <Button danger loading={rebuilding}>立即重建索引</Button>
            </Popconfirm>
            {rebuildState && (
              <Text type={rebuildState.failed ? 'danger' : 'secondary'} style={{ fontSize: 12 }}>
                {rebuildState.text}
              </Text>
            )}
          </Space>
        </Space>
      </Card>
    </div>
  );
};
