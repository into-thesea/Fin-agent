import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  Table, Button, Progress, Tag, Card, Row, Col, Statistic,
  message, Typography, Space, List, Badge, notification, Modal,
  Tooltip, Popover, Empty,
} from 'antd';
import {
  UploadOutlined, InboxOutlined, FolderOpenOutlined,
  ApiOutlined, DeleteOutlined, StopOutlined,
  SearchOutlined, ThunderboltOutlined, FileOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import { api } from '../services/api';
import { WSClient } from '../services/websocket';
import type { Document, TaskStatus, BatchUploadResult, SystemStatus } from '../types/api';

const { Text } = Typography;

// ── 文件项状态 ──
type FileItemStatus =
  | 'pending' | 'uploading' | 'accepted' | 'skipped'
  | 'rejected' | 'etl-done' | 'etl-failed' | 'cancelled';

interface FileItem {
  uid: string;
  name: string;
  size: number;
  status: FileItemStatus;
  message?: string;
  taskId?: string;
  etlProgress?: number;
  etlStage?: string;
}

export const KnowledgePage: React.FC = () => {
  const [docs, setDocs] = useState<Document[]>([]);
  const [stats, setStats] = useState({ documents: 0, queries: 0, chunks: 0 });
  const [loading, setLoading] = useState(false);
  const [fileQueue, setFileQueue] = useState<FileItem[]>([]);
  const [batchUploading, setBatchUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const existingAbortRef = useRef<AbortController | null>(null);

  // ── 删除确认 ──
  const [deleteTarget, setDeleteTarget] = useState<Document | null>(null);
  const [deleting, setDeleting] = useState(false);

  // ── 已有文件处理 ──
  const [existingFiles, setExistingFiles] = useState<string[]>([]);
  const [existingModalOpen, setExistingModalOpen] = useState(false);
  const [processingExisting, setProcessingExisting] = useState(false);

  // ── 实体统计 ──

  // ── 系统服务状态 ──
  const [sysStatus, setSysStatus] = useState<SystemStatus | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [docsRes, statsRes] = await Promise.all([
        api.getDocuments(),
        api.getKnowledgeStats(),
      ]);
      setDocs(docsRes.documents);
      setStats(statsRes);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // 轮询系统服务状态 (每 30 秒)
  useEffect(() => {
    const fetchStatus = async () => {
      try { setSysStatus(await api.getSystemStatus()); } catch { /* ignore */ }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const updateFile = (uid: string, patch: Partial<FileItem>) =>
    setFileQueue((prev) => prev.map((f) => (f.uid === uid ? { ...f, ...patch } : f)));

  // ── 监听 ETL 进度 (带超时和重试) ──
  const watchEtl = (uid: string, taskId: string) => {
    // WebSocket 实时推送 (首选)
    let ws: WSClient | null = null;
    try {
      ws = new WSClient(taskId);
      ws.onProgress((status: TaskStatus) => {
        updateFile(uid, { etlProgress: status.progress, etlStage: status.stage });
        if (status.status === 'completed') {
          ws?.disconnect();
          updateFile(uid, { status: 'etl-done', etlProgress: 100 });
          loadData();
        } else if (status.status === 'failed') {
          ws?.disconnect();
          updateFile(uid, { status: 'etl-failed', message: status.error });
        }
      });
      ws.connect();
    } catch {
      // WS 失败不影响轮询降级
    }

    // HTTP 轮询降级 (健壮版: 不因临时错误停止)
    const MAX_WAIT_MS = 30 * 60 * 1000; // 30 分钟超时
    const startTime = Date.now();
    let failureCount = 0;
    const MAX_FAILURES = 5;

    const poll = setInterval(async () => {
      if (Date.now() - startTime > MAX_WAIT_MS) {
        clearInterval(poll);
        ws?.disconnect();
        updateFile(uid, { status: 'etl-failed', message: 'ETL 处理超时 (超过 30 分钟)' });
        return;
      }

      try {
        const status = await api.getTaskStatus(taskId);
        failureCount = 0;

        updateFile(uid, { etlProgress: status.progress, etlStage: status.stage });
        if (status.status === 'completed') {
          clearInterval(poll);
          ws?.disconnect();
          updateFile(uid, { status: 'etl-done', etlProgress: 100 });
          loadData();
        } else if (status.status === 'failed') {
          clearInterval(poll);
          ws?.disconnect();
          updateFile(uid, { status: 'etl-failed', message: status.error || 'ETL 处理失败' });
        }
      } catch {
        failureCount++;
        if (failureCount >= MAX_FAILURES) {
          clearInterval(poll);
          ws?.disconnect();
          updateFile(uid, { status: 'etl-failed', message: '无法获取任务状态 (多次重试失败)' });
        }
      }
    }, 3000);
  };

  // ── 批量上传 ──
  const handleBatchUpload = async (files: File[]) => {
    if (files.length === 0) return;

    const newItems: FileItem[] = files.map((f) => ({
      uid: `${Date.now()}-${f.name}`,
      name: f.name,
      size: f.size,
      status: 'pending' as FileItemStatus,
    }));
    setFileQueue((prev) => [...newItems, ...prev]);
    setBatchUploading(true);

    newItems.forEach((item) => updateFile(item.uid, { status: 'uploading' }));

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await api.batchUpload(files, controller.signal);

      res.results.forEach((r: BatchUploadResult) => {
        const item = newItems.find((n) => n.name === r.filename);
        if (!item) return;

        if (r.status === 'accepted') {
          updateFile(item.uid, { status: 'accepted', taskId: r.task_id });
          if (r.task_id) watchEtl(item.uid, r.task_id);
        } else if (r.status === 'skipped') {
          updateFile(item.uid, { status: 'skipped', message: r.message });
        } else {
          updateFile(item.uid, { status: 'rejected', message: r.message });
        }
      });

      const summaryParts: string[] = [];
      if (res.accepted > 0) summaryParts.push(`✅ ${res.accepted} 个已提交`);
      if (res.skipped > 0) summaryParts.push(`⏭️ ${res.skipped} 个已跳过（重复）`);
      if (res.rejected > 0) summaryParts.push(`❌ ${res.rejected} 个失败`);
      notification.success({
        message: '批量上传完成',
        description: summaryParts.join(' | ') || '全部处理完成',
        duration: 5,
      });

      loadData();
    } catch (err: any) {
      if (err.name === 'CanceledError' || err.name === 'AbortError') {
        newItems.forEach((item) => updateFile(item.uid, { status: 'cancelled', message: '用户取消' }));
        notification.info({ message: '批量上传已取消', duration: 3 });
      } else {
        message.error(`批量上传失败: ${err?.message || '未知错误'}`);
        newItems.forEach((item) =>
          updateFile(item.uid, { status: 'rejected', message: '上传请求失败' }),
        );
      }
    }

    setBatchUploading(false);
    abortRef.current = null;
  };

  // ── 扫描已有文件 ──
  const scanExistingFiles = async () => {
    try {
      const res = await api.getExistingFiles();
      const unprocessed = res.files
        .filter((f) => !f.in_db)
        .map((f) => f.name);
      setExistingFiles(unprocessed);
      setExistingModalOpen(true);
    } catch {
      // 降级：使用知识库中的文件名推断
      const docNames = new Set(docs.map((d) => d.filename));
      setExistingFiles(Array.from(docNames));
      setExistingModalOpen(true);
    }
  };

  // ── 处理已有文件 ──
  const processExistingFiles = async () => {
    if (existingFiles.length === 0) return;
    setProcessingExisting(true);

    // 通过 API 获取文件内容
    const controller = new AbortController();
    existingAbortRef.current = controller;
    const signal = controller.signal;
    const filesToUpload: File[] = [];
    for (const name of existingFiles) {
      if (signal.aborted) break;
      try {
        const resp = await fetch(`/api/v1/knowledge/existing-files/${encodeURIComponent(name)}`, { signal });
        if (resp.ok) {
          const blob = await resp.blob();
          filesToUpload.push(new File([blob], name, { type: 'application/pdf' }));
        }
      } catch {
        // 忽略单个文件失败
      }
    }

    if (!signal.aborted && filesToUpload.length > 0) {
      setExistingModalOpen(false);
      await handleBatchUpload(filesToUpload);
    } else {
      notification.info({
        message: '需要手动上传',
        description: `以下文件需要手动拖拽上传: ${existingFiles.join(', ')}`,
        duration: 8,
      });
      setExistingModalOpen(false);
    }
    setProcessingExisting(false);
    existingAbortRef.current = null;
  };

  const cancelExistingProcessing = () => {
    if (existingAbortRef.current) {
      existingAbortRef.current.abort();
      existingAbortRef.current = null;
    }
    setExistingModalOpen(false);
    setProcessingExisting(false);
  };

  const cancelUpload = () => {
    if (abortRef.current) abortRef.current.abort();
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const res = await api.deleteDocument(deleteTarget.id);
      if (res.deleted) {
        message.success(res.message);
        loadData();
      } else {
        message.warning(res.message);
      }
    } catch (err: any) {
      message.error(`删除失败: ${err?.message || '未知错误'}`);
    }
    setDeleting(false);
    setDeleteTarget(null);
  };

  const clearDone = () =>
    setFileQueue((prev) =>
      prev.filter((f) =>
        f.status === 'pending' || f.status === 'uploading' || f.status === 'accepted',
      ),
    );

  const activeCount = fileQueue.filter(
    (f) => f.status === 'pending' || f.status === 'uploading' || f.status === 'accepted',
  ).length;

  // ── ETL 阶段 Tips ──
  const etlStageTooltip = (stage: string) => {
    const stageMap: Record<string, string> = {
      'dedup': '内容去重校验',
      'parsing': 'PDF 文本解析',
      'vectorizing': 'BGE 向量化',
      'bm25': 'BM25 索引重建',
      'db_record': '元数据写入',
      'cache_invalidation': '缓存刷新',
    };
    return stageMap[stage] || stage;
  };

  // ── 文档列表列 ──
  const columns = [
    {
      title: '文件名', dataIndex: 'filename', key: 'filename', ellipsis: true,
      render: (name: string) => (
        <Space>
          <FileOutlined />
          <Text ellipsis style={{ maxWidth: 300 }}>{name}</Text>
        </Space>
      ),
    },
    { title: '版本', dataIndex: 'version', key: 'version', width: 60 },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 100,
      render: (s: string) => (
        <Tag color={s === 'done' ? 'success' : s === 'failed' ? 'error' : 'processing'}>{s}</Tag>
      ),
    },
    {
      title: '分块', dataIndex: 'chunks_count', key: 'chunks_count', width: 70,
      sorter: (a: any, b: any) => (a.chunks_count || 0) - (b.chunks_count || 0),
    },
    { title: '入库时间', dataIndex: 'upload_time', key: 'upload_time', width: 170,
      sorter: (a: any, b: any) => new Date(a.upload_time).getTime() - new Date(b.upload_time).getTime(),
    },
    {
      title: '操作', key: 'action', width: 180,
      render: (_: any, record: Document) => (
        <Space>
          <Tooltip title="在图谱中探索该文档的实体">
            <Button
              type="link"
              size="small"
              icon={<SearchOutlined />}
              onClick={() => {
                // 跳转到图谱页并搜索文档中的核心实体
                const entity = record.filename
                  .replace(/_/g, ' ')
                  .replace(/\.pdf$/i, '')
                  .replace(/\d{4}.*$/, '')
                  .trim();
                window.location.href = `/graph?q=${encodeURIComponent(entity)}`;
              }}
            >
              探索
            </Button>
          </Tooltip>
          <Tooltip title="删除文档及相关数据">
            <Button
              type="link"
              danger
              size="small"
              icon={<DeleteOutlined />}
              onClick={() => setDeleteTarget(record)}
            >
              删除
            </Button>
          </Tooltip>
        </Space>
      ),
    },
  ];

  const fmtSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const statusBadge = (item: FileItem) => {
    const map: Record<FileItemStatus, { color: 'processing' | 'success' | 'error' | 'default' | 'warning'; text: string }> = {
      pending: { color: 'default', text: '等待中' },
      uploading: { color: 'processing', text: '上传中...' },
      accepted: { color: 'processing', text: 'ETL 进行中' },
      skipped: { color: 'warning', text: '已跳过（重复）' },
      rejected: { color: 'error', text: '已拒绝' },
      'etl-done': { color: 'success', text: 'ETL 完成' },
      'etl-failed': { color: 'error', text: 'ETL 失败' },
      cancelled: { color: 'default', text: '已取消' },
    };
    const { color, text } = map[item.status];
    return <Badge status={color} text={text} />;
  };

  const handleFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    handleBatchUpload(Array.from(files));
    e.target.value = '';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!batchUploading) setDragOver(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (batchUploading) return;
    const files = Array.from(e.dataTransfer.files).filter((f) => f.name.endsWith('.pdf'));
    if (files.length === 0) {
      message.warning('目前仅支持 PDF 文件');
      return;
    }
    handleBatchUpload(files);
  };

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={handleFileSelected}
      />

      {/* ── 统计卡片 ── */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="文档总数" value={stats.documents} prefix={<FolderOpenOutlined />} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="分块总数" value={stats.chunks} prefix={<ApiOutlined />} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card><Statistic title="查询次数" value={stats.queries} /></Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Space>
              <Button
                type="primary"
                icon={<UploadOutlined />}
                loading={batchUploading}
                onClick={() => fileInputRef.current?.click()}
              >
                {batchUploading ? '上传中...' : '批量上传文档'}
              </Button>
              {batchUploading && (
                <Button danger icon={<StopOutlined />} onClick={cancelUpload}>取消</Button>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      {/* ── 操作栏 ── */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={24}>
          <Card size="small">
            <Space>
              <Button
                icon={<PlayCircleOutlined />}
                onClick={scanExistingFiles}
              >
                处理已有文件
              </Button>
              <Button
                icon={<ThunderboltOutlined />}
                onClick={() => window.open('/graph', '_blank')}
              >
                知识图谱探索
              </Button>
              <Text type="secondary" style={{ fontSize: 12 }}>
                拖拽 产品说明书 / 政策规则文档 (PDF/Markdown) 到下方区域，自动入库并建立检索索引
              </Text>
              {sysStatus && (
                <Popover
                  title="服务状态"
                  content={
                    <div style={{ fontSize: 12 }}>
                      {Object.entries(sysStatus.services).map(([name, s]) => (
                        <div key={name} style={{ marginBottom: 4 }}>
                          <Badge
                            status={s === 'ok' ? 'success' : s === 'degraded' ? 'warning' : 'error'}
                            text={`${name}: ${s === 'ok' ? '正常' : s === 'degraded' ? '降级' : '离线'}`}
                          />
                        </div>
                      ))}
                    </div>
                  }
                  trigger="hover"
                >
                  <Badge
                    status={sysStatus.status === 'ok' ? 'success' : 'warning'}
                    text={sysStatus.status === 'ok' ? '所有服务正常' : '部分服务异常'}
                    style={{ cursor: 'pointer' }}
                  />
                </Popover>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      {/* ── 拖拽上传区 ── */}
      <Card
        size="small"
        style={{
          marginBottom: 16,
          cursor: batchUploading ? 'not-allowed' : 'pointer',
          borderColor: dragOver ? '#1677ff' : undefined,
          borderStyle: dragOver ? 'dashed' : undefined,
          background: dragOver ? '#e6f4ff' : undefined,
          transition: 'all 0.2s',
        }}
        onClick={() => !batchUploading && fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div style={{ textAlign: 'center', padding: '40px 0', pointerEvents: 'none' }}>
          <InboxOutlined style={{ fontSize: 48, color: dragOver ? '#1677ff' : '#999' }} />
          <p style={{ marginTop: 16, fontSize: 16, color: dragOver ? '#1677ff' : '#333' }}>
            {dragOver ? '松开以上传文件' : '点击或拖拽文档到此处 (PDF)'}
          </p>
          <p style={{ color: '#999' }}>支持多文件批量上传 · 每文件最大 50MB · 上传后自动执行 8 阶段 ETL</p>
        </div>
      </Card>

      {/* ── 上传队列进度 ── */}
      {fileQueue.length > 0 && (
        <Card
          title={`上传队列 (${activeCount} 个进行中)`}
          size="small"
          style={{ marginBottom: 16 }}
          extra={
            <Space>
              {batchUploading && (
                <Button size="small" danger icon={<StopOutlined />} onClick={cancelUpload}>
                  取消全部
                </Button>
              )}
              <Button size="small" onClick={clearDone}>清除已完成</Button>
            </Space>
          }
        >
          <List
            size="small"
            dataSource={fileQueue}
            renderItem={(item) => (
              <List.Item
                actions={
                  item.status === 'uploading'
                    ? [<Button key="cancel" size="small" danger icon={<StopOutlined />} onClick={cancelUpload}>取消</Button>]
                    : undefined
                }
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <Text ellipsis style={{ maxWidth: 400 }}>{item.name}</Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>{fmtSize(item.size)}</Text>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Space>
                        {statusBadge(item)}
                        {item.message && <Text type="secondary" style={{ fontSize: 12 }}>{item.message}</Text>}
                        {item.etlStage && (
                          <Popover content={etlStageTooltip(item.etlStage)}>
                            <Tag style={{ cursor: 'pointer' }}>{item.etlStage}</Tag>
                          </Popover>
                        )}
                      </Space>
                      {(item.status === 'uploading' || item.status === 'accepted' || item.status === 'etl-failed') && (
                        <Progress
                          percent={item.etlProgress || (item.status === 'uploading' ? 10 : undefined)}
                          size="small"
                          status={item.status === 'etl-failed' ? 'exception' : 'active'}
                          style={{ marginBottom: 0 }}
                        />
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* ── 文档列表 ── */}
      <Card title={`📄 已入库文档 (${docs.length})`} size="small">
        <Table
          dataSource={docs}
          columns={columns}
          rowKey="id"
          loading={loading}
          size="small"
          pagination={{ pageSize: 10, showSizeChanger: true }}
          locale={{ emptyText: <Empty description="暂无理财知识文档，请上传 产品说明书 / 政策规则" /> }}
        />
      </Card>

      {/* ── 删除确认弹窗 ── */}
      <Modal
        title="确认删除文档"
        open={!!deleteTarget}
        onOk={handleDelete}
        onCancel={() => setDeleteTarget(null)}
        confirmLoading={deleting}
        okText="确认删除"
        cancelText="取消"
        okButtonProps={{ danger: true }}
      >
        {deleteTarget && (
          <Space direction="vertical">
            <Text>确定要删除文档 <Text strong>{deleteTarget.filename}</Text> 吗？</Text>
            <Text type="secondary" style={{ fontSize: 12 }}>
              将同时清理：SQLite 记录、PDF 源文件、FAISS 向量、Neo4j 图谱数据
            </Text>
          </Space>
        )}
      </Modal>

      {/* ── 已有文件处理弹窗 ── */}
      <Modal
        title="处理已有 PDF 文件"
        open={existingModalOpen}
        onOk={processExistingFiles}
        onCancel={cancelExistingProcessing}
        confirmLoading={processingExisting}
        okText="开始处理"
        cancelText="取消"
      >
        {existingFiles.length > 0 ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text>以下文件已在 data_reports 目录中，尚未入库：</Text>
            <List
              size="small"
              dataSource={existingFiles}
              renderItem={(name) => (
                <List.Item>
                  <Space><FileOutlined /><Text>{name}</Text></Space>
                </List.Item>
              )}
            />
            <Text type="secondary">点击"开始处理"将自动对这些文件执行 ETL 流水线</Text>
          </Space>
        ) : (
          <Empty description="所有文件已处理完毕" />
        )}
      </Modal>
    </div>
  );
};
