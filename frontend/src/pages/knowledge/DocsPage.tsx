import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  Table, Button, Progress, Tag, Card, Row, Col, Statistic,
  message, Typography, Space, List, Badge, notification, Modal,
  Tooltip, Popover, Empty, Alert,
} from 'antd';
import {
  UploadOutlined, InboxOutlined, FolderOpenOutlined,
  ApiOutlined, DeleteOutlined, StopOutlined,
  FileOutlined, PlayCircleOutlined,
} from '@ant-design/icons';
import { api } from '../../services/api';
import type { Document, BatchUploadResult, SystemStatus } from '../../types/api';
import { PALETTE } from '../../styles/theme';

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

// 后端会把文件名里的空格/斜杠换成下划线, 匹配时必须做同样归一化
const safeName = (s: string) => s.replace(/ /g, '_').replace(/\//g, '_');

export const DocsPage: React.FC = () => {
  const [docs, setDocs] = useState<Document[]>([]);
  const [stats, setStats] = useState({ documents: 0, queries: 0, chunks: 0 });
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [fileQueue, setFileQueue] = useState<FileItem[]>([]);
  const [batchUploading, setBatchUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const existingAbortRef = useRef<AbortController | null>(null);
  // ETL 观察者清理登记: 卸载时统一收尾
  const etlCleanupRef = useRef<Array<() => void>>([]);

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
      setLoadError(null);
    } catch (err: any) {
      // 原先这里 catch {} 全静默: 后端 401/500 时页面只显示「0 文档」,
      // 使用者完全不知道是知识库空了还是服务挂了。
      const msg = err?.response?.data?.detail || err?.message || '未知错误';
      setLoadError(`知识库数据加载失败：${msg}`);
    }
    setLoading(false);
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // 卸载时收尾所有 ETL 观察者 (WS + 轮询定时器)
  useEffect(() => () => {
    etlCleanupRef.current.forEach((fn) => fn());
    etlCleanupRef.current = [];
  }, []);

  // 轮询系统服务状态 (每 30 秒)
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        setSysStatus(await api.getSystemStatus());
      } catch (err: any) {
        setSysStatus(null);  // 置空让顶部状态标识消失, 而不是继续显示上一次的快照
        console.error('服务状态查询失败', err);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const updateFile = (uid: string, patch: Partial<FileItem>) =>
    setFileQueue((prev) => prev.map((f) => (f.uid === uid ? { ...f, ...patch } : f)));

  // ── 监听 ETL 进度 (轮询) ──
  // 原先 WS + 轮询双通道并存: WS 只推 4~5 个阶段点, 却要为每条上传任务建一条
  // 连接、依赖 API 进程里的 Redis 监听线程; 轮询又必须留着兜丢消息。
  // 两条都半残, 已于 2026-09-14 统一为轮询。
  const watchEtl = (uid: string, taskId: string) => {
    let poll: ReturnType<typeof setInterval> | null = null;

    const cleanup = () => {
      if (poll) { clearInterval(poll); poll = null; }
      etlCleanupRef.current = etlCleanupRef.current.filter((fn) => fn !== cleanup);
    };
    etlCleanupRef.current.push(cleanup);

    const MAX_WAIT_MS = 30 * 60 * 1000; // 30 分钟超时
    const startTime = Date.now();
    let failureCount = 0;
    const MAX_FAILURES = 5;

    poll = setInterval(async () => {
      if (Date.now() - startTime > MAX_WAIT_MS) {
        cleanup();
        updateFile(uid, { status: 'etl-failed', message: 'ETL 处理超时 (超过 30 分钟)' });
        return;
      }

      try {
        const status = await api.getTaskStatus(taskId);
        failureCount = 0;

        updateFile(uid, { etlProgress: status.progress, etlStage: status.stage });
        if (status.status === 'completed') {
          cleanup();
          updateFile(uid, { status: 'etl-done', etlProgress: 100 });
          loadData();
        } else if (status.status === 'failed') {
          cleanup();
          updateFile(uid, { status: 'etl-failed', message: status.error || 'ETL 处理失败' });
        }
      } catch (err) {
        failureCount++;
        if (failureCount >= MAX_FAILURES) {
          cleanup();
          updateFile(uid, { status: 'etl-failed', message: '无法获取任务状态 (多次重试失败)' });
        } else {
          console.error('ETL 状态轮询失败', err);
        }
      }
    }, 3000);
  };

  // ── 批量上传 ──
  const handleBatchUpload = async (files: File[]) => {
    if (files.length === 0) return;

    // uid 带上下标: 同一批次里出现两个同名文件时, 原先 `${Date.now()}-${name}`
    // 会生成相同 uid, 两条队列项的状态互相覆盖。
    const newItems: FileItem[] = files.map((f, i) => ({
      uid: `${Date.now()}-${i}-${f.name}`,
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
        // 后端返回的是归一化后的文件名 (空格→下划线), 直接和原名比会匹配不上,
        // 结果就是该项永远停在「上传中」——既不上报进度也无法取消。
        const item =
          newItems.find((n) => safeName(n.name) === r.filename) ||
          newItems.find((n) => n.name === r.filename);
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
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || '未知错误';
      message.error(`扫描已有文件失败：${msg}`);
      setExistingFiles([]);
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
    const failed: string[] = [];
    for (const name of existingFiles) {
      if (signal.aborted) break;
      try {
        const blob = await api.getExistingFileBlob(name, signal);
        filesToUpload.push(new File([blob], name, { type: 'application/pdf' }));
      } catch (err) {
        if (signal.aborted) break;
        // 原先这里静默忽略: 全部失败也只会收到一句「需要手动上传」,
        // 看不出到底是 401、404 还是网络问题。
        console.error(`读取已有文件失败: ${name}`, err);
        failed.push(name);
      }
    }

    if (failed.length > 0) {
      notification.warning({
        message: `${failed.length} 个文件读取失败`,
        description: failed.slice(0, 5).join(', ') + (failed.length > 5 ? ' …' : ''),
        duration: 8,
      });
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
      'vectorizing': '向量化 + BM25 索引重建',
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
      title: '操作', key: 'action', width: 120,
      render: (_: any, record: Document) => (
        <Space>
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
    const fileList = Array.from(files);
    const pdfFiles = fileList.filter((f) => f.name.endsWith('.pdf'));
    const kbFiles = fileList.filter((f) => /\.(md|txt|jsonl)$/i.test(f.name));
    if (pdfFiles.length > 0) handleBatchUpload(pdfFiles);
    if (kbFiles.length > 0) handleKbUpload(kbFiles);
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

    const files = Array.from(e.dataTransfer.files);
    const pdfFiles = files.filter((f) => f.name.endsWith('.pdf'));
    const kbFiles = files.filter((f) => /\.(md|txt|jsonl)$/i.test(f.name));

    if (pdfFiles.length > 0) {
      handleBatchUpload(pdfFiles);
    }
    if (kbFiles.length > 0) {
      handleKbUpload(kbFiles);
    }
    if (pdfFiles.length === 0 && kbFiles.length === 0) {
      message.warning('仅支持 PDF / Markdown / TXT / JSONL 文件');
    }
  };

  // ── 内置知识库文件上传 (.md/.txt/.jsonl) ──
  const handleKbUpload = async (files: File[]) => {
    for (const file of files) {
      try {
        const res = await api.uploadKbFile(file);
        if (res.status === 'merged') {
          message.success(`${file.name}: 已合并到 ${res.target}，新增 ${res.added} 条`);
        } else {
          message.success(`${file.name}: 已保存，5秒后自动同步`);
        }
      } catch (err: any) {
        message.error(`${file.name} 上传失败: ${err?.response?.data?.detail || err?.message}`);
      }
    }
    loadData();
  };

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.md,.txt,.jsonl"
        style={{ display: 'none' }}
        onChange={handleFileSelected}
      />

      {/* ── 加载失败提示 ── */}
      {loadError && (
        <Alert
          type="error"
          showIcon
          closable
          message={loadError}
          onClose={() => setLoadError(null)}
          action={<Button size="small" onClick={loadData}>重试</Button>}
          style={{ marginBottom: 16 }}
        />
      )}

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
          borderColor: dragOver ? PALETTE.primary : undefined,
          borderStyle: dragOver ? 'dashed' : undefined,
          background: dragOver ? PALETTE.primarySoft : undefined,
          transition: 'all 0.2s',
        }}
        onClick={() => !batchUploading && fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div style={{ textAlign: 'center', padding: '40px 0', pointerEvents: 'none' }}>
          <InboxOutlined style={{ fontSize: 48, color: dragOver ? PALETTE.primary : PALETTE.textMuted }} />
          <p style={{ marginTop: 16, fontSize: 16, color: dragOver ? PALETTE.primary : PALETTE.text }}>
            {dragOver ? '松开以上传文件' : '点击或拖拽文档到此处'}
          </p>
          <p style={{ color: PALETTE.textMuted }}>
            PDF 走 ETL 解析入库 · Markdown/TXT/JSONL 保存到内置知识库后自动同步 · 每文件最大 50MB
          </p>
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
              将同时清理：分块文件、BM25 索引、向量索引、PDF 源文件、数据库记录
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
