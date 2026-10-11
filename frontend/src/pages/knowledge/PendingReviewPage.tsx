import React, { useEffect, useState, useCallback } from 'react';
import {
  Table, Button, Card, Row, Col, Tag, Space, Typography, message,
  Modal, Input, Empty, Badge, Tooltip, Alert,
} from 'antd';
import {
  CheckOutlined, CloseOutlined, FileTextOutlined,
  ReloadOutlined, EyeOutlined, DownOutlined, UpOutlined,
  AppstoreOutlined, UnorderedListOutlined, ArrowLeftOutlined,
} from '@ant-design/icons';
import { api } from '../../services/api';
import { PALETTE } from '../../styles/theme';

const { Text, Title } = Typography;
const { TextArea } = Input;

interface PendingDoc {
  id: number;
  filename: string;
  status: string;
  upload_time: string;
  chunks_count: number;
  version: number;
  content_hash: string;
  file_size: number;
  source_label: string;
}

interface ChunkItem {
  chunk_id: string;
  source: string;
  section: string;
  page: number;
  content: string;
  content_hash: string;
  length: number;
  chunk_type: string;  // parent / child / normal
  parent_id: string;
}

export const PendingReviewPage: React.FC = () => {
  const [docs, setDocs] = useState<PendingDoc[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<PendingDoc | null>(null);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [expandedParents, setExpandedParents] = useState<Set<string>>(new Set());
  const [rejectModalOpen, setRejectModalOpen] = useState(false);
  const [rejectReason, setRejectReason] = useState('');
  const [approving, setApproving] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [currentParentIdx, setCurrentParentIdx] = useState(0);
  const [reviewedParents, setReviewedParents] = useState<Set<string>>(new Set());

  const loadDocs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listPendingDocs();
      setDocs(res.items || []);
    } catch (err: any) {
      message.error(`加载待审核列表失败：${err?.response?.data?.detail || err?.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  const loadChunks = useCallback(async (docId: number) => {
    setChunksLoading(true);
    setExpandedParents(new Set());
    setCurrentParentIdx(0);
    setReviewedParents(new Set());
    try {
      const res = await api.getPendingChunks(docId, 1, 2000);
      setChunks(res.items || []);
    } catch (err: any) {
      message.error(`加载分块预览失败：${err?.response?.data?.detail || err?.message}`);
    } finally {
      setChunksLoading(false);
    }
  }, []);

  const handleSelectDoc = (doc: PendingDoc) => {
    setSelectedDoc(doc);
    loadChunks(doc.id);
  };

  const handleBack = () => {
    setSelectedDoc(null);
    setChunks([]);
    setExpandedParents(new Set());
    setCurrentParentIdx(0);
    setReviewedParents(new Set());
  };

  const handleApprove = async () => {
    if (!selectedDoc) return;
    Modal.confirm({
      title: '确认审核通过？',
      content: (
        <div>
          <p>文档：<Text strong>{selectedDoc.filename}</Text></p>
          <p>分块数：<Text strong>{selectedDoc.chunks_count}</Text></p>
          <Alert
            type="info"
            showIcon
            message="整文档入库"
            description="所有分块将一次性写入正式索引（Milvus + BM25），入库后立即可被检索。"
          />
        </div>
      ),
      okText: '确认入库',
      cancelText: '取消',
      onOk: async () => {
        setApproving(true);
        try {
          await api.approvePendingDoc(selectedDoc.id);
          message.success(`文档「${selectedDoc.filename}」已入库`);
          handleBack();
          loadDocs();
        } catch (err: any) {
          message.error(`入库失败：${err?.response?.data?.detail || err?.message}`);
        } finally {
          setApproving(false);
        }
      },
    });
  };

  const handleReject = () => {
    setRejectReason('');
    setRejectModalOpen(true);
  };

  const confirmReject = async () => {
    if (!selectedDoc) return;
    setRejecting(true);
    try {
      await api.rejectPendingDoc(selectedDoc.id, rejectReason);
      message.success(`文档「${selectedDoc.filename}」已拒绝`);
      setRejectModalOpen(false);
      setSelectedDoc(null);
      setChunks([]);
      loadDocs();
    } catch (err: any) {
      message.error(`拒绝失败：${err?.response?.data?.detail || err?.message}`);
    } finally {
      setRejecting(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  // 按父块分组: 父块 + 它的子块
  const groupedChunks = () => {
    const parents = chunks.filter(c => c.chunk_type === 'parent' || !c.parent_id);
    const childrenByParent = new Map<string, ChunkItem[]>();
    chunks.filter(c => c.chunk_type === 'child' && c.parent_id).forEach(c => {
      const list = childrenByParent.get(c.parent_id) || [];
      list.push(c);
      childrenByParent.set(c.parent_id, list);
    });
    return parents.map(p => ({
      parent: p,
      children: childrenByParent.get(p.chunk_id) || [],
    }));
  };

  const toggleParent = (chunkId: string) => {
    setExpandedParents(prev => {
      const next = new Set(prev);
      if (next.has(chunkId)) next.delete(chunkId);
      else next.add(chunkId);
      return next;
    });
  };

  const expandAll = () => {
    const allIds = groupedChunks().map(g => g.parent.chunk_id);
    setExpandedParents(new Set(allIds));
  };

  const collapseAll = () => {
    setExpandedParents(new Set());
  };

  const docColumns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      ellipsis: true,
      render: (text: string, record: PendingDoc) => (
        <Tooltip title={text}>
          <Space size={4}>
            <FileTextOutlined style={{ color: PALETTE.primary, flexShrink: 0 }} />
            <Text strong={selectedDoc?.id === record.id} ellipsis style={{ maxWidth: 180 }}>{text}</Text>
          </Space>
        </Tooltip>
      ),
    },
    {
      title: '分块',
      dataIndex: 'chunks_count',
      key: 'chunks_count',
      width: 60,
      render: (v: number) => <Tag color="blue" style={{ fontSize: 11 }}>{v}</Tag>,
    },
    {
      title: '操作',
      key: 'action',
      width: 60,
      render: (_: any, record: PendingDoc) => (
        <Button
          type="link"
          size="small"
          icon={<EyeOutlined />}
          onClick={() => handleSelectDoc(record)}
        />
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]} align="top">
        {/* 左侧: 待审核列表 */}
        <Col span={selectedDoc ? 7 : 24}>
          <Card
            style={{ height: 'fit-content' }}
            size="small"
            title={
              <Space size="small">
                <span>待审核文档</span>
                <Badge count={docs.length} style={{ backgroundColor: PALETTE.accent }} />
              </Space>
            }
            extra={
              <Button icon={<ReloadOutlined />} onClick={loadDocs} loading={loading} size="small" type="text">
                刷新
              </Button>
            }
          >
            {docs.length === 0 && !loading ? (
              <Empty description="暂无待审核文档" />
            ) : (
              <Table
                dataSource={docs}
                columns={docColumns}
                rowKey="id"
                loading={loading}
                pagination={false}
                size="small"
                onRow={(record) => ({
                  onClick: () => handleSelectDoc(record),
                  style: { cursor: 'pointer',
                    backgroundColor: selectedDoc?.id === record.id ? '#e6f4ff' : undefined },
                })}
              />
            )}
          </Card>
        </Col>

        {/* 右侧: 分块预览 + 审核操作 */}
        {selectedDoc && (
          <Col span={17}>
            <Card
              style={{ height: 'calc(100vh - 48px)' }}
              styles={{ body: { height: '100%', display: 'flex', flexDirection: 'column', padding: 0 } }}
              title={
                <Space>
                  <Button
                    type="text"
                    icon={<ArrowLeftOutlined />}
                    onClick={handleBack}
                    size="small"
                    style={{ marginRight: 8 }}
                  >
                    返回
                  </Button>
                  <FileTextOutlined style={{ color: PALETTE.primary }} />
                  <span style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {selectedDoc.filename}
                  </span>
                  <Tag color="orange">待审核</Tag>
                </Space>
              }
              extra={
                <Space>
                  <Button
                    type="primary"
                    icon={<CheckOutlined />}
                    onClick={handleApprove}
                    loading={approving}
                    style={{ backgroundColor: '#52c41a', borderColor: '#52c41a' }}
                  >
                    审核通过并入库
                  </Button>
                  <Button
                    danger
                    icon={<CloseOutlined />}
                    onClick={handleReject}
                    loading={rejecting}
                  >
                    拒绝
                  </Button>
                </Space>
              }
            >
              {/* 统计栏 */}
              <div style={{
                padding: '10px 20px',
                background: '#fafafa',
                borderBottom: '1px solid #f0f0f0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexShrink: 0,
              }}>
                <Space size="large">
                  <span>
                    <AppstoreOutlined style={{ color: '#722ed1', marginRight: 4 }} />
                    <Text strong>{groupedChunks().length}</Text>
                    <Text type="secondary"> 父块</Text>
                  </span>
                  <span>
                    <UnorderedListOutlined style={{ color: '#52c41a', marginRight: 4 }} />
                    <Text strong>{chunks.filter(c => c.chunk_type === 'child').length}</Text>
                    <Text type="secondary"> 子块</Text>
                  </span>
                  <span>
                    <Text strong>{chunks.reduce((s, c) => s + c.length, 0)}</Text>
                    <Text type="secondary"> 字</Text>
                  </span>
                  <span>
                    <Text strong style={{ color: '#52c41a' }}>{reviewedParents.size}</Text>
                    <Text type="secondary">/{groupedChunks().length} 已审核</Text>
                  </span>
                </Space>
              </div>

              {/* 上方: 父块横向卡片列表 */}
              <div style={{
                flexShrink: 0,
                borderBottom: '2px solid #e8e8e8',
                background: '#fff',
              }}>
                <div style={{
                  display: 'flex',
                  overflowX: 'auto',
                  padding: '12px 16px',
                  gap: 10,
                  minHeight: 130,
                }}>
                  {chunksLoading ? (
                    <div style={{ textAlign: 'center', width: '100%', padding: 20 }}>
                      <Empty description="加载中..." />
                    </div>
                  ) : groupedChunks().length === 0 ? (
                    <div style={{ textAlign: 'center', width: '100%', padding: 20 }}>
                      <Empty description="暂无分块" />
                    </div>
                  ) : (
                    groupedChunks().map((group, idx) => {
                      const p = group.parent;
                      const isActive = idx === currentParentIdx;
                      const isReviewed = reviewedParents.has(p.chunk_id);
                      return (
                        <div
                          key={p.chunk_id || idx}
                          onClick={() => {
                            setCurrentParentIdx(idx);
                            setReviewedParents(prev => new Set(prev).add(p.chunk_id));
                          }}
                          style={{
                            flex: '0 0 auto',
                            width: 160,
                            padding: 10,
                            borderRadius: 8,
                            border: `2px solid ${isActive ? PALETTE.primary : '#f0f0f0'}`,
                            background: isActive ? '#e6f4ff' : isReviewed ? '#f6ffed' : '#fafafa',
                            cursor: 'pointer',
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'space-between',
                            transition: 'all 0.2s',
                          }}
                        >
                          <div>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                              <Text type="secondary" style={{ fontSize: 11 }}>#{idx + 1}</Text>
                              {isReviewed && <Tag color="green" style={{ fontSize: 10, margin: 0 }}>已审</Tag>}
                            </div>
                            <Text strong style={{
                              fontSize: 13,
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical',
                              overflow: 'hidden',
                              lineHeight: 1.4,
                              minHeight: 36,
                            }}>
                              {p.section || '无标题'}
                            </Text>
                          </div>
                          <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                            <Tag style={{ fontSize: 10, margin: 0 }}>{p.length}字</Tag>
                            {group.children.length > 0 && (
                              <Tag color="blue" style={{ fontSize: 10, margin: 0 }}>{group.children.length}子块</Tag>
                            )}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* 下方: 当前选中父块详情 */}
              <div style={{ flex: 1, overflow: 'auto', padding: '20px 24px' }}>
                {chunksLoading || groupedChunks().length === 0 ? (
                  <div style={{ textAlign: 'center', padding: 40 }}>
                    <Empty description="请先选择一个章节" />
                  </div>
                ) : (
                  (() => {
                    const group = groupedChunks()[currentParentIdx];
                    if (!group) return null;
                    const p = group.parent;
                    return (
                      <div>
                        {/* 父块内容 */}
                        <div style={{
                          padding: '16px 20px',
                          background: '#fafafa',
                          borderRadius: 8,
                          borderLeft: `4px solid ${PALETTE.primary}`,
                          marginBottom: 20,
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                            <Space size="small">
                              <Text strong style={{ fontSize: 16 }}>{p.section || '无标题'}</Text>
                              <Tag>p.{p.page}</Tag>
                              <Tag color="purple">{p.length}字</Tag>
                            </Space>
                            <Button
                              size="small"
                              type={reviewedParents.has(p.chunk_id) ? 'default' : 'primary'}
                              onClick={() => {
                                setReviewedParents(prev => {
                                  const next = new Set(prev);
                                  if (next.has(p.chunk_id)) next.delete(p.chunk_id);
                                  else next.add(p.chunk_id);
                                  return next;
                                });
                              }}
                            >
                              {reviewedParents.has(p.chunk_id) ? '取消已审' : '标记已审'}
                            </Button>
                          </div>
                          <Text style={{ whiteSpace: 'pre-wrap', lineHeight: 1.9, fontSize: 14, color: '#333' }}>
                            {p.content}
                          </Text>
                        </div>

                        {/* 子块列表 */}
                        {group.children.length > 0 && (
                          <div>
                            <Text type="secondary" style={{ fontSize: 13, marginBottom: 12, display: 'block' }}>
                              检索子块（{group.children.length}个，~150字/块，入Milvus/BM25索引）：
                            </Text>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                              {group.children.map((c, ci) => (
                                <div
                                  key={c.chunk_id}
                                  style={{
                                    flex: '1 1 calc(50% - 12px)',
                                    minWidth: 280,
                                    padding: 12,
                                    background: '#fff',
                                    borderRadius: 6,
                                    border: '1px solid #e8e8e8',
                                  }}
                                >
                                  <div style={{ marginBottom: 8 }}>
                                    <Tag color="blue" style={{ fontSize: 11 }}>子块{ci + 1} · {c.length}字</Tag>
                                  </div>
                                  <Text style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7, fontSize: 12.5, color: '#555' }}>
                                    {c.content}
                                  </Text>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()
                )}
              </div>
            </Card>
          </Col>
        )}
      </Row>

      {/* 拒绝原因弹窗 */}
      <Modal
        title="拒绝原因"
        open={rejectModalOpen}
        onOk={confirmReject}
        onCancel={() => setRejectModalOpen(false)}
        confirmLoading={rejecting}
        okText="确认拒绝"
        cancelText="取消"
      >
        <TextArea
          rows={4}
          placeholder="请输入拒绝原因（可选），例如：分块质量差、内容不相关、PDF解析错误..."
          value={rejectReason}
          onChange={(e) => setRejectReason(e.target.value)}
        />
      </Modal>
    </div>
  );
};
