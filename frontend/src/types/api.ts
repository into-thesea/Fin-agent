// === 后端 API 类型定义 ===

// 知识库文档
export interface Document {
  id: number;
  filename: string;
  status: string;
  upload_time: string;
  chunks_count: number;
  content_hash: string | null;
  version: number;
}

// ETL 任务状态
export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  stage: string;
  error?: string;
  started_at?: string;
}

// 上传响应
export interface UploadResponse {
  status: 'accepted' | 'skipped';
  task_id?: string;
  doc_id?: number;
  message?: string;
  content_hash?: string;
  file?: string;
}

// 批量上传响应
export interface BatchUploadResult {
  filename: string;
  status: 'accepted' | 'skipped' | 'rejected';
  task_id?: string;
  doc_id?: number;
  message?: string;
  content_hash?: string;
}

export interface BatchUploadResponse {
  total: number;
  accepted: number;
  skipped: number;
  rejected: number;
  results: BatchUploadResult[];
}

// 查询响应
export interface QueryResponse {
  answer: string;
  entities: string | null;
  sources: string[];
  review: {
    score: number | null;
    verdict: string | null;
    claims: number;
  };
  handoff?: boolean;
  compliance?: { clean: boolean; hits?: string[] };
  session_id?: string;
  rewritten?: string | null;
  trace_id?: string;
}

// 知识库统计
export interface KnowledgeStats {
  documents: number;
  queries: number;
  chunks: number;
  db_path: string;
}

// 健康检查
export interface HealthStatus {
  status: 'ok' | 'degraded';
  version: string;
  timestamp: string;
  components: {
    redis: 'ok' | 'degraded';
    neo4j: 'ok' | 'degraded';
  };
}

// 监控指标
export interface Metrics {
  cache_hit_ratio: number;
  cache_keys: number;
  documents_count: number;
  queries_count: number;
  chunks_count: number;
  uptime_days: number;
  intent_stats?: Record<string, number>;
  handoff_count?: number;
  handoff_rate?: number;
}

// P9: SLI 指标聚合 (网关语义缓存 + 指标层)
export interface SLI {
  window: string;
  buckets: number;
  total: number;
  qps: number;
  success_rate: number;
  error_rate: number;
  timeout_rate: number;
  cache_hit_rate: number;
  avg_ms: number;
  p50_ms: number;
  p90_ms: number;
  p99_ms: number;
  samples: number;
}

// 服务状态
export interface SystemStatus {
  status: 'ok' | 'degraded';
  services: {
    redis: string;
    celery_worker: string;
    neo4j: string;
    faiss: string;
    model: string;
  };
}

// 转人工工单
export interface HandoffTicket {
  id: number;
  session_id: string;
  user_id: string;
  reason: string;
  status: 'open' | 'taken' | 'closed';
  agent_id?: string;
  created_at: string;
  resolved_at?: string;
  session_context?: { role: string; query: string; answer: string }[];
}

export interface HandoffMessage {
  id: number;
  role: string;
  content: string;
  created_at: string;
}

// 审计日志
export interface AuditLog {
  id: number;
  query: string;
  latency: number;
  tokens_used: number;
  cache_hit: boolean;
  user_id: string;
  timestamp: string;
}

// ── 评测看板 ──

export interface EvalRow {
  id: string;
  question: string;
  intent: string;
  evidence_covered: boolean;
  path_consistent: boolean | null;
  graph_hit: boolean;
  entities?: string[];
}

export interface EvalHistoryPoint {
  at: string;
  version: string;
  n: number;
  evidence_coverage: number | null;
  path_structure_consistency: number | null;
  graph_trigger_ratio: number | null;
}

export interface EvalSummary {
  current: {
    version: string;
    golden_n: number;
    n: number;
    metrics: Record<string, number | null>;
    rows: EvalRow[];
    online: unknown | null;
  } | null;
  history: EvalHistoryPoint[];
  metric_labels: Record<string, string>;
  available: { history: boolean; report: boolean; online: boolean };
  raw_history_count: number;
}

// ── 片段浏览 ──

export interface ChunkItem {
  chunk_id: string;
  source: string;
  section: string;
  page: number;
  content: string;
  content_hash: string;
  length: number;
}

export interface ChunkPage {
  total: number;
  page: number;
  size: number;
  items: ChunkItem[];
}
