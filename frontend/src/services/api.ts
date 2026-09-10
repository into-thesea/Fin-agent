/**
 * Fin-Agent API Client
 * axios 实例 + 请求/响应拦截器 + 所有 API 端点封装
 */
import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  HealthStatus, UploadResponse, BatchUploadResponse, TaskStatus, QueryResponse,
  KnowledgeStats, Document, Metrics, AuditLog, SystemStatus, SLI,
  HandoffTicket, HandoffMessage,
} from '../types/api';

const BASE_URL = import.meta.env.VITE_API_BASE || '';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      timeout: 30000,
    });

    // 请求拦截器: 注入 JWT Token
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('fin_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      const traceId = crypto.randomUUID().slice(0, 12);
      config.headers['X-Trace-ID'] = traceId;
      return config;
    });

    // 响应拦截器: 统一错误处理
    this.client.interceptors.response.use(
      (res) => res,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('fin_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      },
    );
  }

  // ========== 健康检查 ==========
  async health(): Promise<HealthStatus> {
    const { data } = await this.client.get('/health');
    return data;
  }

  // ========== 问答 ==========
  async chat(query: string, timeout = 120000, sessionId = ''): Promise<QueryResponse> {
    if (!query?.trim()) {
      return { answer: '', entities: null, sources: [], review: { score: null, verdict: null, claims: 0 } };
    }
    const { data } = await this.client.post('/api/v1/chat/sync', { query, session_id: sessionId }, { timeout });
    return data;
  }

  /**
   * SSE 流式聊天
   *
   * @param query      用户问题
   * @param onResult   收到最终结果时回调 (含 answer/sources/review/handoff)
   * @param onStage    阶段更新时回调 (stage, message)
   * @param onError    出错时回调
   * @param sessionId  会话 ID (多轮上下文关联)
   * @param onHandoff  收到转人工事件时回调
   * @param onMeta     收到 result_meta (sources/entities) 时回调
   * @returns AbortController 用于取消请求
   */
  chatStream(
    query: string,
    onResult: (result: QueryResponse) => void,
    onStage?: (stage: string, message: string) => void,
    onError?: (error: string) => void,
    sessionId = '',
    onHandoff?: (ticketId?: number) => void,
    onMeta?: (sources: string[], entities: string | null) => void,
  ): AbortController {
    const controller = new AbortController();
    const token = localStorage.getItem('fin_token');

    if (!query?.trim()) {
      onError?.('查询内容不能为空');
      return controller;
    }

    let receivedResult = false;

    fetch(`${BASE_URL}/api/v1/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ query, session_id: sessionId }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text().catch(() => '');
          onError?.(`服务器错误 (${response.status}): ${text || response.statusText}`);
          return;
        }
        const reader = response.body?.getReader();
        if (!reader) {
          onError?.('无法读取响应流');
          return;
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // 保留可能不完整的行

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;

            try {
              const parsed = JSON.parse(trimmed.slice(6));

              switch (parsed.type) {
                case 'stage':
                  onStage?.(parsed.stage, parsed.message);
                  break;
                case 'result_meta':
                  onMeta?.(parsed.sources || [], parsed.entities || null);
                  break;
                case 'handoff':
                  onHandoff?.(parsed.ticket_id);
                  break;
                case 'result':
                  receivedResult = true;
                  onResult({
                    answer: parsed.answer || '',
                    sources: parsed.sources || [],
                    entities: parsed.entities || null,
                    review: parsed.review || { score: null, verdict: null, claims: 0 },
                    handoff: !!parsed.handoff,
                    compliance: parsed.compliance,
                    session_id: parsed.session_id,
                    trace_id: parsed.trace_id || '',
                  });
                  break;
                case 'error':
                  onError?.(parsed.message || '未知错误');
                  break;
                case 'done':
                  if (!receivedResult) {
                    onError?.('服务异常：流式响应未包含有效回答');
                  }
                  break;
              }
            } catch {
              // 跳过解析失败的行
            }
          }
        }
      })
      .catch((err) => {
        if (err.name === 'AbortError') return; // 用户取消，不报错
        onError?.(err.message || '网络连接异常，请检查网络后重试');
      });

    return controller;
  }

  // ========== 文档上传 ==========
  async uploadDocument(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await this.client.post('/api/v1/knowledge/upload', formData, {
      timeout: 60000,
    });
    return data;
  }

  async batchUpload(files: File[], signal?: AbortSignal): Promise<BatchUploadResponse> {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    const { data } = await this.client.post('/api/v1/knowledge/batch-upload', formData, {
      timeout: 300000,
      signal,
    });
    return data;
  }

  // ========== 文档删除 ==========
  async deleteDocument(docId: number): Promise<{ deleted: boolean; message: string; details: Record<string, boolean> }> {
    const { data } = await this.client.delete(`/api/v1/knowledge/documents/${docId}`);
    return data;
  }

  // ========== 任务状态 ==========
  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const { data } = await this.client.get(`/api/v1/tasks/${taskId}`);
    return data;
  }

  // ========== 知识库 ==========
  async getDocuments(): Promise<{ total: number; documents: Document[] }> {
    const { data } = await this.client.get('/api/v1/knowledge/documents');
    return data;
  }

  async getKnowledgeStats(): Promise<KnowledgeStats> {
    const { data } = await this.client.get('/api/v1/knowledge/stats');
    return data;
  }

  async getExistingFiles(): Promise<{ files: Array<{name: string; size: number; in_db: boolean}>; total: number; unprocessed: number }> {
    const { data } = await this.client.get('/api/v1/knowledge/existing-files');
    return data;
  }

  // ========== 监控 ==========
  async getMetrics(): Promise<Metrics> {
    const { data } = await this.client.get('/api/v1/monitor/metrics');
    return data;
  }

  async getAuditLogs(limit = 20): Promise<{ logs: AuditLog[] }> {
    const { data } = await this.client.get('/api/v1/monitor/audit-logs', { params: { limit } });
    return data;
  }

  // P9: SLI 指标聚合 (QPS/成功率/缓存命中率/延迟分位数)
  async getSLI(window = '1h'): Promise<SLI> {
    const { data } = await this.client.get('/api/v1/monitor/sli', { params: { window } });
    return data;
  }

  // ========== 转人工工单闭环 ==========
  async getHandoffQueue(): Promise<{ tickets: HandoffTicket[] }> {
    const { data } = await this.client.get('/api/v1/handoff/queue');
    return data;
  }
  async takeHandoff(id: number, agentId: string): Promise<void> {
    await this.client.post(`/api/v1/handoff/${id}/take`, { agent_id: agentId });
  }
  async replyHandoff(id: number, content: string): Promise<void> {
    await this.client.post(`/api/v1/handoff/${id}/reply`, { content });
  }
  async closeHandoff(id: number): Promise<void> {
    await this.client.post(`/api/v1/handoff/${id}/close`, {});
  }
  async getHandoffMessages(id: number): Promise<{ messages: HandoffMessage[] }> {
    const { data } = await this.client.get(`/api/v1/handoff/${id}/messages`);
    return data;
  }
  async suggestHandoff(id: number): Promise<{ draft: string }> {
    const { data } = await this.client.post(`/api/v1/handoff/${id}/suggest`);
    return data;
  }

  // ========== 系统状态 ==========
  async getSystemStatus(): Promise<SystemStatus> {
    const { data } = await this.client.get('/api/v1/system/status');
    return data;
  }

  // ========== 反馈 ==========
  async sendFeedback(sessionId: string, rating: number, comment = '', traceId = ''): Promise<{ ok: boolean }> {
    const { data } = await this.client.post('/api/v1/chat/feedback', {
      session_id: sessionId,
      rating,
      comment,
      trace_id: traceId,
    });
    return data;
  }

  // ========== Badcase 闭环 ==========
  async listBadcases(status = '', limit = 100, offset = 0): Promise<{ items: any[] }> {
    const { data } = await this.client.get('/api/v1/badcases', {
      params: { status, limit, offset },
    });
    return data;
  }

  async getBadcase(traceId: string): Promise<any> {
    const { data } = await this.client.get(`/api/v1/badcases/${encodeURIComponent(traceId)}`);
    return data;
  }

  async patchBadcase(traceId: string, status: string, note = ''): Promise<any> {
    const { data } = await this.client.patch(
      `/api/v1/badcases/${encodeURIComponent(traceId)}`, { status, note },
    );
    return data;
  }

  async promoteBadcase(traceId: string, note = ''): Promise<any> {
    const { data } = await this.client.post(
      `/api/v1/badcases/${encodeURIComponent(traceId)}/promote`, { note },
    );
    return data;
  }

  // ========== 认证 ==========
  async login(username: string, password: string): Promise<{ token: string; user: any }> {
    const { data } = await this.client.post('/api/v1/auth/login', { username, password });
    return data;
  }

  async getProfile(): Promise<any> {
    const { data } = await this.client.get('/api/v1/auth/profile');
    return data;
  }
}

export const api = new ApiClient();
