/**
 * Fin-Agent WebSocket 客户端 — 实时 ETL 进度推送
 *
 * 增强:
 *   - 25s 心跳 Ping (防止代理/防火墙断开空闲连接)
 *   - onStatusChange 回调 + 公开 status getter
 *   - 错误日志 (不再静默吞异常)
 *   - intentionalClose 标记防止断开后重连
 *
 * 用法:
 *   const ws = new WSClient('task_xxx');
 *   ws.onProgress((data) => console.log(data.progress, data.stage));
 *   ws.onStatusChange((s) => console.log('WS status:', s));
 *   ws.connect();
 */
import type { TaskStatus } from '../types/api';

export type WSConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'failed';

type ProgressCallback = (status: TaskStatus) => void;
type StatusChangeCallback = (status: WSConnectionStatus) => void;

export class WSClient {
  private ws: WebSocket | null = null;
  private url: string;
  private progressCallbacks: ProgressCallback[] = [];
  private statusCallbacks: StatusChangeCallback[] = [];
  private reconnectAttempts = 0;
  private maxRetries = 5;
  private taskId: string;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private intentionalClose = false;
  private _status: WSConnectionStatus = 'disconnected';

  constructor(taskId: string) {
    this.taskId = taskId;
    const base = import.meta.env.VITE_WS_BASE || `ws://${window.location.hostname}:8000`;
    this.url = `${base}/ws/task/${taskId}`;
  }

  get status(): WSConnectionStatus {
    return this._status;
  }

  private setStatus(s: WSConnectionStatus) {
    this._status = s;
    this.statusCallbacks.forEach((cb) => cb(s));
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
      return;
    }

    this.intentionalClose = false;
    this.setStatus('connecting');

    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      console.warn(`[WSClient] 连接失败 (task=${this.taskId}):`, err);
      this.setStatus('failed');
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.setStatus('connected');
      this.startHeartbeat();
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as TaskStatus & { type?: string };
        if (data.type === 'pong') return;
        this.progressCallbacks.forEach((cb) => cb(data));
      } catch { /* ignore malformed */ }
    };

    this.ws.onclose = () => {
      this.stopHeartbeat();
      if (this.intentionalClose) {
        this.setStatus('disconnected');
        return;
      }
      if (this.reconnectAttempts < this.maxRetries) {
        const delay = 2000 * Math.pow(2, this.reconnectAttempts);
        setTimeout(() => {
          this.reconnectAttempts++;
          this.connect();
        }, delay);
      } else {
        this.setStatus('failed');
      }
    };

    this.ws.onerror = () => {
      console.warn(`[WSClient] WebSocket 错误 (task=${this.taskId})`);
      // onclose 会在 onerror 后触发，重连逻辑在 onclose 中
    };
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 25000);
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  onProgress(callback: ProgressCallback) {
    this.progressCallbacks.push(callback);
  }

  onStatusChange(callback: StatusChangeCallback) {
    this.statusCallbacks.push(callback);
  }

  send(data: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  disconnect() {
    this.intentionalClose = true;
    this.maxRetries = 0;
    this.stopHeartbeat();
    this.ws?.close();
    this.ws = null;
    this.progressCallbacks = [];
    this.statusCallbacks = [];
    this.setStatus('disconnected');
  }
}
