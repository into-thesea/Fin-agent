"""
LLM 客户端工厂 — 统一 DeepSeek / Gemini 调用接口

通过 .env 的 LLM_PROVIDER 切换:
  LLM_PROVIDER=deepseek  → 使用 DeepSeek API（默认）
  LLM_PROVIDER=gemini    → 使用 Gemini API

用法:
    from src.llm.llm_client import create_client
    client = create_client()
    resp = client.chat(messages=[...], temperature=0.1)
    for token in client.chat_stream(messages=[...], temperature=0.1):
        print(token)
"""

import asyncio
import json
import logging
import threading
import time
from typing import AsyncGenerator, Generator

from dotenv import load_dotenv

# 加载 .env, 确保环境变量在任何导入顺序下都可用
load_dotenv()

from src.config import settings

logger = logging.getLogger(__name__)

from src.exceptions import LLMError

# OpenAI SDK 错误类型 (用于重试)
try:
    from openai import APIError, APITimeoutError, RateLimitError
except ImportError:
    class APIError(LLMError):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, code="OPENAI_API_ERROR", **kwargs)
    class APITimeoutError(LLMError):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, code="OPENAI_TIMEOUT", **kwargs)
    class RateLimitError(LLMError):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, code="OPENAI_RATE_LIMIT", **kwargs)

# ── 模型映射 ──────────────────────────────────────────
DEEPSEEK_MODEL = "deepseek-chat"
CHEAP_MODEL = "deepseek-chat"        # 路由/审查用
GEMINI_MODEL_DEFAULT = "gemini-2.5-pro-exp-03-25"
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"


def get_provider() -> str:
    """返回当前 LLM 提供商: 'deepseek' | 'gemini'"""
    return settings.llm_provider.strip().lower()


def get_api_key() -> str:
    """根据提供商返回对应的 API Key"""
    provider = get_provider()
    if provider == "gemini":
        return settings.gemini_api_key or ""
    return settings.deepseek_api_key


def get_model(default: str = None) -> str:
    """返回当前主模型名"""
    provider = get_provider()
    if provider == "gemini":
        return settings.gemini_model or GEMINI_MODEL_DEFAULT
    return default or settings.deepseek_model


def get_cheap_model() -> str:
    """返回便宜模型名（路由/审查用）"""
    provider = get_provider()
    if provider == "gemini":
        # Gemini 只有一个模型档次
        return settings.gemini_cheap_model or "gemini-2.5-flash"
    return settings.cheap_model


# ──────────────────────────────────────────────
# Function Calling 工具原语 (纯函数, 便于单测)
# ──────────────────────────────────────────────

def _schema_to_gemini(schema: dict):
    """JSON Schema (object) → genai types.Schema"""
    from google.genai import types
    props = {}
    for name, p in (schema.get("properties") or {}).items():
        ptype = p.get("type", "string")
        mapped = {
            "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
            "boolean": "BOOLEAN", "object": "OBJECT", "array": "ARRAY",
        }.get(ptype, "STRING")
        props[name] = types.Schema(type=mapped, description=p.get("description", ""))
    return types.Schema(type="OBJECT", properties=props, required=schema.get("required") or [])


def build_tools_payload(tools: list, provider: str = None):
    """
    tools: [{name, description, parameters}] → provider 专用声明
      provider=="gemini" → [genai Tool(function_declarations=[...])]
      其他 (deepseek/openai 兼容) → OpenAI function 格式列表
    """
    provider = (provider or get_provider()).strip().lower()
    if provider == "gemini":
        from google.genai import types
        declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=_schema_to_gemini(
                    t.get("parameters", {"type": "object", "properties": {}})
                ),
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=declarations)]
    return [
        {"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        }}
        for t in tools
    ]


def _parse_deepseek_response(raw: dict) -> dict:
    """OpenAI 兼容响应 → ToolResult dict"""
    try:
        msg = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {"type": "text", "text": ""}
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        calls = []
        for tc in tool_calls:
            try:
                arguments = json.loads(tc["function"]["arguments"] or "{}")
            except (json.JSONDecodeError, KeyError, TypeError):
                arguments = {}
            calls.append({
                "id": tc.get("id", ""),
                "name": tc["function"]["name"],
                "arguments": arguments,
            })
        return {"type": "tool_calls", "calls": calls}
    return {"type": "text", "text": msg.get("content") or ""}


def _parse_gemini_response(raw) -> dict:
    """genai generate_content 响应 → ToolResult dict"""
    try:
        parts = raw.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError):
        return {"type": "text", "text": ""}
    calls = []
    text_parts = []
    for p in parts:
        fc = getattr(p, "function_call", None)
        if fc is not None:
            args = {}
            if fc.args is not None:
                try:
                    args = dict(fc.args)
                except (TypeError, ValueError):
                    args = {}
            calls.append({
                "id": f"gem-{fc.name}", "name": fc.name, "arguments": args,
                # 保留原始 FunctionCall 对象: 其 proto 内含 thought_signature,
                # 回传时必须原样携带 (gemini-3.x 要求), 重建会丢失该字段
                "_raw_function_call": fc,
            })
        else:
            t = getattr(p, "text", None)
            if t:
                text_parts.append(t)
    if calls:
        return {"type": "tool_calls", "calls": calls}
    return {"type": "text", "text": "".join(text_parts)}


def parse_tool_response(raw, provider: str = None) -> dict:
    """归一化 LLM 响应为 ToolResult: {type:'text', text} 或 {type:'tool_calls', calls:[...]}"""
    provider = (provider or get_provider()).strip().lower()
    if provider == "gemini":
        if isinstance(raw, dict):
            return _parse_gemini_rest_response(raw)
        return _parse_gemini_response(raw)  # pydantic 响应 (旧路径, 不含 thoughtSignature)
    return _parse_deepseek_response(raw)


def _to_gemini_tool_messages(messages: list) -> tuple:
    """OpenAI 风格 messages (system/user/assistant[含tool_calls]/tool) → (gemini contents, system_instruction)"""
    from google.genai import types
    contents = []
    system_instruction = None
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if role == "system":
            system_instruction = content
        elif role == "tool":
            contents.append({
                "role": "user",
                "parts": [types.Part(function_response=types.FunctionResponse(
                    name=msg.get("name", ""),
                    response={"result": content or ""},
                ))],
            })
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content or ""}]})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = []
                for tc in tool_calls:
                    raw_fc = tc.get("_raw_function_call")
                    if raw_fc is not None:
                        # 原样回传 (保留 thought_signature)
                        parts.append(types.Part(function_call=raw_fc))
                        continue
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    parts.append(types.Part(function_call=types.FunctionCall(
                        name=fn.get("name", ""), args=args)))
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "model", "parts": [{"text": content or ""}]})
    return contents, system_instruction


def build_tools_payload_rest(tools: list) -> list:
    """tools → Gemini REST tools (plain dict, functionDeclarations 格式)"""
    return [{"functionDeclarations": [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]}]


def _build_gemini_contents(messages: list) -> tuple:
    """
    OpenAI 风格 messages → Gemini REST contents (list[dict])

    assistant 的 function_call 若带 _raw_function_call (dict, 含 thoughtSignature)
    则原样回传, 满足 gemini-3.x 的要求。
    """
    contents = []
    system_instruction = None
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if role == "system":
            system_instruction = content
        elif role == "tool":
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": {
                    "name": msg.get("name", ""),
                    "response": {"result": content or ""},
                }}],
            })
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content or ""}]})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = []
                for tc in tool_calls:
                    raw_fc = tc.get("_raw_function_call")
                    if isinstance(raw_fc, dict) and raw_fc.get("name"):
                        # 原始 functionCall dict (含 thoughtSignature), 原样回传
                        parts.append({"functionCall": raw_fc})
                        continue
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "model", "parts": [{"text": content or ""}]})
    return contents, system_instruction


def _parse_gemini_rest_response(raw: dict) -> dict:
    """Gemini REST 响应 dict → ToolResult dict (保留原始 functionCall dict 含 thoughtSignature)"""
    try:
        parts = raw["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return {"type": "text", "text": ""}
    calls = []
    text_parts = []
    for p in parts:
        fc = p.get("functionCall")
        if fc and isinstance(fc, dict):
            calls.append({
                "id": f"gem-{fc.get('name', '')}",
                "name": fc.get("name", ""),
                "arguments": fc.get("args") or {},
                "_raw_function_call": fc,  # 含 thoughtSignature
            })
        else:
            t = p.get("text")
            if t:
                text_parts.append(t)
    if calls:
        return {"type": "tool_calls", "calls": calls}
    return {"type": "text", "text": "".join(text_parts)}


def append_tool_result(messages: list, call: dict, result: str) -> list:
    """在 OpenAI 风格 messages 上追加工具执行结果 (deepseek/gemini 均以 OpenAI 风格存储)"""
    provider = get_provider().strip().lower()
    if provider == "gemini":
        messages.append({"role": "tool", "name": call["name"], "content": result})
    else:
        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
    return messages


class LLMResponse:
    """统一响应格式"""
    def __init__(self, content: str):
        self.content = content


class LLMClient:
    """统一的 LLM 调用接口"""

    def __init__(self, model: str = None, cheap: bool = False):
        """
        Args:
            model: 模型名，None 则自动选择
            cheap: 是否使用便宜模型（路由/审查任务）
        """
        self.provider = get_provider()
        self.api_key = get_api_key()

        if model:
            self.model_name = model
        elif cheap:
            self.model_name = get_cheap_model()
        else:
            self.model_name = get_model()

        # 初始化 HTTP 客户端
        self._http_client = None
        if self.provider == "gemini":
            from google import genai
            self._gemini_client = genai.Client(api_key=self.api_key)
            logger.info("Gemini 客户端已创建, 模型: %s", self.model_name)
        else:
            self._base_url = settings.deepseek_api_base
            logger.debug("DeepSeek 客户端已创建, 模型: %s", self.model_name)

    def chat(self, messages: list, temperature: float = 0.1,
             json_mode: bool = False) -> str:
        """非流式调用"""
        if self.provider == "gemini":
            return self._call_with_fallback("chat", messages, temperature, json_mode)
        return self._chat_deepseek(messages, temperature, json_mode)

    def chat_stream(self, messages: list, temperature: float = 0.1,
                    json_mode: bool = False) -> Generator[str, None, None]:
        """流式调用"""
        if self.provider == "gemini":
            yield from self._call_with_fallback_stream("chat_stream", messages, temperature, json_mode)
        else:
            yield from self._chat_deepseek_stream(messages, temperature, json_mode)

    # ── Function Calling ─────────────────────────────

    def chat_with_tools(self, messages: list, tools: list,
                        tool_choice: str = "auto", temperature: float = 0.1) -> dict:
        """
        带工具调用的对话 (非流式).

        Args:
            messages: OpenAI 风格消息 (system/user/assistant/tool)
            tools: [{name, description, parameters}]

        Returns:
            ToolResult dict:
              {"type": "text", "text": "..."} 或
              {"type": "tool_calls", "calls": [{"id", "name", "arguments"}]}
        """
        if self.provider == "gemini":
            return self._chat_gemini_tools(messages, tools, temperature)
        return self._chat_deepseek_tools(messages, tools, tool_choice, temperature)

    def _chat_deepseek_tools(self, messages, tools, tool_choice, temperature) -> dict:
        """DeepSeek function calling (原生 httpx, OpenAI 兼容格式)"""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "tools": build_tools_payload(tools, "deepseek"),
            "tool_choice": tool_choice,
        }

        def _do_request():
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DeepSeek API 返回 {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.json()

        raw = self._call_with_retry_http(_do_request)
        return parse_tool_response(raw, "deepseek")

    def _chat_gemini_tools(self, messages, tools, temperature) -> dict:
        """
        Gemini function calling — REST 直连 (保留 thoughtSignature)

        genai SDK 的 pydantic 层会剥离 thought_signature (gemini-3.x 要求回传),
        因此这里直接用 httpx 调 generateContent REST API, 原始 functionCall dict
        在会话中随消息原样回传。
        """
        import httpx

        contents, system_instruction = _build_gemini_contents(messages)
        body = {
            "contents": contents,
            "tools": build_tools_payload_rest(tools),
            "generationConfig": {"temperature": temperature},
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{GEMINI_REST_BASE}/models/{self.model_name}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        def _do_request():
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
                resp = client.post(url, json=body, headers=headers)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Gemini API 返回 {resp.status_code}: {resp.text[:300]}"
                    )
                return resp.json()

        raw = self._call_with_retry_http(_do_request)
        return _parse_gemini_rest_response(raw)

    def _chat_deepseek(self, messages, temp, json_mode) -> str:
        """使用 httpx 直接调用 DeepSeek API (避开 OpenAI SDK 在 Windows 上的 httpx 代理检测 hang)"""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        def _do_request():
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                if resp.status_code == 401:
                    raise PermissionError(
                        "DeepSeek API 认证失败 (401)，请检查 API Key"
                    )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DeepSeek API 返回 {resp.status_code}: {resp.text[:200]}"
                    )
                result = resp.json()
                return result["choices"][0]["message"]["content"] or ""

        return self._call_with_retry_http(_do_request)

    def _chat_deepseek_stream(self, messages, temp, json_mode) -> Generator:
        """流式调用 DeepSeek API (SSE)"""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "stream": True,
        }

        with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            with client.stream(
                "POST", f"{self._base_url}/chat/completions",
                json=body, headers=headers,
            ) as resp:
                if resp.status_code == 401:
                    raise PermissionError("DeepSeek API 认证失败 (401)")
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DeepSeek API 返回 {resp.status_code}"
                    )
                for line in resp.iter_lines():
                    if not line or line.startswith(":") or line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        try:
                            import json
                            chunk = json.loads(line[6:])
                            # 兼容空 choices chunk (keepalive/结束标记), 避免 IndexError
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    def _call_with_retry_http(self, func, max_retries=2, base_delay=1.0):
        """
        带指数后退重试的 HTTP 调用包装
        重试 5xx/超时/连接错误, 不重试 4xx
        """
        import httpx

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return func()
            except (httpx.TimeoutException, httpx.ConnectError,
                    httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM HTTP 失败 (attempt %d/%d), %.1fs 后重试: %s",
                        attempt + 1, max_retries + 1, delay, e,
                    )
                    time.sleep(delay)
                else:
                    logger.error("LLM HTTP 重试耗尽: %s", e)
                    raise
            except (PermissionError, ValueError):
                # 4xx 或认证错误 — 不重试
                raise
            except Exception:
                # 其他意外错误 — 重试一次
                if attempt < max_retries:
                    time.sleep(base_delay)
                else:
                    raise
        raise last_error

    # ── Async 调用 (httpx.AsyncClient) ──────────────────

    async def chat_async(self, messages: list, temperature: float = 0.1,
                         json_mode: bool = False) -> str:
        """非流式异步调用 (不阻塞事件循环线程)"""
        if self.provider == "gemini":
            # Gemini SDK 暂不支持原生 async, 回退到线程执行
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._call_with_fallback, "chat", messages, temperature, json_mode
            )
        return await self._chat_deepseek_async(messages, temperature, json_mode)

    async def _chat_deepseek_async(self, messages, temp, json_mode) -> str:
        """异步 httpx.AsyncClient 调用 DeepSeek API"""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async def _do_request():
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0)
            ) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                if resp.status_code == 401:
                    raise PermissionError(
                        "DeepSeek API 认证失败 (401)，请检查 API Key"
                    )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DeepSeek API 返回 {resp.status_code}: {resp.text[:200]}"
                    )
                result = resp.json()
                return result["choices"][0]["message"]["content"] or ""

        return await self._call_with_retry_http_async(_do_request)

    async def _call_with_retry_http_async(self, func, max_retries=2,
                                           base_delay=1.0):
        """
        异步版带指数后退重试的 HTTP 调用包装

        与同步版逻辑一致:
          - 重试 5xx/超时/连接错误
          - 不重试 4xx
          - asyncio.sleep() 不阻塞线程
        """
        import httpx

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except (httpx.TimeoutException, httpx.ConnectError,
                    httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM HTTP 失败 (attempt %d/%d), %.1fs 后重试: %s",
                        attempt + 1, max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("LLM HTTP 重试耗尽: %s", e)
                    raise
            except (PermissionError, ValueError):
                # 4xx 或认证错误 — 不重试
                raise
            except Exception:
                # 其他意外错误 — 重试一次
                if attempt < max_retries:
                    await asyncio.sleep(base_delay)
                else:
                    raise
        raise last_error

    async def chat_stream_async(self, messages: list, temperature: float = 0.1,
                                json_mode: bool = False) -> AsyncGenerator[str, None]:
        """流式异步调用 (async generator, 不阻塞事件循环)"""
        if self.provider == "gemini":
            # Gemini 通过队列桥接到 async
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _sync_gen():
                try:
                    for token in self._call_with_fallback_stream("chat_stream",
                                                                 messages, temperature,
                                                                 json_mode):
                        queue.put_nowait(token)
                except Exception as e:
                    queue.put_nowait(e)
                finally:
                    queue.put_nowait(None)  # sentinel

            thread = threading.Thread(target=_sync_gen, daemon=True)
            thread.start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        else:
            async for token in self._chat_deepseek_stream_async(
                messages, temperature, json_mode
            ):
                yield token

    async def _chat_deepseek_stream_async(
        self, messages, temp, json_mode
    ) -> AsyncGenerator[str, None]:
        """异步流式调用 DeepSeek API (SSE via httpx.AsyncClient)"""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "stream": True,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0)
        ) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions",
                json=body, headers=headers,
            ) as resp:
                if resp.status_code == 401:
                    raise PermissionError("DeepSeek API 认证失败 (401)")
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DeepSeek API 返回 {resp.status_code}"
                    )
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":") or line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    # ── Gemini (google-genai SDK) ──────────────────
    def _to_gemini_messages(self, messages: list) -> list:
        """将 OpenAI 格式的消息转为 Gemini 格式"""
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })

        return gemini_contents, system_instruction

    def _chat_gemini(self, messages, temp, json_mode) -> str:
        contents, system = self._to_gemini_messages(messages)
        config = dict(temperature=temp)
        if json_mode:
            config["response_mime_type"] = "application/json"
        if system:
            config["system_instruction"] = system

        # 移除 content 中的 system 消息后调用
        user_contents = [c for c in contents if c["role"] == "user"]
        if not user_contents:
            # 只有 system 消息时，增加一个占位 user 消息
            from google.genai import types
            resp = self._gemini_client.models.generate_content(
                model=self.model_name,
                contents="请回复。",
                config=types.GenerateContentConfig(**config),
            )
        else:
            from google.genai import types
            resp = self._gemini_client.models.generate_content(
                model=self.model_name,
                contents=user_contents,
                config=types.GenerateContentConfig(**config),
            )
        return resp.text or ""

    def _chat_gemini_stream(self, messages, temp, json_mode) -> Generator:
        contents, system = self._to_gemini_messages(messages)
        config = dict(temperature=temp)
        if json_mode:
            config["response_mime_type"] = "application/json"
        if system:
            config["system_instruction"] = system

        from google.genai import types
        user_contents = [c for c in contents if c["role"] == "user"] or ["请回复。"]

        stream = self._gemini_client.models.generate_content_stream(
            model=self.model_name,
            contents=user_contents,
            config=types.GenerateContentConfig(**config),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    # ── 模型降级机制 ──────────────────────────────────

    def _get_fallback_models(self) -> list:
        """从配置解析降级模型列表"""
        raw = settings.gemini_fallback_models or ""
        return [m.strip() for m in raw.split(",") if m.strip()]

    def _call_with_fallback(self, method: str, *args, **kwargs) -> any:
        """
        带降级链的 Gemini 调用包装器。
        method: 'chat' 或 'chat_stream'
        依次尝试主模型 → 降级模型链，第一个成功的返回。
        401/认证错误不降级。
        """
        models_to_try = [self.model_name] + self._get_fallback_models()
        seen = set()
        unique_models = []
        for m in models_to_try:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        last_error = None
        for i, model in enumerate(unique_models):
            if i > 0:
                logger.warning("Gemini 降级到模型: %s (前序: %s)", model,
                               ", ".join(unique_models[:i]))
                self.model_name = model
            try:
                if method == "chat":
                    return self._chat_gemini(*args, **kwargs)
            except Exception as e:
                last_error = e
                if isinstance(e, PermissionError) or "API_KEY" in str(e).upper():
                    raise
                logger.warning("Gemini 模型 %s 失败: %s", model, e)
                continue

        raise last_error  # 全部失败

    def _call_with_fallback_stream(self, method: str, *args, **kwargs):
        """
        流式降级包装器（返回 Generator）。
        预取第一个 token 提前捕获错误，失败则降级到下一个模型。
        """
        models_to_try = [self.model_name] + self._get_fallback_models()
        seen = set()
        unique_models = []
        for m in models_to_try:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        last_error = None
        for i, model in enumerate(unique_models):
            if i > 0:
                logger.warning("Gemini 流式降级到模型: %s", model)
                self.model_name = model
            try:
                gen = self._chat_gemini_stream(*args, **kwargs)
                first_token = next(gen)
                yield first_token
                yield from gen
                return
            except StopIteration:
                return
            except Exception as e:
                last_error = e
                if isinstance(e, PermissionError) or "API_KEY" in str(e).upper():
                    raise
                logger.warning("Gemini 流式模型 %s 失败: %s", model, e)
                continue

        raise last_error


# ── 便捷工厂函数 ──────────────────────────────────────

def create_client(model: str = None, cheap: bool = False) -> LLMClient:
    """创建 LLM 客户端"""
    return LLMClient(model=model, cheap=cheap)
