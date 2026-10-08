"""
Fin-Agent Web 搜索工具

支持多种后端:
  - DuckDuckGo: 免费，无需 API Key (默认)
  - Bing: 需要 BING_API_KEY
  - SerpAPI: 需要 SERPAPI_API_KEY

用法:
  from src.tools.web_search import WebSearchTool

  searcher = WebSearchTool()
  results = searcher.search("稳盈添利30天 收益")
"""

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""


@dataclass
class SearchResponse:
    """搜索响应"""
    query: str = ""
    results: list = field(default_factory=list)
    total: int = 0
    backend: str = ""


class WebSearchTool:
    """
    Web 搜索工具 (多后端自适应)

    搜索后端选择优先级:
      1. BING_API_KEY → Bing Search API
      2. SERPAPI_API_KEY → SerpAPI (Google)
      3. DuckDuckGo (免费，无 API Key)
    """

    def __init__(self):
        self.bing_key = os.getenv("BING_API_KEY", "")
        self.serpapi_key = os.getenv("SERPAPI_API_KEY", "")
        self._backend = self._select_backend()

    def _select_backend(self) -> str:
        """自动选择可用后端"""
        if self.bing_key:
            return "bing"
        if self.serpapi_key:
            return "serpapi"
        return "duckduckgo"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def available(self) -> bool:
        """至少有一个后端可用"""
        return True  # DuckDuckGo 总是可用的

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        执行 Web 搜索

        Args:
            query: 搜索关键词
            max_results: 返回结果数

        Returns:
            SearchResponse
        """
        logger.info("Web 搜索: %s (backend=%s)", query[:60], self._backend)

        if self._backend == "bing":
            return self._search_bing(query, max_results)
        elif self._backend == "serpapi":
            return self._search_serpapi(query, max_results)
        else:
            return self._search_duckduckgo(query, max_results)

    def search_finance(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        金融专用搜索 (自动附加金融相关关键词)

        Args:
            query: 搜索关键词
            max_results: 返回结果数

        Returns:
            SearchResponse
        """
        enhanced_query = f"{query} 银行理财 产品说明"
        return self.search(enhanced_query, max_results)

    # ─── DuckDuckGo (免费, 无 API Key) ─────────

    def _search_duckduckgo(self, query: str, max_results: int) -> SearchResponse:
        """DuckDuckGo 搜索"""
        try:
            import urllib.parse
            import urllib.request

            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Fin-Agent/4.0"},
            )

            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            # DuckDuckGo 的 API 返回结构
            if data.get("AbstractText"):
                results.append(SearchResult(
                    title=data.get("Heading", ""),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("AbstractText", ""),
                    source="duckduckgo",
                ))

            # 相关主题
            for topic in data.get("RelatedTopics", [])[:max_results - 1]:
                if "Text" in topic:
                    results.append(SearchResult(
                        title=topic.get("Text", "")[:80],
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                        source="duckduckgo",
                    ))

            return SearchResponse(
                query=query,
                results=results,
                total=len(results),
                backend="duckduckgo",
            )

        except Exception as e:
            logger.warning("DuckDuckGo 搜索失败: %s", e)
            return SearchResponse(query=query, backend="duckduckgo")

    # ─── Bing Search API ────────────────────────

    def _search_bing(self, query: str, max_results: int) -> SearchResponse:
        """Bing Search API"""
        try:
            import urllib.parse
            import urllib.request

            url = f"https://api.bing.microsoft.com/v7.0/search?q={urllib.parse.quote(query)}&count={max_results}&mkt=zh-CN"

            req = urllib.request.Request(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key": self.bing_key,
                    "User-Agent": "Fin-Agent/4.0",
                },
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for item in data.get("webPages", {}).get("value", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="bing",
                ))

            return SearchResponse(
                query=query,
                results=results,
                total=len(results),
                backend="bing",
            )

        except Exception as e:
            logger.warning("Bing 搜索失败: %s", e)
            return SearchResponse(query=query, backend="bing")

    # ─── SerpAPI (Google) ───────────────────────

    def _search_serpapi(self, query: str, max_results: int) -> SearchResponse:
        """SerpAPI 搜索"""
        try:
            import urllib.parse
            import urllib.request

            params = urllib.parse.urlencode({
                "q": query,
                "api_key": self.serpapi_key,
                "num": max_results,
                "engine": "google",
            })
            url = f"https://serpapi.com/search?{params}"

            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google",
                ))

            return SearchResponse(
                query=query,
                results=results,
                total=len(results),
                backend="serpapi",
            )

        except Exception as e:
            logger.warning("SerpAPI 搜索失败: %s", e)
            return SearchResponse(query=query, backend="serpapi")

    # ─── 搜索结果格式化 ─────────────────────────

    def format_for_prompt(self, response: SearchResponse) -> str:
        """将搜索结果格式化为 LLM 友好的上下文"""
        if not response.results:
            return ""

        parts = [f"## Web 搜索结果 (来源: {response.backend})"]
        for i, r in enumerate(response.results, 1):
            parts.append(f"\n{i}. **{r.title}**")
            parts.append(f"   来源: {r.url}")
            parts.append(f"   摘要: {r.snippet[:300]}")

        return "\n".join(parts)


# ─── 便捷函数 ─────────────────────────────────

def search_web(query: str, max_results: int = 5) -> SearchResponse:
    """快速搜索"""
    tool = WebSearchTool()
    return tool.search(query, max_results)


def search_finance(query: str, max_results: int = 5) -> SearchResponse:
    """金融专用搜索"""
    tool = WebSearchTool()
    return tool.search_finance(query, max_results)
