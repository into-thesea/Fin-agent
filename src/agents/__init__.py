# Fin-Agent 多智能体系统 (惰性导入)

def __getattr__(name):
    import importlib
    module_map = {
        "RouterAgent": "src.agents.router_agent",
        "QueryIntent": "src.agents.router_agent",
        "FactAgent": "src.agents.fact_agent",
        "TrendAgent": "src.agents.trend_agent",
        "CompareAgent": "src.agents.compare_agent",
        "ReviewerAgent": "src.agents.reviewer_agent",
        "AgenticAgent": "src.agents.agentic_agent",
    }
    if name in module_map:
        module = importlib.import_module(module_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RouterAgent", "QueryIntent",
    "FactAgent", "TrendAgent", "CompareAgent",
    "ReviewerAgent", "AgenticAgent",
]
