"""Configuration management for the research engine.
"""
from typing import Optional

from langchain_core.runnables import RunnableConfig


class Configuration:
    """Configuration settings for the research engine."""
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        local_llm: str = "qwen3:30b-a3b",
        temperature: float = 0.0
    ):
        self.ollama_base_url = ollama_base_url
        self.local_llm = local_llm
        self.temperature = temperature
        
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig]) -> 'Configuration':
        """Create a Configuration instance from a RunnableConfig."""
        if not config:
            return cls()
            
        configurable = config.get("configurable", {})
        return cls(
            ollama_base_url=configurable.get("ollama_base_url", "http://localhost:11434"),
            local_llm=configurable.get("local_llm", "deepseek-r1:32b"),
            temperature=configurable.get("temperature", 0.0)
        ) 