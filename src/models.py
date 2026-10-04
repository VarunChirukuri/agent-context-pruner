from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    role: MessageRole
    content: str
    pinned: bool = False
    tokens_estimated: int = 0

@dataclass
class PruningPolicy:
    max_token_budget: int = 4000
    preserve_system_prompts: bool = True
    preserve_recent_turns: int = 4
    summarize_stale_tool_outputs: bool = True
    deduplicate_identical_outputs: bool = True

@dataclass
class PruningMetrics:
    original_tokens: int
    pruned_tokens: int
    tokens_saved: int
    compression_ratio: float
