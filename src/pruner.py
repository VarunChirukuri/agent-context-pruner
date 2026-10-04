import json, hashlib
from typing import List, Tuple, Optional
from src.models import Message, MessageRole, PruningPolicy, PruningMetrics

class ContextPruner:
    def __init__(self, policy: Optional[PruningPolicy] = None):
        self.policy = policy or PruningPolicy()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 3.8)) if text else 0

    def prune_context(self, messages: List[Message]) -> Tuple[List[Message], PruningMetrics]:
        total_tokens = sum(self.estimate_tokens(m.content) for m in messages)
        if total_tokens <= self.policy.max_token_budget:
            return messages, PruningMetrics(total_tokens, total_tokens, 0, 1.0)

        pruned = []
        seen_hashes = set()
        cutoff = max(0, len(messages) - self.policy.preserve_recent_turns)

        for i, m in enumerate(messages):
            if m.role == MessageRole.SYSTEM and self.policy.preserve_system_prompts:
                pruned.append(m)
                continue
            if m.pinned or i >= cutoff:
                pruned.append(m)
                continue
            if m.role == MessageRole.TOOL:
                h = hashlib.md5(m.content.strip().encode("utf-8")).hexdigest()
                if h in seen_hashes and self.policy.deduplicate_identical_outputs:
                    pruned.append(Message(m.role, "[Identical tool output suppressed]"))
                    continue
                seen_hashes.add(h)
                if len(m.content) > 150 and self.policy.summarize_stale_tool_outputs:
                    pruned.append(Message(m.role, f"{m.content[:80]}... [truncated {len(m.content)-80} chars]"))
                    continue
            pruned.append(m)

        final_tokens = sum(self.estimate_tokens(m.content) for m in pruned)
        saved = max(0, total_tokens - final_tokens)
        ratio = round(final_tokens / max(1, total_tokens), 3)
        return pruned, PruningMetrics(total_tokens, final_tokens, saved, ratio)
