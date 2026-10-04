import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Message, MessageRole, PruningPolicy
from src.pruner import ContextPruner

class TestContextPruner(unittest.TestCase):
    def test_pruning_and_deduplication(self):
        policy = PruningPolicy(max_token_budget=150, preserve_recent_turns=2)
        pruner = ContextPruner(policy)
        long_text = "result " * 50
        msgs = [
            Message(MessageRole.SYSTEM, "System prompt"),
            Message(MessageRole.USER, "Run tool"),
            Message(MessageRole.TOOL, long_text),
            Message(MessageRole.ASSISTANT, "Acknowledged"),
            Message(MessageRole.USER, "Run again"),
            Message(MessageRole.TOOL, long_text),
            Message(MessageRole.USER, "Current question"),
            Message(MessageRole.ASSISTANT, "Final answer")
        ]
        pruned, metrics = pruner.prune_context(msgs)
        self.assertEqual(pruned[0].role, MessageRole.SYSTEM)
        self.assertGreater(metrics.tokens_saved, 0)
        self.assertLess(metrics.compression_ratio, 1.0)

if __name__ == "__main__":
    unittest.main()
