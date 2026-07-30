import unittest
from pathlib import Path


PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / ".opencode" / "agents" / "Agent.md"
)


class SplitPromptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        cls.normalized_prompt = " ".join(cls.prompt.split())

    def test_timestamp_is_not_required_for_ordered_data(self):
        self.assertIn(
            "A dataset can be temporally ordered without containing a datetime column.",
            self.normalized_prompt,
        )
        self.assertIn(
            "Do not infer that rows are independent merely because there is no timestamp",
            self.normalized_prompt,
        )

    def test_separate_streams_are_split_before_combining(self):
        self.assertIn(
            "split each stream at its own chronological or ordinal boundary before combining",
            self.normalized_prompt,
        )
        self.assertIn(
            "Do not concatenate streams and then randomly split their rows.",
            self.normalized_prompt,
        )

    def test_resetting_order_variables_require_audit_keys(self):
        self.assertIn("If an ordering variable resets", self.normalized_prompt)
        self.assertIn("source identity and original row", self.normalized_prompt)
        self.assertIn(
            "explicit per-stream boundary assertions",
            self.normalized_prompt,
        )

    def test_croissant_generation_is_owned_by_host_pipeline(self):
        self.assertIn(
            "Do not create or edit `croissant.json`",
            self.prompt,
        )
        self.assertNotIn("## 5. Generate ML Croissant", self.prompt)

    def test_large_tables_are_sampled_and_summarized_in_chunks(self):
        self.assertIn(
            "Never use the `read` tool to read a complete CSV",
            self.prompt,
        )
        self.assertIn("pd.read_csv(..., nrows=100)", self.prompt)
        self.assertIn("pd.read_csv(..., chunksize=...)", self.prompt)
        self.assertIn(
            "never paste full rows or large table contents into the conversation",
            self.normalized_prompt,
        )


if __name__ == "__main__":
    unittest.main()
