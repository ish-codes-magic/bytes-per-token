from fastserve.quality.prompts import TASKS, chat_prompt_ids


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return f"<user>{messages[0]['content']}<assistant>"

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [ord(c) for c in text]


def test_chat_prompt_wraps_the_message_in_the_template():
    ids = chat_prompt_ids(FakeTokenizer(), "hi")
    assert "".join(map(chr, ids)) == "<user>hi<assistant>"


def test_task_builders_shape_each_dataset_row():
    assert TASKS["chat"].build({"instruction": "Name a fish.", "context": ""}) == "Name a fish."
    assert TASKS["chat"].build({"instruction": "Use the passage.", "context": "A passage."}) is None
    assert "```python\ndef f():\n```" in TASKS["code"].build({"prompt": "def f():\n"})
    assert TASKS["math"].build({"question": "2 + 2?"}).endswith("Solve it step by step.")
    summary = TASKS["summarize"].build({"article": "word " * 1000})
    assert (
        summary.startswith("Summarize this article") and len(summary.split()) < 470
    )  # truncated to 450 words
