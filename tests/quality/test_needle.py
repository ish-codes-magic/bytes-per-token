from fastserve.quality.needle import QUESTION, build_case, grid, passed


class WordTokenizer:
    """One token per word: enough to test lengths and placement without downloading a tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return f"<user>{messages[0]['content']}</user><assistant>"


def _document(case):
    return case.prompt.removeprefix("<user>").split("\n\n")[0]


def test_needle_is_placed_at_the_requested_depth():
    tok = WordTokenizer()
    start = build_case(tok, 300, 0.0, "4729")
    end = build_case(tok, 300, 1.0, "4729")
    assert _document(start).startswith("The special magic number")
    assert _document(end).endswith("is 4729.")
    assert 250 <= len(_document(start).split()) <= 320  # about the requested context size
    assert start.prompt.count("4729") == 1 and QUESTION in start.prompt


def test_grid_is_reproducible_and_scoring_checks_the_secret():
    tok = WordTokenizer()
    a = grid(tok, [100, 200], [0.0, 0.5], ["11", "22"])
    assert len(a) == 8 and a == grid(tok, [100, 200], [0.0, 0.5], ["11", "22"])
    assert passed(a[0], "The number is 11.") and not passed(a[0], "I don't know.")
