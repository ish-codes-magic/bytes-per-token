from fastserve.quality.gsm8k import classify, is_looping, summarize, talks_past_answer


def test_repetition_is_looping_and_a_normal_solution_is_not():
    step = "She has 3 apples and buys 2 more apples at the shop. "
    assert is_looping(step * 3)
    solution = "Janet sells 16 - 3 - 4 = 9 eggs. She makes 9 * 2 = 18 dollars.\n#### 18"
    assert not is_looping(solution)


def test_classify_checks_correctness_then_loops_then_the_answer_marker():
    loop = "The answer is 5 and then we add 5 more to get " * 4
    assert classify(loop, correct=True) == "correct"  # flexible-extract may still find the right number
    assert classify(loop, correct=False) == "looping"
    assert classify("3 + 4 = 7\n#### 7", correct=False) == "wrong answer"
    assert classify("3 + 4 = 7, so she then goes to the", correct=False) == "no final answer"
    kept_talking = "3 + 4 = 7\n#### 7\n\nOkay, let's check: 3 + 4 is 8"
    assert classify(kept_talking, correct=False, strict=True) == "right, then kept talking"


def test_talks_past_answer_needs_words_after_the_final_answer():
    assert not talks_past_answer("3 + 4 = 7\n#### 7")
    assert not talks_past_answer("3 + 4 = 7\n#### 7\n\n")
    assert talks_past_answer("3 + 4 = 7\n#### 7\n\nOkay, let's see. So the problem")
    assert not talks_past_answer("no marker here at all, just words and words")


def test_summarize_groups_filters_by_document():
    def entry(doc_id, response, filt, match):
        return {"doc_id": doc_id, "resps": [[response]], "filter": filt, "exact_match": match}

    samples = []
    for doc_id, response, strict, flexible in (
        (0, "2 + 2 = 4\n#### 4", 1.0, 1.0),
        (1, "2 + 2 = 5\n#### 5", 0.0, 0.0),
        (2, "so the total is 4", 0.0, 1.0),  # right number, no marker: strict fails, flexible passes
        (3, "we add one and one and one and one " * 4, 0.0, 0.0),
        (4, "1 + 1 = 2\n#### 2\nOkay, but then 3 more", 1.0, 0.0),  # right after ####, wrong last number
    ):
        samples += [
            entry(doc_id, response, "strict-match", strict),
            entry(doc_id, response, "flexible-extract", flexible),
        ]
    out = summarize(samples)
    assert out["n"] == 5
    assert out["accuracy"] == 0.4 and out["strict_accuracy"] == 0.4
    assert out["buckets"] == {
        "correct": 2,
        "right, then kept talking": 1,
        "wrong answer": 1,
        "looping": 1,
        "no final answer": 0,
    }
    assert out["talks_past_answer"] == 0.2
    assert out["examples"]["wrong answer"] == ["2 + 2 = 5\n#### 5"]
