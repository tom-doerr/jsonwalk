import math

from fake_lm import FakeLM
from jsonwalk.engine import RunConfig, run
from jsonwalk.walk import WalkConfig

VOCAB = ['{"n": "', "a", "b", "ab", "c", '"', '", "b":', " true", " false"]
P = '{"n": "'
TABLE = {
    # value enumeration
    (P,): {"ab": 0.20, "a": 0.50, "c": 0.30},
    (P, "ab"): {'"': 1.0},
    (P, "a"): {"b": 0.50, '"': 0.50},
    (P, "a", "b"): {'"': 1.0},
    (P, "c"): {'"': 1.0},
    # boolean slot, reached via the canonical re-encoding of each value
    (P, "ab", '", "b":'): {" true": 0.60, " false": 0.20},
    (P, "a", '", "b":'): {" true": 0.30, " false": 0.30},
    (P, "c", '", "b":'): {" true": 0.10, " false": 0.70},
}


def do_run():
    lm = FakeLM(VOCAB, TABLE)
    cfg = RunConfig(
        field="n",
        bool_field="b",
        preamble="",
        walk=WalkConfig(k=10, max_tokens=6, child_top_k=10),
        # This toy vocabulary has no capitalised spellings; pooling of
        # true/True is covered in test_score.py.
        true_words=(" true",),
        false_words=(" false",),
    )
    return run(lm, cfg)


def test_rows_are_ranked_by_merged_value_probability():
    result = do_run()
    assert [r.value for r in result.rows] == ["ab", "c", "a"]
    assert [r.rank for r in result.rows] == [1, 2, 3]


def test_shares_are_normalised_over_what_was_found():
    result = do_run()
    assert math.isclose(sum(r.share for r in result.rows), 1.0, abs_tol=1e-9)


def test_each_row_carries_its_own_verdict():
    result = do_run()
    verdicts = {r.value: r.bool_score.verdict for r in result.rows}
    assert verdicts == {"ab": True, "c": False, "a": False}


def test_a_tie_between_true_and_false_is_not_a_yes():
    # "a" sits at exactly 0.30 vs 0.30: delta 0, and verdict is strict.
    result = do_run()
    row = next(r for r in result.rows if r.value == "a")
    assert math.isclose(row.bool_score.delta, 0.0, abs_tol=1e-12)
    assert row.bool_score.verdict is False


def test_row_renders_the_json_object_it_stands_for():
    result = do_run()
    assert result.rows[0].as_object(result.schema) == {"n": "ab", "b": True}
    assert result.rows[1].as_object(result.schema) == {"n": "c", "b": False}


def test_result_serialises_to_plain_data():
    payload = do_run().as_dict()
    assert payload["field"] == "n"
    assert payload["prompt"] == '{"n": "'
    assert payload["rows"][0]["object"] == {"n": "ab", "b": True}
    assert payload["rows"][0]["tokenizations"] == 2


def test_bool_scoring_can_be_switched_off():
    lm = FakeLM(VOCAB, TABLE)
    cfg = RunConfig(field="n", bool_field="b", preamble="", score_bool=False)
    result = run(lm, cfg)
    assert all(r.bool_score is None for r in result.rows)
