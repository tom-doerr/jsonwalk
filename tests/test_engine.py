import math
from dataclasses import replace

import pytest
from fake_lm import FakeLM
from jsonwalk.engine import RunConfig, run, sort_rows
from jsonwalk.score import BoolScore
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


def test_joint_is_value_probability_times_p_true():
    result = do_run()
    row = next(r for r in result.rows if r.value == "ab")
    p_true = 0.60 / (0.60 + 0.20)
    assert math.isclose(row.joint_prob, 0.45 * p_true, rel_tol=1e-9)


def test_joint_sort_reranks_a_likely_but_false_value():
    # "c" is the second most likely value but the model calls it false, so
    # ranking by "likely AND true" must drop it below the less likely "a".
    result = do_run()
    by_value = sort_rows(result.rows, "value")
    by_joint = sort_rows(result.rows, "joint")
    assert [r.value for r in by_value] == ["ab", "c", "a"]
    assert [r.value for r in by_joint] == ["ab", "a", "c"]


def test_delta_sort_differs_from_both():
    result = do_run()
    assert [r.value for r in sort_rows(result.rows, "delta")] == ["ab", "a", "c"]


def test_unknown_sort_mode_is_rejected():
    with pytest.raises(ValueError):
        sort_rows(do_run().rows, "alphabetical")


def test_joint_survives_a_confidently_false_verdict():
    # log(sigmoid(-800)) must not blow up; a plain log(p_true) would.
    score = BoolScore(logprob_true=-800.0, logprob_false=0.0, per_word={})
    assert score.p_true == 0.0
    assert math.isclose(score.log_p_true, -800.0, abs_tol=1e-6)


def test_values_echoed_from_the_preamble_are_flagged():
    lm = FakeLM(VOCAB, TABLE)
    cfg = RunConfig(
        field="n",
        bool_field="b",
        preamble="",
        walk=WalkConfig(k=10, max_tokens=6, child_top_k=10),
        true_words=(" true",),
        false_words=(" false",),
    )
    assert not any(r.echoed for r in run(lm, cfg).rows)

    # Same walk, but now "ab" also sits in the preamble: that is exactly the
    # "few-shot example echoed back as a candidate" case the flag exists for.
    shifted = FakeLM(VOCAB, {("ab", *state): dist for state, dist in TABLE.items()})
    flagged = {
        r.value: r.echoed for r in run(shifted, replace(cfg, preamble="ab")).rows
    }
    assert flagged["ab"] is True
    assert flagged["c"] is False


def test_bool_scoring_can_be_switched_off():
    lm = FakeLM(VOCAB, TABLE)
    cfg = RunConfig(field="n", bool_field="b", preamble="", score_bool=False)
    result = run(lm, cfg)
    assert all(r.bool_score is None for r in result.rows)
