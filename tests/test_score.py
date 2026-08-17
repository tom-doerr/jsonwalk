import math

import pytest
from fake_lm import FakeLM
from jsonwalk.prompt import Schema
from jsonwalk.score import TokenBoundaryError, score_bools

VOCAB = ['{"n": "', "Acme", '", "b":', " true", " True", " false", " False", '"']
STATE = ('{"n": "', "Acme", '", "b":')
TABLE = {
    STATE: {" true": 0.5, " True": 0.1, " false": 0.2, " False": 0.05, '"': 0.15},
}


def test_spellings_are_pooled_within_each_side():
    lm = FakeLM(VOCAB, TABLE)
    prefix = Schema(field="n", bool_field="b", preamble="").bool_prefix("Acme")
    (score,) = score_bools(lm, [prefix])

    assert math.isclose(score.logprob_true, math.log(0.6), abs_tol=1e-12)
    assert math.isclose(score.logprob_false, math.log(0.25), abs_tol=1e-12)
    assert math.isclose(score.delta, math.log(0.6 / 0.25), abs_tol=1e-12)
    assert math.isclose(score.p_true, 0.6 / 0.85, abs_tol=1e-12)
    assert score.verdict is True


def test_bool_mass_reveals_a_prompt_the_model_reads_as_non_boolean():
    # 0.15 of the mass here goes to a quote; bool_mass is what tells you the
    # delta is a ratio between two things the model half-wanted to say.
    lm = FakeLM(VOCAB, TABLE)
    prefix = Schema(field="n", bool_field="b", preamble="").bool_prefix("Acme")
    (score,) = score_bools(lm, [prefix])
    assert math.isclose(score.bool_mass, 0.85, abs_tol=1e-12)


def test_per_word_logprobs_are_reported_unpooled():
    lm = FakeLM(VOCAB, TABLE)
    prefix = Schema(field="n", bool_field="b", preamble="").bool_prefix("Acme")
    (score,) = score_bools(lm, [prefix])
    assert set(score.per_word) == {"true", "True", "false", "False"}
    assert math.isclose(score.per_word["true"], math.log(0.5), abs_tol=1e-12)


def test_resegmented_boundary_is_an_error_not_a_wrong_number():
    # A vocabulary where "prompt + word" merges across the seam: the prompt is
    # no longer a token-prefix, so the conditional probability would be
    # measured against a different context. That must fail loudly.
    vocab = ['{"n": "Acme", "b":', " true", '{"n": "Acme", "b": true']
    lm = FakeLM(vocab, {})
    with pytest.raises(TokenBoundaryError):
        score_bools(lm, ['{"n": "Acme", "b":'], true_words=[" true"],
                    false_words=[" true"])


def test_empty_word_group_is_rejected():
    lm = FakeLM(VOCAB, TABLE)
    with pytest.raises(ValueError):
        score_bools(lm, ['{"n": "Acme", "b":'], true_words=[], false_words=[" false"])
