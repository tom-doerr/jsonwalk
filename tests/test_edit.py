import math

import pytest
from fake_lm import FakeLM
from jsonwalk.edit import Edit, EditConfig, EditSchema, propose_edits
from jsonwalk.score import BoolScore

DOC = "please use the old widget here"


def make_edit(search="the old", replace="the new", delta=1.0, p=0.5):
    return Edit(
        search=search,
        replace=replace,
        search_logprob=math.log(p),
        replace_logprob=0.0,
        occurrences=1,
        bool_score=BoolScore(logprob_true=delta, logprob_false=0.0, per_word={}),
    )


def test_edit_probability_is_the_two_stages_combined():
    e = Edit("a", "b", math.log(0.5), math.log(0.2), 1)
    assert math.isclose(e.prob, 0.10, rel_tol=1e-9)


def test_signal_carries_the_sign_of_the_verdict():
    assert make_edit(delta=+1.0).signal > 0
    assert make_edit(delta=-1.0).signal < 0


def test_applying_an_edit_replaces_one_occurrence():
    assert make_edit().apply_to(DOC) == "please use the new widget here"


def test_applying_an_edit_that_does_not_match_raises():
    # Silently doing nothing would let a bad proposal look like a success.
    with pytest.raises(ValueError):
        make_edit(search="absent text").apply_to(DOC)


def test_schema_prompts_nest_correctly():
    s = EditSchema(document=DOC, objective="Be brief.", preamble="")
    assert s.search_prefix() == '{"search": "'
    assert s.replace_prefix("the old") == '{"search": "the old", "replace": "'
    assert s.bool_prefix("the old", "the new") == (
        '{"search": "the old", "replace": "the new", "improves_objective":'
    )


def test_document_braces_do_not_break_templating():
    # str.format would explode on a document containing JSON or code.
    s = EditSchema(document='{"a": 1}', objective="x")
    assert '{"a": 1}' in s.rendered_preamble()


# End-to-end over a toy vocabulary. "zzzzzzz" is the most likely anchor and is
# not in the document, which is the whole point: it must be unreachable.
S = '{"search": "'
R = '", "replace": "'
B = '", "improves_objective":'
VOCAB = [S, R, B, "the old", "widget", "zzzzzzz", "the new", '"', " true", " false"]
TABLE = {
    (S,): {"zzzzzzz": 0.5, "the old": 0.3, "widget": 0.2},
    (S, "zzzzzzz"): {'"': 1.0},
    (S, "the old"): {'"': 1.0},
    (S, "widget"): {'"': 1.0},
    (S, "the old", R): {"the new": 0.8, "the old": 0.2},
    (S, "the old", R, "the new"): {'"': 1.0},
    (S, "the old", R, "the old"): {'"': 1.0},
    (S, "widget", R): {"the new": 1.0},
    (S, "widget", R, "the new"): {'"': 1.0},
    (S, "the old", R, "the new", B): {" true": 0.9, " false": 0.1},
    (S, "widget", R, "the new", B): {" true": 0.2, " false": 0.8},
    # only reached when no-ops are kept
    (S, "the old", R, "the old", B): {" true": 0.5, " false": 0.5},
}


def run_toy(**kw):
    lm = FakeLM(VOCAB, TABLE)
    schema = EditSchema(document=DOC, objective="Be brief.", preamble="")
    cfg = EditConfig(
        n_search=5,
        n_replace=3,
        true_words=(" true",),
        false_words=(" false",),
        **kw,
    )
    return propose_edits(lm, schema, cfg)


def test_the_likeliest_anchor_is_unreachable_when_it_is_not_in_the_document():
    proposals = run_toy()
    assert "zzzzzzz" not in {e.search for e in proposals.edits}
    assert all(e.search in DOC for e in proposals.edits)
    assert proposals.search_stats.rejected_paths >= 1


def test_noop_edits_are_dropped():
    assert all(not e.is_noop for e in run_toy().edits)
    assert any(e.is_noop for e in run_toy(drop_noops=False).edits)


def test_edits_rank_by_signed_verdict_then_likelihood():
    # "the old"->"the new" is likelier and judged true; "widget" is judged
    # false and must sort last however plausible its text.
    ranked = run_toy().ranked("signal")
    assert (ranked[0].search, ranked[0].replace) == ("the old", "the new")
    assert ranked[-1].search == "widget"
    assert ranked[-1].signal < 0


def test_every_proposed_edit_actually_applies():
    for e in run_toy().edits:
        assert e.apply_to(DOC) != DOC
