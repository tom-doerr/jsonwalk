import itertools
import math
import random

from fake_lm import FakeLM
from jsonwalk.prompt import first_unescaped_quote, unescape
from jsonwalk.walk import WalkConfig, walk_values

# "ab" is reachable two ways: as the single token 'ab', or as 'a'+'b'.
# Neither path alone beats "c", but together they do -- so a searcher that
# ranks token sequences instead of strings gets this table wrong.
VOCAB = ["#", "a", "b", "ab", "c", '"', '",', "\n", "<eos>"]
TABLE = {
    ("#",): {"ab": 0.20, "a": 0.50, "c": 0.30},
    ("#", "ab"): {'"': 1.0},
    ("#", "a"): {"b": 0.50, '"': 0.50},
    ("#", "a", "b"): {'"': 1.0},
    ("#", "c"): {'",': 1.0},
}


def make_lm():
    return FakeLM(VOCAB, TABLE, eos_token="<eos>")


def run(k=10, **kw):
    lm = make_lm()
    cfg = WalkConfig(k=k, max_tokens=8, child_top_k=10, **kw)
    return walk_values(lm, lm.encode("#"), cfg)


def test_merges_tokenizations_of_the_same_string():
    cands, _ = run()
    by_value = {c.value: c for c in cands}
    assert by_value["ab"].n_paths == 2
    assert math.isclose(by_value["ab"].prob, 0.45, abs_tol=1e-9)


def test_merging_changes_the_ranking():
    # Best single path for "ab" is 0.25, below "c" at 0.30. Merged it is 0.45
    # and wins. This ordering is the whole point of walking the tree.
    cands, _ = run()
    assert [c.value for c in cands] == ["ab", "c", "a"]
    assert math.isclose(max(TABLE[("#", "a")].values()) * 0.5, 0.25)


def test_ranking_survives_a_tight_k():
    # k=1 would let the exact per-path bound stop before the second
    # tokenization of "ab" is reached; stop_margin is what prevents that.
    cands, _ = run(k=1)
    assert cands[0].value == "ab"
    assert cands[0].n_paths == 2


def test_quote_inside_a_token_terminates_the_value():
    # "c" is closed by the single token '",' -- the comma must not leak in.
    cands, _ = run()
    assert [c.value for c in cands if c.value.startswith("c")] == ["c"]


def test_probabilities_sum_to_one_when_exhausted():
    cands, stats = run()
    assert stats.exhausted
    assert math.isclose(sum(c.prob for c in cands), 1.0, abs_tol=1e-9)
    assert stats.distinct_values == 3


def test_escaped_quote_does_not_close_the_string():
    vocab = ["#", "q", '\\"', '"']
    table = {
        ("#",): {"q": 1.0},
        ("#", "q"): {'\\"': 1.0},
        ("#", "q", '\\"'): {'"': 1.0},
    }
    lm = FakeLM(vocab, table)
    cands, _ = walk_values(lm, lm.encode("#"), WalkConfig(k=5, max_tokens=6))
    assert [c.value for c in cands] == ['q"']
    assert cands[0].raw == 'q\\"'


def test_unterminated_paths_are_counted_not_silently_dropped():
    vocab = ["#", "a", '"']
    table = {("#",) + ("a",) * d: {"a": 1.0} for d in range(6)}
    lm = FakeLM(vocab, table)
    cands, stats = walk_values(lm, lm.encode("#"), WalkConfig(k=5, max_tokens=3))
    assert cands == []
    assert stats.truncated_paths == 1


def test_control_character_is_an_invalid_branch():
    vocab = ["#", "\n", '"']
    table = {("#",): {"\n": 1.0}}
    lm = FakeLM(vocab, table)
    cands, stats = walk_values(lm, lm.encode("#"), WalkConfig(k=5))
    assert cands == []
    assert stats.invalid_paths == 1


def _brute_force(table, max_tokens):
    """Enumerate every complete path exhaustively, for cross-checking."""
    totals: dict[str, float] = {}

    def rec(state, gen, prob):
        for tok, p in table.get(state, {}).items():
            text = "".join(gen + (tok,))
            quote = first_unescaped_quote(text)
            if quote is not None:
                value = unescape(text[:quote])
                totals[value] = totals.get(value, 0.0) + prob * p
            elif len(gen) + 1 < max_tokens:
                rec(state + (tok,), gen + (tok,), prob * p)

    rec(("#",), (), 1.0)
    return totals


def test_matches_exhaustive_enumeration_on_a_wider_table():
    letters = ["a", "b", "ab", "ba", "c"]
    vocab = ["#", *letters, '"', '",']
    rng = random.Random(20260817)
    table = {}
    for depth in range(3):
        for combo in itertools.product(letters, repeat=depth):
            state = ("#", *combo)
            if depth == 2:
                table[state] = {'"': 1.0}
                continue
            weights = {t: rng.random() + 0.05 for t in [*letters, '"', '",']}
            total = sum(weights.values())
            table[state] = {t: w / total for t, w in weights.items()}

    lm = FakeLM(vocab, table)
    cands, stats = walk_values(
        lm, lm.encode("#"), WalkConfig(k=50, max_tokens=4, child_top_k=20)
    )
    expected = _brute_force(table, max_tokens=4)

    assert stats.exhausted
    assert {c.value for c in cands} == set(expected)
    for c in cands:
        assert math.isclose(c.prob, expected[c.value], rel_tol=1e-9)
    assert [c.value for c in cands] == sorted(
        expected, key=lambda v: (-expected[v], v)
    )


def test_eos_is_an_invalid_branch():
    vocab = ["#", "<eos>", '"']
    table = {("#",): {"<eos>": 1.0}}
    lm = FakeLM(vocab, table, eos_token="<eos>")
    cands, stats = walk_values(lm, lm.encode("#"), WalkConfig(k=5))
    assert cands == []
    assert stats.invalid_paths == 1
