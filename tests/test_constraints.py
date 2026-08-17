import pytest
from fake_lm import FakeLM
from jsonwalk.constraints import SubstringOf
from jsonwalk.walk import WalkConfig, walk_values

TEXT = "the quick brown fox jumps over the lazy dog"


def test_partials_are_allowed_while_they_remain_substrings():
    c = SubstringOf(TEXT, min_length=3)
    assert c.allows("")  # empty must not prune the root
    assert c.allows("quick br")
    assert not c.allows("quick cat")


def test_acceptance_requires_the_minimum_length():
    c = SubstringOf(TEXT, min_length=5, require_unique=False)
    assert not c.accepts("the")
    assert c.accepts("quick")


def test_uniqueness_rejects_an_ambiguous_anchor():
    # "the" appears twice, so replacing it would be a coin flip.
    unique = SubstringOf(TEXT, min_length=3)
    assert unique.occurrences("the") == 2
    assert not unique.accepts("the")
    assert unique.accepts("quick")

    ambiguous_ok = SubstringOf(TEXT, min_length=3, require_unique=False)
    assert ambiguous_ok.accepts("the")


def test_occurrences_counts_overlaps():
    assert SubstringOf("aaa").occurrences("aa") == 2


def test_empty_text_is_rejected_at_construction():
    with pytest.raises(ValueError):
        SubstringOf("")


# "cat" is more likely than "dog" but is not in the document, so an
# unconstrained walk returns it and a constrained walk cannot.
VOCAB = ["#", "ca", "do", "t", "g", '"']
TABLE = {
    ("#",): {"ca": 0.7, "do": 0.3},
    ("#", "ca"): {"t": 1.0},
    ("#", "ca", "t"): {'"': 1.0},
    ("#", "do"): {"g": 1.0},
    ("#", "do", "g"): {'"': 1.0},
}


def test_the_walk_can_only_produce_text_that_exists():
    lm = FakeLM(VOCAB, TABLE)
    cfg = WalkConfig(k=5, max_tokens=6, child_top_k=10)

    free, _ = walk_values(lm, lm.encode("#"), cfg)
    assert [c.value for c in free] == ["cat", "dog"]

    bound, stats = walk_values(
        lm,
        lm.encode("#"),
        cfg,
        constraint=SubstringOf("a dog barks", min_length=3),
    )
    assert [c.value for c in bound] == ["dog"]
    assert stats.rejected_paths >= 1, "the pruning is counted, not hidden"


def test_a_branch_dies_as_soon_as_it_leaves_the_text():
    # "cat" is pruned at "ca", before the full value is ever built: that is
    # what makes the constraint cheap rather than a post-filter.
    lm = FakeLM(VOCAB, TABLE)
    walk_values(
        lm,
        lm.encode("#"),
        WalkConfig(k=5, max_tokens=6, child_top_k=10),
        constraint=SubstringOf("a dog barks", min_length=3),
    )
    expanded = {lm.decode(seq[1:]) for seq in lm.seen_sequences}
    assert "ca" not in expanded and "cat" not in expanded
