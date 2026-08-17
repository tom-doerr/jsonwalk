"""Score the boolean field by the true-vs-false logprob delta.

Rather than *sampling* a boolean, we read the model's belief straight off the
distribution at the slot where the literal must appear. Two numbers matter:

``delta``
    ``log P(true...) - log P(false...)``. A log-odds: +2.3 means the model is
    ~10x more willing to write true than false. This is the judgement.

``bool_mass``
    ``P(true...) + P(false...)`` in absolute terms. This is the sanity check.
    A base model given a bare prompt may want to write a *number* in that slot
    -- then the delta is a ratio between two things it never intended to say.
    Low bool_mass means "fix your preamble", not "the answer is uncertain".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .lm import LanguageModel
from .logmath import logsumexp, sigmoid

# The leading space belongs to the *word*, not the prompt. If the prompt ended
# with ": " the tokenizer would merge that space into " true" when it encoded
# the full text, and the prompt would no longer be a token-prefix of it -- the
# classic tokenization-boundary bug that quietly deflates continuation
# probabilities. Cutting the prompt at the colon keeps the boundary clean.
DEFAULT_TRUE_WORDS: tuple[str, ...] = (" true", " True")
DEFAULT_FALSE_WORDS: tuple[str, ...] = (" false", " False")


class TokenBoundaryError(RuntimeError):
    """The prompt is not a token-prefix of prompt+word for some word."""


@dataclass(frozen=True)
class BoolScore:
    """Model belief about the boolean field for one candidate value."""

    logprob_true: float
    logprob_false: float
    per_word: dict[str, float]

    @property
    def delta(self) -> float:
        """Log-odds of true over false. The headline number."""
        return self.logprob_true - self.logprob_false

    @property
    def p_true(self) -> float:
        """Probability of true *given* the model writes a boolean at all."""
        return sigmoid(self.delta)

    @property
    def bool_mass(self) -> float:
        """Absolute probability the slot is filled with any of the literals."""
        return math.exp(logsumexp((self.logprob_true, self.logprob_false)))

    @property
    def verdict(self) -> bool:
        return self.delta > 0.0


def score_bools(
    lm: LanguageModel,
    prefixes: Sequence[str],
    *,
    true_words: Sequence[str] = DEFAULT_TRUE_WORDS,
    false_words: Sequence[str] = DEFAULT_FALSE_WORDS,
) -> list[BoolScore]:
    """Score every prompt in ``prefixes`` in one batched pass.

    Each prompt must end exactly where the boolean literal goes (the colon of
    the bool field). Spellings within a group are combined with logsumexp, so
    ``true`` and ``True`` are treated as two ways of saying the same thing.
    """
    words = tuple(true_words) + tuple(false_words)
    if not true_words or not false_words:
        raise ValueError("need at least one spelling on each side")

    items: list[tuple[list[int], list[int]]] = []
    for prefix_text in prefixes:
        prefix_ids = lm.encode(prefix_text)
        for word in words:
            full = lm.encode(prefix_text + word)
            if full[: len(prefix_ids)] != prefix_ids:
                raise TokenBoundaryError(
                    f"tokenizer re-segmented the prompt boundary for {word!r}; "
                    "the prompt must end at the colon, not after a space"
                )
            cont = full[len(prefix_ids) :]
            if not cont:
                raise TokenBoundaryError(f"{word!r} encodes to no tokens")
            items.append((prefix_ids, cont))

    flat = lm.sequence_logprobs(items)
    n_true = len(true_words)
    scores: list[BoolScore] = []
    for i in range(len(prefixes)):
        chunk = flat[i * len(words) : (i + 1) * len(words)]
        per_word = {w.strip(): lp for w, lp in zip(words, chunk)}
        scores.append(
            BoolScore(
                logprob_true=logsumexp(chunk[:n_true]),
                logprob_false=logsumexp(chunk[n_true:]),
                per_word=per_word,
            )
        )
    return scores
