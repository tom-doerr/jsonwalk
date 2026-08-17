"""The language-model interface the rest of jsonwalk talks to.

Deliberately narrow: the search needs exactly two things from a model,
"what are the likely next tokens after this sequence" and "how likely is
this specific continuation". Keeping it a Protocol means the search can be
tested against a deterministic fake model with no GPU involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# Lives here rather than in hf.py so that argument parsing can name the
# default model without importing torch. `jsonwalk --help` should not pay for
# a CUDA context.
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"


@dataclass(frozen=True)
class TopK:
    """Truncated next-token distribution at one position.

    ``kept_mass`` is the total probability of the tokens actually returned.
    It is reported rather than discarded so callers can tell the difference
    between "the model was confident" and "we threw away half the
    distribution to keep the search small".
    """

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    kept_mass: float

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError("token_ids and logprobs must have equal length")


class LanguageModel(Protocol):
    """Minimal causal-LM surface used by the walker and the bool scorer."""

    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` without adding special tokens."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a full token sequence to text.

        Must be safe on partial sequences: byte-level BPE tokens can split a
        UTF-8 codepoint, so callers always decode the whole sequence rather
        than concatenating per-token strings.
        """
        ...

    def top_next(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        top_k: int,
        top_p: float,
    ) -> list[TopK]:
        """Truncated next-token distributions, one per input sequence."""
        ...

    def sequence_logprobs(
        self,
        items: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        """Total log P(continuation | prefix) for each ``(prefix, continuation)``."""
        ...
