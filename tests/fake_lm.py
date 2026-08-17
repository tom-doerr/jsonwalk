"""A deterministic toy language model.

The search logic is the interesting part of jsonwalk, and it should be
testable without a GPU, a download, or a 248k-token vocabulary. FakeLM is a
handful of string "tokens" plus an explicit probability table keyed by the
exact token sequence so far -- which makes it possible to write a vocabulary
where one string has several tokenizations and assert what the walker does
about it.
"""

from __future__ import annotations

import math

from jsonwalk.lm import TopK


class FakeLM:
    def __init__(
        self,
        vocab: list[str],
        table: dict[tuple[str, ...], dict[str, float]],
        eos_token: str | None = None,
    ) -> None:
        self.vocab = list(vocab)
        self.ids = {tok: i for i, tok in enumerate(self.vocab)}
        self.table = {k: dict(v) for k, v in table.items()}
        self.eos_token_id = self.ids[eos_token] if eos_token is not None else None
        self._longest_first = sorted(self.vocab, key=len, reverse=True)
        self.forward_calls = 0
        self.rows_forwarded = 0

    # -- tokenizer -------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Greedy longest-match tokenization (a stand-in for BPE merges)."""
        out: list[int] = []
        i = 0
        while i < len(text):
            for tok in self._longest_first:
                if text.startswith(tok, i):
                    out.append(self.ids[tok])
                    i += len(tok)
                    break
            else:
                raise ValueError(f"cannot tokenize {text[i:]!r}")
        return out

    def decode(self, token_ids) -> str:
        return "".join(self.vocab[i] for i in token_ids)

    def _key(self, token_ids) -> tuple[str, ...]:
        return tuple(self.vocab[i] for i in token_ids)

    def _dist(self, token_ids) -> dict[str, float]:
        key = self._key(token_ids)
        if key not in self.table:
            raise KeyError(f"FakeLM has no distribution for state {key}")
        return self.table[key]

    # -- LanguageModel ---------------------------------------------------
    def top_next(self, sequences, *, top_k: int, top_p: float) -> list[TopK]:
        self.forward_calls += 1
        self.rows_forwarded += len(sequences)
        results = []
        for seq in sequences:
            ranked = sorted(self._dist(seq).items(), key=lambda kv: (-kv[1], kv[0]))
            ranked = ranked[:top_k]
            before = 0.0
            kept = []
            for tok, prob in ranked:
                if before >= top_p and kept:
                    break
                kept.append((tok, prob))
                before += prob
            results.append(
                TopK(
                    token_ids=tuple(self.ids[t] for t, _ in kept),
                    logprobs=tuple(math.log(p) for _, p in kept),
                    kept_mass=sum(p for _, p in kept),
                )
            )
        return results

    def sequence_logprobs(self, items) -> list[float]:
        out = []
        for prefix, cont in items:
            state = list(prefix)
            total = 0.0
            for tid in cont:
                prob = self._dist(state).get(self.vocab[tid], 0.0)
                if prob <= 0.0:
                    total = float("-inf")
                    break
                total += math.log(prob)
                state.append(tid)
            out.append(total)
        return out
