"""Transformers backend.

Two constraints shape this file.

**Vocabulary size.** Qwen3.5 has ~248k tokens, so one position of fp32 logits
is ~1 MB. Letting transformers return logits for every position of a batch of
long prompts would cost gigabytes, so every forward here passes
``logits_to_keep`` and asks only for the positions actually needed.

**No padding.** ``logits_to_keep`` counts from the *end* of the tensor, which
is the wrong place for a right-padded short row. Instead sequences are bucketed
by length and each bucket is stacked unpadded. Batches from the walker share a
prompt and differ by a token or two, so the buckets stay large.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lm import DEFAULT_MODEL, TopK

__all__ = ["DEFAULT_MODEL", "HFLanguageModel"]


def _buckets(sequences: Sequence[Sequence[int]], key) -> dict:
    groups: dict = {}
    for i, seq in enumerate(sequences):
        groups.setdefault(key(seq), []).append(i)
    return groups


def _chunks(items: list[int], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class HFLanguageModel:
    """A :class:`jsonwalk.lm.LanguageModel` backed by transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 24,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    @torch.no_grad()
    def top_next(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        top_k: int,
        top_p: float,
    ) -> list[TopK]:
        results: list[TopK | None] = [None] * len(sequences)
        for _, idxs in _buckets(sequences, len).items():
            for chunk in _chunks(idxs, self.batch_size):
                ids = torch.tensor(
                    [list(sequences[i]) for i in chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self.model(input_ids=ids, logits_to_keep=1).logits[:, -1, :]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                k = min(top_k, logprobs.shape[-1])
                vals, inds = torch.topk(logprobs, k, dim=-1)
                probs = vals.exp()
                # topk is sorted descending, so "mass strictly before this
                # token < top_p" is a prefix mask and always keeps at least one.
                before = probs.cumsum(dim=-1) - probs
                kept = (before < top_p).sum(dim=-1).clamp(min=1)
                for row, i in enumerate(chunk):
                    n = int(kept[row].item())
                    results[i] = TopK(
                        token_ids=tuple(inds[row, :n].tolist()),
                        logprobs=tuple(vals[row, :n].tolist()),
                        kept_mass=float(probs[row, :n].sum().item()),
                    )
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:  # pragma: no cover - defensive
            raise RuntimeError(f"no distribution computed for rows {missing}")
        return results  # type: ignore[return-value]

    @torch.no_grad()
    def sequence_logprobs(
        self,
        items: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[float]:
        results = [float("nan")] * len(items)
        groups: dict[tuple[int, int], list[int]] = {}
        for i, (prefix, cont) in enumerate(items):
            if not prefix:
                raise ValueError("prefix must contain at least one token")
            if not cont:
                raise ValueError("continuation must contain at least one token")
            groups.setdefault((len(prefix), len(cont)), []).append(i)

        for (plen, clen), idxs in groups.items():
            for chunk in _chunks(idxs, self.batch_size):
                full = torch.tensor(
                    [list(items[i][0]) + list(items[i][1]) for i in chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                # Keeping clen+1 positions puts the slot that predicts the
                # first continuation token at kept index 0.
                logits = self.model(input_ids=full, logits_to_keep=clen + 1).logits
                lp = torch.log_softmax(logits[:, :clen, :].float(), dim=-1)
                targets = full[:, plen:].unsqueeze(-1)
                per_token = lp.gather(-1, targets).squeeze(-1)
                for row, i in enumerate(chunk):
                    results[i] = float(per_token[row].sum().item())
        return results
