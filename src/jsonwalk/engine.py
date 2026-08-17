"""Walk values, then judge each one. The whole pipeline in one call."""

from __future__ import annotations

import math
from collections.abc import Callable
# Aliased: this module has a dataclass attribute literally named "field",
# which would shadow dataclasses.field inside the class body.
from dataclasses import dataclass
from dataclasses import field as dc_field

from .lm import LanguageModel
from .logmath import NEG_INF
from .prompt import DEFAULT_PREAMBLE, Schema
from .score import (
    DEFAULT_FALSE_WORDS,
    DEFAULT_TRUE_WORDS,
    BoolScore,
    score_bools,
)
from .walk import ValueCandidate, WalkConfig, WalkStats, walk_values


@dataclass
class RunConfig:
    field: str = "startup_name"
    bool_field: str = "good_sounding_name"
    preamble: str = DEFAULT_PREAMBLE
    walk: WalkConfig = dc_field(default_factory=WalkConfig)
    true_words: tuple[str, ...] = DEFAULT_TRUE_WORDS
    false_words: tuple[str, ...] = DEFAULT_FALSE_WORDS
    score_bool: bool = True

    def schema(self) -> Schema:
        return Schema(
            field=self.field, bool_field=self.bool_field, preamble=self.preamble
        )


@dataclass(frozen=True)
class Row:
    """One enumerated value together with its verdict."""

    rank: int
    candidate: ValueCandidate
    share: float
    bool_score: BoolScore | None

    @property
    def value(self) -> str:
        return self.candidate.value

    def as_object(self, schema: Schema) -> dict[str, object]:
        """The JSON object this row stands for."""
        truth = self.bool_score.verdict if self.bool_score else False
        return schema.as_object(self.candidate.value, truth)

    def as_dict(self, schema: Schema) -> dict[str, object]:
        out: dict[str, object] = {
            "rank": self.rank,
            "object": self.as_object(schema),
            "value_logprob": self.candidate.logprob,
            "value_prob": self.candidate.prob,
            "share": self.share,
            "tokenizations": self.candidate.n_paths,
            "tokens": len(self.candidate.best_path),
        }
        if self.bool_score is not None:
            out["delta_true_false"] = self.bool_score.delta
            out["p_true"] = self.bool_score.p_true
            out["bool_mass"] = self.bool_score.bool_mass
            out["per_word_logprob"] = self.bool_score.per_word
        return out


@dataclass(frozen=True)
class RunResult:
    schema: Schema
    rows: tuple[Row, ...]
    stats: WalkStats
    value_prompt: str

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.schema.field,
            "bool_field": self.schema.bool_field,
            "prompt": self.value_prompt,
            "stats": self.stats.as_dict(),
            "rows": [r.as_dict(self.schema) for r in self.rows],
        }


def run(
    lm: LanguageModel,
    config: RunConfig | None = None,
    on_progress: Callable[[WalkStats], None] | None = None,
) -> RunResult:
    """Enumerate values for the string field and score the boolean field."""
    cfg = config or RunConfig()
    schema = cfg.schema()
    value_prompt = schema.value_prefix()

    candidates, stats = walk_values(
        lm, lm.encode(value_prompt), cfg.walk, on_progress=on_progress
    )

    scores: list[BoolScore | None] = [None] * len(candidates)
    if cfg.score_bool and candidates:
        scores = list(
            score_bools(
                lm,
                [schema.bool_prefix(c.raw) for c in candidates],
                true_words=cfg.true_words,
                false_words=cfg.false_words,
            )
        )

    total = math.exp(stats.found_logmass) if stats.found_logmass > NEG_INF else 0.0
    rows = tuple(
        Row(
            rank=i + 1,
            candidate=c,
            share=(c.prob / total) if total > 0 else 0.0,
            bool_score=s,
        )
        for i, (c, s) in enumerate(zip(candidates, scores))
    )
    return RunResult(
        schema=schema, rows=rows, stats=stats, value_prompt=value_prompt
    )
