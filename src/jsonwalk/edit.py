"""Propose edits to a text artifact as search/replace/verdict triples.

The object is::

    {"search": "<verbatim from the document>", "replace": "<new text>",
     "improves_objective": true}

Three ideas from the rest of jsonwalk combine here:

* ``search`` is walked under a :class:`~jsonwalk.constraints.SubstringOf`
  constraint, so the anchor is guaranteed to exist in the document and to be
  unique. The usual failure of model-written search/replace -- an anchor that
  is not in the file -- is not filtered out, it is unreachable.
* ``replace`` is walked normally, conditioned on the chosen ``search``.
* the boolean is scored by its true/false logprob delta against a stated
  objective, and edits are ranked by ``P(edit) x delta`` so that an edit the
  model rejects sinks in proportion to how obvious it was.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from dataclasses import field as dc_field

from .constraints import SubstringOf
from .lm import LanguageModel
from .score import (
    DEFAULT_FALSE_WORDS,
    DEFAULT_TRUE_WORDS,
    BoolScore,
    score_bools,
)
from .walk import WalkConfig, walk_values

# The worked example is about a different document on purpose -- it teaches
# the record shape and gets the boolean slot filled, and it cannot pollute the
# search anchors because those are constrained to the real document.
DEFAULT_EDIT_PREAMBLE = """\
Edits proposed by a careful editor, one JSON object per line.
"search" is copied verbatim from the document; "replace" is what it becomes.

# Example, for a different document
{"search": "in the event that", "replace": "if", "{bool_field}": true}
{"search": "the", "replace": "teh", "{bool_field}": false}

# Document
{document}

# Objective
{objective}

# Edits
"""


@dataclass(frozen=True)
class EditSchema:
    """The document, the goal, and the shape of an edit record."""

    document: str
    objective: str = "Make the text clearer and more concise."
    search_field: str = "search"
    replace_field: str = "replace"
    bool_field: str = "improves_objective"
    preamble: str = DEFAULT_EDIT_PREAMBLE

    def rendered_preamble(self) -> str:
        # Manual replacement, not str.format: the template is full of literal
        # JSON braces and the document may contain any braces at all.
        return (
            self.preamble.replace("{bool_field}", self.bool_field)
            .replace("{document}", self.document)
            .replace("{objective}", self.objective)
        )

    def search_prefix(self) -> str:
        return f"{self.rendered_preamble()}{{{json.dumps(self.search_field)}: \""

    def replace_prefix(self, search_raw: str) -> str:
        return (
            f'{self.search_prefix()}{search_raw}", '
            f'{json.dumps(self.replace_field)}: "'
        )

    def bool_prefix(self, search_raw: str, replace_raw: str) -> str:
        return (
            f'{self.replace_prefix(search_raw)}{replace_raw}", '
            f"{json.dumps(self.bool_field)}:"
        )


@dataclass(frozen=True)
class Edit:
    """One proposed search/replace, with the model's verdict on it."""

    search: str
    replace: str
    search_logprob: float
    replace_logprob: float
    occurrences: int
    #: as the model wrote them, still JSON-escaped -- kept so the scoring
    #: prompt is rebuilt from the exact tokens it was conditioned on
    search_raw: str = ""
    replace_raw: str = ""
    bool_score: BoolScore | None = None

    @property
    def logprob(self) -> float:
        """log P(search) + log P(replace | search)."""
        return self.search_logprob + self.replace_logprob

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    @property
    def signal(self) -> float:
        """``P(edit) * D(T-F)`` -- same signed ranking as the main tool."""
        if self.bool_score is None:
            raise ValueError("signal needs a boolean score")
        return self.prob * self.bool_score.delta

    @property
    def is_noop(self) -> bool:
        return self.search == self.replace

    def apply_to(self, document: str) -> str:
        """Apply this edit once. Raises rather than guessing."""
        if self.search not in document:
            raise ValueError(f"search text is not in the document: {self.search!r}")
        return document.replace(self.search, self.replace, 1)


@dataclass
class EditConfig:
    """How wide to search at each of the two stages."""

    n_search: int = 8
    n_replace: int = 4
    min_anchor: int = 6
    require_unique: bool = True
    drop_noops: bool = True
    search_walk: WalkConfig = dc_field(
        default_factory=lambda: WalkConfig(max_tokens=24, max_expansions=6000)
    )
    replace_walk: WalkConfig = dc_field(
        default_factory=lambda: WalkConfig(max_tokens=24, max_expansions=2000)
    )
    true_words: tuple[str, ...] = DEFAULT_TRUE_WORDS
    false_words: tuple[str, ...] = DEFAULT_FALSE_WORDS


@dataclass
class EditProposals:
    schema: EditSchema
    edits: tuple[Edit, ...]
    search_stats: object
    n_anchors: int

    def ranked(self, mode: str = "signal") -> list[Edit]:
        if mode == "signal":
            return sorted(self.edits, key=lambda e: -e.signal)
        if mode == "delta":
            return sorted(self.edits, key=lambda e: -e.bool_score.delta)
        if mode == "value":
            return sorted(self.edits, key=lambda e: -e.logprob)
        raise ValueError(f"unknown mode {mode!r}")


def propose_edits(
    lm: LanguageModel,
    schema: EditSchema,
    config: EditConfig | None = None,
) -> EditProposals:
    """Walk anchors, then replacements, then judge each pair."""
    cfg = config or EditConfig()
    constraint = SubstringOf(
        schema.document,
        min_length=cfg.min_anchor,
        require_unique=cfg.require_unique,
    )

    search_cfg = replace(cfg.search_walk, k=cfg.n_search)
    anchors, search_stats = walk_values(
        lm,
        lm.encode(schema.search_prefix()),
        search_cfg,
        constraint=constraint,
    )

    edits: list[Edit] = []
    for anchor in anchors:
        replacements, _ = walk_values(
            lm,
            lm.encode(schema.replace_prefix(anchor.raw)),
            replace(cfg.replace_walk, k=cfg.n_replace),
        )
        for rep in replacements:
            if cfg.drop_noops and rep.value == anchor.value:
                continue
            edits.append(
                Edit(
                    search=anchor.value,
                    replace=rep.value,
                    search_logprob=anchor.logprob,
                    replace_logprob=rep.logprob,
                    occurrences=constraint.occurrences(anchor.value),
                    search_raw=anchor.raw,
                    replace_raw=rep.raw,
                )
            )

    if edits:
        scores = score_bools(
            lm,
            [schema.bool_prefix(e.search_raw, e.replace_raw) for e in edits],
            true_words=cfg.true_words,
            false_words=cfg.false_words,
        )
        edits = [replace(e, bool_score=s) for e, s in zip(edits, scores)]

    return EditProposals(
        schema=schema,
        edits=tuple(edits),
        search_stats=search_stats,
        n_anchors=len(anchors),
    )
