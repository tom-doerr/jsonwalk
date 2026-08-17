"""Best-first walk of the completion tree for one JSON string value.

Why a tree walk and not sampling or beam search:

* Sampling gives you frequencies, not probabilities, and needs many draws to
  rank the tail.
* Beam search fixes the beam width per *step*, which silently favours values
  that happen to use fewer tokens.

Here the frontier is a priority queue ordered by cumulative logprob, so a
five-token value and a one-token value compete on the same axis -- the joint
probability of the whole string. And because every extra token can only
*lower* a path's logprob, the frontier's best score is an upper bound on
everything still unexplored: once it drops below the k-th best finished
candidate, the top-k is provably complete. That bound is reported as
``WalkStats.frontier_bound``.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .lm import LanguageModel
from .logmath import NEG_INF, logsumexp
from .prompt import InvalidJSONString, first_unescaped_quote, unescape

# JSON forbids raw control characters inside strings; if the model emits one
# it has left the string rather than closed it, and the branch is dead.
_CONTROL_CHARS = frozenset(chr(c) for c in range(0x20))


@dataclass(frozen=True)
class ValueCandidate:
    """One distinct string value, with every tokenization of it merged."""

    value: str
    raw: str
    logprob: float
    n_paths: int
    best_path: tuple[int, ...]

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


@dataclass
class WalkConfig:
    """Search knobs. Defaults are tuned for a 0.8B base model."""

    k: int = 20
    max_tokens: int = 16
    child_top_k: int = 40
    child_top_p: float = 0.9995
    batch_size: int = 24
    max_expansions: int = 4000
    min_logprob: float = -28.0
    # Keep exploring until an unexplored path could contribute at most this
    # fraction of the k-th best value. Stopping the instant the bound crosses
    # the k-th best is exact per *path*, but a value can be reached by several
    # tokenizations and the later ones arrive deeper in the tree -- that margin
    # is what lets them be found and merged. 1.0 disables it.
    stop_margin: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.stop_margin <= 1.0:
            raise ValueError("stop_margin must be in (0, 1]")


@dataclass
class WalkStats:
    """What the search did, including everything it threw away."""

    expansions: int = 0
    nodes_pushed: int = 0
    distinct_values: int = 0
    frontier_bound: float = NEG_INF
    exhausted: bool = False
    complete_top_k: bool = False
    truncated_paths: int = 0
    invalid_paths: int = 0
    found_logmass: float = NEG_INF

    def as_dict(self) -> dict[str, object]:
        return {
            "expansions": self.expansions,
            "nodes_pushed": self.nodes_pushed,
            "distinct_values": self.distinct_values,
            "frontier_bound": self.frontier_bound,
            "exhausted": self.exhausted,
            "complete_top_k": self.complete_top_k,
            "truncated_paths": self.truncated_paths,
            "invalid_paths": self.invalid_paths,
            "found_mass": math.exp(self.found_logmass),
        }


@dataclass
class _Merged:
    """All tokenizations that decode to the same string.

    The running total is accumulated incrementally: a flat distribution can
    leave thousands of distinct values in this dict, and the stopping check
    reads every total on every batch.
    """

    n_paths: int = 0
    total: float = NEG_INF
    best_logprob: float = NEG_INF
    best_raw: str = ""
    best_path: tuple[int, ...] = ()

    def add(self, logprob: float, raw: str, path: tuple[int, ...]) -> None:
        self.n_paths += 1
        self.total = logsumexp((self.total, logprob))
        if logprob > self.best_logprob:
            self.best_logprob = logprob
            self.best_raw = raw
            self.best_path = path


def _top_k_complete(
    merged: dict[str, _Merged], k: int, bound: float, margin: float
) -> bool:
    """True when no unexplored path can meaningfully change the top-k.

    Extending a path only lowers its logprob, so the frontier's best score is
    an upper bound on everything unexplored -- exact for single paths. The
    ``margin`` pushes the cut-off further down to catch the extra
    tokenizations that merge into values already found.
    """
    if len(merged) < k:
        return False
    kth = heapq.nlargest(k, (m.total for m in merged.values()))[-1]
    return bound <= kth + math.log(margin)


def walk_values(
    lm: LanguageModel,
    prefix_ids: Sequence[int],
    config: WalkConfig | None = None,
    on_progress: Callable[[WalkStats], None] | None = None,
) -> tuple[list[ValueCandidate], WalkStats]:
    """Enumerate the most likely values for the string field.

    ``prefix_ids`` must end at the value's opening quote (see
    :meth:`jsonwalk.prompt.Schema.value_prefix`). Returns candidates sorted by
    descending probability plus the stats of the search that produced them.
    """
    cfg = config or WalkConfig()
    prefix = tuple(prefix_ids)
    stats = WalkStats()
    merged: dict[str, _Merged] = {}
    frontier: list[tuple[float, int, tuple[int, ...]]] = [(0.0, 0, ())]
    stats.nodes_pushed = 1
    tiebreak = 1

    while frontier and stats.expansions < cfg.max_expansions:
        if _top_k_complete(merged, cfg.k, -frontier[0][0], cfg.stop_margin):
            stats.complete_top_k = True
            break

        batch: list[tuple[float, tuple[int, ...]]] = []
        while frontier and len(batch) < cfg.batch_size:
            neg_lp, _, tokens = heapq.heappop(frontier)
            batch.append((-neg_lp, tokens))

        tops = lm.top_next(
            [prefix + tokens for _, tokens in batch],
            top_k=cfg.child_top_k,
            top_p=cfg.child_top_p,
        )
        stats.expansions += len(batch)

        for (node_lp, tokens), top in zip(batch, tops):
            for tid, tlp in zip(top.token_ids, top.logprobs):
                child_lp = node_lp + tlp
                if child_lp < cfg.min_logprob:
                    continue
                if lm.eos_token_id is not None and tid == lm.eos_token_id:
                    stats.invalid_paths += 1
                    continue

                child = tokens + (tid,)
                text = lm.decode(child)
                quote = first_unescaped_quote(text)

                if quote is not None:
                    raw = text[:quote]
                    try:
                        value = unescape(raw)
                    except InvalidJSONString:
                        stats.invalid_paths += 1
                        continue
                    merged.setdefault(value, _Merged()).add(child_lp, raw, child)
                    continue

                if _CONTROL_CHARS.intersection(text):
                    # Left the string without closing it: dead branch.
                    stats.invalid_paths += 1
                    continue
                if len(child) >= cfg.max_tokens:
                    stats.truncated_paths += 1
                    continue

                heapq.heappush(frontier, (-child_lp, tiebreak, child))
                tiebreak += 1
                stats.nodes_pushed += 1

        stats.distinct_values = len(merged)
        if on_progress is not None:
            on_progress(stats)

    stats.exhausted = not frontier
    stats.frontier_bound = -frontier[0][0] if frontier else NEG_INF

    candidates = [
        ValueCandidate(
            value=value,
            raw=m.best_raw,
            logprob=m.total,
            n_paths=m.n_paths,
            best_path=m.best_path,
        )
        for value, m in merged.items()
    ]
    candidates.sort(key=lambda c: (-c.logprob, c.value))
    stats.distinct_values = len(candidates)
    stats.found_logmass = logsumexp([c.logprob for c in candidates])
    return candidates[: cfg.k], stats
