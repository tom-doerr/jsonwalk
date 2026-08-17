"""Headless entry point: same engine as the TUI, printed or piped."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .engine import RunConfig, RunResult, run
from .lm import DEFAULT_MODEL
from .prompt import DEFAULT_PREAMBLE, SCHEMA_ONLY_PREAMBLE
from .walk import WalkConfig

EPILOG = """\
examples:
  jsonwalk city_name is_in_europe            rank cities, judge each
  jsonwalk startup_name good_name -k 30      more candidates
  jsonwalk product is_expensive --objects    one JSON object per line
  jsonwalk pet_name is_cute --json | jq .    full detail, machine readable
  jsonwalk                                   open the interactive TUI

Read bool_mass first: it is the absolute probability the model writes a
boolean at all. If it is low the delta is meaningless and the preamble,
not the search, is what needs fixing.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jsonwalk",
        description=(
            "Walk a base LM's completion tree to rank the values of a JSON "
            "string field, then score a boolean field by the true/false "
            "logprob delta."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"jsonwalk {__version__}")
    p.add_argument("field", nargs="?", default="startup_name", help="string field")
    p.add_argument("bool_field", nargs="?", default="good_sounding_name")
    p.add_argument("-k", type=int, default=20, help="values to return")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=16, help="cap per value")
    p.add_argument("--child-top-k", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-expansions", type=int, default=4000)
    p.add_argument(
        "--preamble", help="literal preamble text; {field}/{bool_field} expand"
    )
    p.add_argument("--preamble-file", help="read the preamble from a file")
    p.add_argument(
        "--schema-only-preamble",
        action="store_true",
        help="use the no-examples preamble (weaker verdicts, no echoed values)",
    )
    p.add_argument("--no-bool", action="store_true", help="skip boolean scoring")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument(
        "--objects",
        action="store_true",
        help="emit one JSON object per line, nothing else",
    )
    return p


def resolve_preamble(args: argparse.Namespace) -> str:
    chosen = [
        bool(args.preamble),
        bool(args.preamble_file),
        args.schema_only_preamble,
    ]
    if sum(chosen) > 1:
        raise SystemExit("pick at most one of --preamble/--preamble-file/--schema-only")
    if args.preamble is not None:
        return args.preamble
    if args.preamble_file:
        with open(args.preamble_file, encoding="utf-8") as fh:
            return fh.read()
    if args.schema_only_preamble:
        return SCHEMA_ONLY_PREAMBLE
    return DEFAULT_PREAMBLE


def format_table(result: RunResult) -> str:
    head = (
        f"{'#':>3}  {'value':<32} {'P(value)':>9} {'tok':>3} {'paths':>5} "
        f"{'D(T-F)':>7} {'P(true)':>7} {'bool_mass':>9}"
    )
    lines = [head, "-" * len(head)]
    for row in result.rows:
        b = row.bool_score
        cells = (
            f"{row.rank:>3}  {row.value[:32]:<32} {row.candidate.prob:>9.5f} "
            f"{len(row.candidate.best_path):>3} {row.candidate.n_paths:>5} "
        )
        if b is None:
            lines.append(cells + f"{'-':>7} {'-':>7} {'-':>9}")
        else:
            lines.append(
                cells + f"{b.delta:>+7.2f} {b.p_true:>7.2f} {b.bool_mass:>9.3f}"
            )
    s = result.stats
    lines.append("")
    lines.append(
        f"{s.distinct_values} distinct values seen, covering "
        f"{s.as_dict()['found_mass']:.1%} of the probability mass; "
        f"{s.expansions} expansions"
        + ("; search exhausted" if s.exhausted else "")
        + ("; top-k provably complete" if s.complete_top_k else "")
    )
    if result.rows and result.rows[0].bool_score is not None:
        worst = min(r.bool_score.bool_mass for r in result.rows)
        if worst < 0.5:
            lines.append(
                f"WARNING: bool_mass as low as {worst:.3f} -- the model is not "
                "reliably writing a boolean there. Strengthen the preamble."
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunConfig(
        field=args.field,
        bool_field=args.bool_field,
        preamble=resolve_preamble(args),
        score_bool=not args.no_bool,
        walk=WalkConfig(
            k=args.k,
            max_tokens=args.max_tokens,
            child_top_k=args.child_top_k,
            batch_size=args.batch_size,
            max_expansions=args.max_expansions,
        ),
    )
    # Imported here, not at module scope: pulling in torch costs seconds, and
    # --help / --version must not pay for it.
    from .hf import HFLanguageModel

    print(f"loading {args.model} ...", file=sys.stderr)
    lm = HFLanguageModel(args.model, batch_size=args.batch_size)
    result = run(lm, config)

    if args.objects:
        for row in result.rows:
            print(json.dumps(row.as_object(result.schema), ensure_ascii=False))
    elif args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_table(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
