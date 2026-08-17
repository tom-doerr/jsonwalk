"""Headless entry point: same engine as the TUI, printed or piped."""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from . import __version__
from .engine import SORT_LABELS, SORT_MODES, RunConfig, RunResult, run, sort_rows
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
    p.add_argument(
        "--sort",
        choices=SORT_MODES,
        default="value",
        help=(
            "value: most likely values (default). joint: P(value) x P(true), "
            "i.e. likely AND true. delta: strongest verdict first."
        ),
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


def format_table(result: RunResult, sort: str = "value", width: int = 0) -> str:
    # Numeric columns are fixed; the value column takes whatever is left, so
    # the table fits an 80-column terminal and uses a wide one.
    fixed = len("  #   P(value)   P(v&T)   D(T-F) P(true)  mass tok paths pre")
    width = width or shutil.get_terminal_size((100, 24)).columns
    vw = max(12, min(40, width - fixed))

    head = (
        f"{'#':>3}  {'value':<{vw}} {'P(value)':>9} {'P(v&T)':>8} "
        f"{'D(T-F)':>7} {'P(true)':>7} {'mass':>5} {'tok':>3} {'paths':>5} pre"
    )
    lines = [head, "-" * len(head)]
    for row in sort_rows(result.rows, sort):
        b = row.bool_score
        cells = f"{row.rank:>3}  {row.value[:vw]:<{vw}} {row.candidate.prob:>9.5f} "
        if b is None:
            cells += f"{'-':>8} {'-':>7} {'-':>7} {'-':>5} "
        else:
            cells += (
                f"{row.joint_prob:>8.5f} {b.delta:>+7.2f} "
                f"{b.p_true:>7.2f} {b.bool_mass:>5.3f} "
            )
        cells += f"{len(row.candidate.best_path):>3} {row.candidate.n_paths:>5}"
        lines.append(cells + ("   *" if row.echoed else ""))
    s = result.stats
    lines.append("")
    lines.append(
        f"{s.distinct_values} distinct values seen, covering "
        f"{s.as_dict()['found_mass']:.1%} of the probability mass; "
        f"{s.expansions} expansions"
        + ("; search exhausted" if s.exhausted else "")
        + ("; top-k provably complete" if s.complete_top_k else "")
    )
    echoed = [r.value for r in result.rows if r.echoed]
    if echoed:
        lines.append(
            f"* {len(echoed)} value(s) appear verbatim in the preamble "
            f"({', '.join(echoed[:4])}) -- few-shot examples get echoed back."
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
        for row in sort_rows(result.rows, args.sort):
            print(json.dumps(row.as_object(result.schema), ensure_ascii=False))
    elif args.json:
        payload = result.as_dict()
        payload["sort"] = args.sort
        payload["rows"] = [
            r.as_dict(result.schema) for r in sort_rows(result.rows, args.sort)
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_table(result, sort=args.sort))
        print(f"sorted by {SORT_LABELS[args.sort]}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
