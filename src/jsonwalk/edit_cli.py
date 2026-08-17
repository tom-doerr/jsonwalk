"""``jsonwalk edit`` -- propose (and optionally apply) edits to a file."""

from __future__ import annotations

import argparse
import difflib
import json
import sys

from .edit import EditConfig, EditProposals, EditSchema, propose_edits
from .lm import DEFAULT_MODEL

EDIT_EPILOG = """\
examples:
  jsonwalk edit draft.md -o "Make it concise."
  jsonwalk edit draft.md -o "Remove hedging." --sort delta
  jsonwalk edit draft.md -o "Be concise." --iterations 3 --in-place

The search anchor is generated under a constraint that it appear verbatim and
exactly once in the file, so a proposed edit always applies. Nothing is
written unless you pass --in-place.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jsonwalk edit",
        description="Propose search/replace edits ranked by a stated objective.",
        epilog=EDIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", help="text artifact to edit ('-' for stdin)")
    p.add_argument(
        "-o", "--objective", default="Make the text clearer and more concise."
    )
    p.add_argument("-n", "--anchors", type=int, default=8, help="search anchors")
    p.add_argument("-r", "--replacements", type=int, default=4, help="per anchor")
    p.add_argument("--min-anchor", type=int, default=6, help="min anchor chars")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sort", choices=("signal", "delta", "value"), default="signal")
    p.add_argument("--iterations", type=int, default=1, help="apply-and-repeat")
    p.add_argument(
        "--in-place", action="store_true", help="write the file (prints a diff first)"
    )
    p.add_argument("--json", action="store_true", help="emit the proposals as JSON")
    return p


def format_proposals(proposals: EditProposals, sort: str) -> str:
    lines = [
        f"{'D(T-F)':>7} {'P(edit)':>9}  edit",
        "-" * 60,
    ]
    for e in proposals.ranked(sort):
        lines.append(
            f"{e.bool_score.delta:>+7.2f} {e.prob:>9.2e}  "
            f"{e.search!r} -> {e.replace!r}"
        )
    s = proposals.search_stats
    lines.append("")
    lines.append(
        f"{proposals.n_anchors} anchors, {len(proposals.edits)} edits; "
        f"{s.rejected_paths} branches pruned for leaving the document "
        f"in {s.expansions} expansions"
    )
    return "\n".join(lines)


def diff(before: str, after: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=name,
            tofile=f"{name} (edited)",
        )
    )


def edit_as_dict(e) -> dict[str, object]:
    return {
        "search": e.search,
        "replace": e.replace,
        "occurrences": e.occurrences,
        "prob": e.prob,
        "delta_true_false": e.bool_score.delta,
        "p_true": e.bool_score.p_true,
        "bool_mass": e.bool_score.bool_mass,
        "signal": e.signal,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.in_place and args.file == "-":
        raise SystemExit("--in-place needs a real file, not stdin")

    document = (
        sys.stdin.read()
        if args.file == "-"
        else open(args.file, encoding="utf-8").read()
    )
    if not document.strip():
        raise SystemExit(f"{args.file} is empty; nothing to edit")

    from .hf import HFLanguageModel

    print(f"loading {args.model} ...", file=sys.stderr)
    lm = HFLanguageModel(args.model)
    config = EditConfig(
        n_search=args.anchors,
        n_replace=args.replacements,
        min_anchor=args.min_anchor,
    )
    original, current, applied = document, document, []

    for round_no in range(1, args.iterations + 1):
        schema = EditSchema(document=current, objective=args.objective)
        proposals = propose_edits(lm, schema, config)
        if not proposals.edits:
            print(f"round {round_no}: no valid edits found", file=sys.stderr)
            break
        _report(proposals, args, round_no)

        best = proposals.ranked(args.sort)[0]
        if best.bool_score.delta <= 0:
            print("best edit is judged no improvement; stopping", file=sys.stderr)
            break
        current = best.apply_to(current)
        applied.append(best)

    return _finish(original, current, applied, args)


def _report(proposals: EditProposals, args, round_no: int) -> None:
    if args.json:
        print(
            json.dumps(
                {
                    "round": round_no,
                    "objective": args.objective,
                    "edits": [edit_as_dict(e) for e in proposals.ranked(args.sort)],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"--- round {round_no} ---")
        print(format_proposals(proposals, args.sort))


def _finish(original: str, current: str, applied: list, args) -> int:
    if not applied:
        return 0
    if not args.json:
        print("\n" + diff(original, current, args.file))
    if args.in_place:
        with open(args.file, "w", encoding="utf-8") as fh:
            fh.write(current)
        print(f"wrote {len(applied)} edit(s) to {args.file}", file=sys.stderr)
    else:
        print("(dry run; pass --in-place to write)", file=sys.stderr)
    return 0
