"""Textual front end.

The model lives on the GPU and a walk takes seconds, so everything that
touches it runs in a thread worker and reports back through
``call_from_thread``. The UI thread never blocks.
"""

from __future__ import annotations

import json

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TextArea,
)

from .engine import RunConfig, RunResult, run
from .lm import DEFAULT_MODEL
from .prompt import DEFAULT_PREAMBLE
from .walk import WalkConfig, WalkStats

COLUMNS = (
    ("#", 3),
    ("value", 34),
    ("P(value)", 10),
    ("tok", 4),
    ("paths", 6),
    ("D(T-F)", 8),
    ("P(true)", 8),
    ("bool_mass", 10),
)

CSS = """
Screen { layout: vertical; }
#config { height: auto; border: round $primary; padding: 0 1; }
#preamble { height: 6; border: none; }
#fields { height: auto; }
#fields Input { width: 1fr; }
#fields Label { width: auto; padding: 1 1 0 1; }
#status { height: auto; padding: 0 1; color: $text-muted; }
#detail { height: 3; padding: 0 1; border: round $secondary; }
DataTable { height: 1fr; }
"""


class JsonWalkApp(App):
    """Enter a field name, watch its likely values, read the verdicts."""

    CSS = CSS
    TITLE = "jsonwalk"
    BINDINGS = [
        ("ctrl+r", "run", "Run"),
        ("ctrl+s", "sort", "Sort"),
        ("ctrl+y", "copy", "Copy JSON"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model_id: str = DEFAULT_MODEL, k: int = 20) -> None:
        super().__init__()
        self.model_id = model_id
        self.k = k
        self.lm = None  # HFLanguageModel, built on first run inside the worker
        self.result: RunResult | None = None
        self.sort_by_delta = False
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="config"):
            yield TextArea(DEFAULT_PREAMBLE, id="preamble", show_line_numbers=False)
            with Horizontal(id="fields"):
                yield Label("string field")
                yield Input(value="startup_name", id="field")
                yield Label("bool field")
                yield Input(value="good_sounding_name", id="bool_field")
                yield Label("k")
                yield Input(value=str(self.k), id="k", type="integer")
                yield Button("Run", id="run", variant="primary")
        yield Static("", id="status")
        yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
        yield Static("", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        for name, width in COLUMNS:
            table.add_column(name, width=width, key=name)
        self.set_status(f"press ctrl+r to run  |  model {self.model_id}")

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # -- actions ---------------------------------------------------------
    @on(Button.Pressed, "#run")
    def _on_run_button(self) -> None:
        self.action_run()

    @on(Input.Submitted)
    def _on_submit(self) -> None:
        self.action_run()

    def action_run(self) -> None:
        if self.busy:
            self.set_status("already running -- wait for the current walk")
            return
        try:
            config = self.read_config()
        except ValueError as exc:
            self.set_status(f"invalid input: {exc}")
            return
        self.busy = True
        self.do_run(config)

    def read_config(self) -> RunConfig:
        field = self.query_one("#field", Input).value.strip()
        bool_field = self.query_one("#bool_field", Input).value.strip()
        raw_k = self.query_one("#k", Input).value.strip()
        if not field or not bool_field:
            raise ValueError("both field names are required")
        if not raw_k.isdigit() or int(raw_k) < 1:
            raise ValueError("k must be a positive integer")
        return RunConfig(
            field=field,
            bool_field=bool_field,
            preamble=self.query_one("#preamble", TextArea).text,
            walk=WalkConfig(k=int(raw_k)),
        )

    def action_sort(self) -> None:
        if self.result is None:
            return
        self.sort_by_delta = not self.sort_by_delta
        self.fill_table()

    def action_copy(self) -> None:
        if self.result is None:
            return
        payload = "\n".join(
            json.dumps(r.as_object(self.result.schema), ensure_ascii=False)
            for r in self.result.rows
        )
        self.copy_to_clipboard(payload)
        self.set_status(f"copied {len(self.result.rows)} JSON objects to clipboard")

    # -- worker ----------------------------------------------------------
    @work(thread=True, exclusive=True)
    def do_run(self, config: RunConfig) -> None:
        try:
            if self.lm is None:
                self.call_from_thread(
                    self.set_status, f"loading {self.model_id} (first run only)..."
                )
                # Deferred so the UI paints before torch is imported.
                from .hf import HFLanguageModel

                self.lm = HFLanguageModel(self.model_id)

            def progress(stats: WalkStats) -> None:
                self.call_from_thread(
                    self.set_status,
                    f"walking: {stats.expansions} expansions, "
                    f"{stats.distinct_values} distinct values",
                )

            self.call_from_thread(self.set_status, "walking the completion tree...")
            result = run(self.lm, config, on_progress=progress)
            self.call_from_thread(self.show_result, result)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.call_from_thread(
                self.set_status, f"failed: {type(exc).__name__}: {exc}"
            )
            raise
        finally:
            self.busy = False

    # -- rendering -------------------------------------------------------
    def show_result(self, result: RunResult) -> None:
        self.result = result
        self.fill_table()
        s = result.stats
        note = (
            f"{s.distinct_values} distinct values, "
            f"{s.as_dict()['found_mass']:.1%} of the mass, "
            f"{s.expansions} expansions"
        )
        if result.rows and result.rows[0].bool_score is not None:
            worst = min(r.bool_score.bool_mass for r in result.rows)
            if worst < 0.5:
                note += (
                    f"  |  WARNING bool_mass down to {worst:.2f}: the model is "
                    "not reliably writing a boolean -- strengthen the preamble"
                )
        self.set_status(note)

    def fill_table(self) -> None:
        if self.result is None:
            return
        table = self.query_one("#table", DataTable)
        table.clear()
        rows = list(self.result.rows)
        if self.sort_by_delta:
            rows.sort(key=lambda r: -(r.bool_score.delta if r.bool_score else 0.0))
        for row in rows:
            b = row.bool_score
            table.add_row(
                str(row.rank),
                row.value,
                f"{row.candidate.prob:.5f}",
                str(len(row.candidate.best_path)),
                str(row.candidate.n_paths),
                f"{b.delta:+.2f}" if b else "-",
                f"{b.p_true:.2f}" if b else "-",
                f"{b.bool_mass:.3f}" if b else "-",
                key=str(row.rank),
            )
        order = "true/false delta" if self.sort_by_delta else "value likelihood"
        self.sub_title = f"sorted by {order}"

    @on(DataTable.RowHighlighted)
    def _on_row(self, event: DataTable.RowHighlighted) -> None:
        if self.result is None or event.row_key.value is None:
            return
        rank = int(event.row_key.value)
        row = next((r for r in self.result.rows if r.rank == rank), None)
        if row is None:
            return
        obj = json.dumps(row.as_object(self.result.schema), ensure_ascii=False)
        detail = obj
        if row.bool_score is not None:
            words = "  ".join(
                f"{w}={lp:.2f}" for w, lp in row.bool_score.per_word.items()
            )
            detail += f"\n{words}"
        self.query_one("#detail", Static).update(detail)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="jsonwalk tui",
        description=f"{JsonWalkApp.__doc__} Also opened by bare `jsonwalk`.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("-k", type=int, default=20, help="values to return")
    args = p.parse_args(argv)
    JsonWalkApp(model_id=args.model, k=args.k).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
