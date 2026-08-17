"""Textual front end.

The model lives on the GPU and a walk takes seconds, so everything that
touches it runs in a thread worker and reports back through
``call_from_thread``. The UI thread never blocks.

Layout is built for a small terminal. Inputs are stacked one per row with the
label to their left, because side by side they collapse to a couple of
characters each on a narrow window. The preamble is a whole document, so it
lives on its own screen (F2) rather than eating six rows of the main view,
and everything that needs explaining is on the help screen (F1).
"""

from __future__ import annotations

import json

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Static,
    TextArea,
)

from .engine import (
    DEFAULT_SORT,
    SORT_LABELS,
    SORT_MODES,
    RunConfig,
    RunResult,
    run,
)
from .lm import DEFAULT_MODEL
from .prompt import DEFAULT_PREAMBLE
from .walk import WalkConfig, WalkStats

COLUMNS = (
    ("#", 3),
    ("value", 24),
    ("P*D", 9),
    ("P(value)", 10),
    ("P(v&T)", 9),
    ("D(T-F)", 8),
    ("P(true)", 8),
    ("mass", 6),
    ("tok", 4),
    ("paths", 6),
    ("pre", 4),
)

HELP = """\
# jsonwalk

Builds `{"<string field>": "<value>", "<bool field>": true}` by asking a base
language model to write it, then reads the answer off the probabilities.

## What you type

| Field | Meaning |
| --- | --- |
| **string field** | The JSON key whose values get enumerated, e.g. `city_name`. |
| **bool field** | A yes/no key applied to each value, e.g. `is_in_europe`. |
| **values (k)** | **How many rows to show.** 20 is a good start. Five times that many are enumerated and scored behind the scenes, and the k are picked *after* sorting - so switching sort mode can surface a value that was never in the visible list. |

## What you get back

| Column | Meaning |
| --- | --- |
| **P*D** | `P(value)` x `D(T-F)`: likely **and** true. The default ranking. Signed, so anything the model rejects is negative and sinks. |
| **P(value)** | Probability of the whole string, summed over every way of writing it in tokens. |
| **P(v&T)** | `P(value)` x `P(true)`: the probability of the whole object. |
| **D(T-F)** | `log P(true) - log P(false)`. **The verdict.** `+2.3` means the model is ~10x more willing to write true. |
| **P(true)** | The same thing as a probability, given a boolean is written at all. |
| **mass** | `P(true) + P(false)` in absolute terms. **Read this first.** |
| **tok** | Tokens in the most likely spelling. Values of different lengths compete fairly. |
| **paths** | How many token sequences were merged into this row. |
| **pre** | `*` means this value appears verbatim in your preamble. |

## Sorting (ctrl+s cycles)

1. **P(value) x D(T-F)** - the default: likely *and* true.
2. **P(value) x P(true)** - the probability of the whole object.
3. **value likelihood** - what the model would write, verdict ignored.
4. **true/false delta** - strongest verdict first, however unlikely.

The default multiplies by the *signed* delta, so every value the model
accepts outranks every value it rejects and `P(value)` only orders within a
verdict. Measured against hand-labelled answers it gets 1.00 precision on
both `is_metal` and `is_in_europe`, against 0.94 and 0.41 for mode 2, whose
`P(true)` saturates and lets likelihood swamp the verdict.

Mode 3 has nothing to do with the boolean, so a list in that order looks
arbitrary if you read it as a judgement. Mode 4 separates as well as the
default but fills the top with rare values - `Lyon, Bordeaux` rather than
`Paris, London`.

Each mode re-picks the k from the whole scored pool, so a value ranked far
down by likelihood can appear at the top under another mode. On
`element` / `is_metal` that is the difference between five metals padded out
with hydrogen, helium and carbon, and eight metals.

## Why "paths" is more than 1

Highlight a row and the bottom pane spells them out. Almost always the value
is spelled one way and the variants are different *closing* tokens: `",`,
`"}`, `"},` are each single tokens in this vocabulary, so the same value
finishes several ways. They are summed because they are the same answer. The
top path usually holds ~97% of it.

## bool_mass is the sanity check

If `bool_mass` is low the model never intended to write a boolean in that
slot, and `D(T-F)` is a ratio between two things it did not want to say. With
no preamble at all its favourite continuation there is a *digit*. The fix is
the preamble (F2), not a bigger k.

## Preamble (F2)

A base model continues a document, so the preamble decides both which values
appear and whether the boolean means anything. The default uses worked
examples from unrelated domains on purpose: examples that reuse the field you
are asking about hijack it - ask for `city_name` with startup examples and you
get Stripe and Google. `{field}` and `{bool_field}` expand to what you typed.

## Keys

Every action has a button on the main screen as well as a key, and no key is
a function key - `F1`/`F2` are aliases, not requirements.

| Key | Action |
| --- | --- |
| `ctrl+r` or `enter` | Run |
| `ctrl+s` | Cycle the sort |
| `ctrl+y` | Copy every row as JSON |
| `ctrl+b` or `F2` | Edit the preamble |
| `ctrl+g` or `F1` | This help |
| `ctrl+q` | Quit |
| `escape` | Close this screen |
"""

CSS = """
Screen { layout: vertical; }

/* One input per row: side by side they collapse to a few characters on a
   narrow terminal. Borderless and height 1 so three rows cost three rows. */
#fields { height: auto; border: round $primary; padding: 0 1; }
.row { height: 1; }
.row Label { width: 14; color: $text-muted; }
.row Input { height: 1; border: none; padding: 0; background: $boost; }
#actions { height: 1; margin-top: 0; }
#actions Button {
    height: 1;
    border: none;
    min-width: 8;
    margin-right: 2;
    padding: 0 1;
}

#status { height: auto; padding: 0 1; color: $text-muted; }
#detail { height: 3; padding: 0 1; color: $text-muted; }
DataTable { height: 1fr; }

HelpScreen, PreambleScreen { align: center middle; }
#sheet {
    width: 90%;
    max-width: 96;
    height: 85%;
    border: round $primary;
    background: $surface;
    padding: 0 1;
}
#preamble { height: 1fr; border: none; }
#buttons { height: 3; align: right middle; }
#buttons Button { margin: 0 1; }
"""


class HelpScreen(ModalScreen[None]):
    """Everything that would otherwise need a label on the main screen."""

    BINDINGS = [("escape,f1,q", "close", "Close")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="sheet"):
            yield Markdown(HELP)
        yield Footer()

    def action_close(self) -> None:
        self.dismiss(None)


class PreambleScreen(ModalScreen[str]):
    """The preamble is a document, not a field. It gets its own screen."""

    BINDINGS = [("escape", "cancel", "Cancel"), ("ctrl+s", "save", "Save")]

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="sheet"):
            yield Label(
                "Preamble  -  {field} and {bool_field} expand to your field names"
            )
            yield TextArea(self._text, id="preamble", show_line_numbers=False)
            with Horizontal(id="buttons"):
                yield Button("Reset to default", id="reset")
                yield Button("Cancel", id="cancel")
                yield Button("Save", id="save", variant="primary")
        yield Footer()

    @on(Button.Pressed, "#reset")
    def _reset(self) -> None:
        self.query_one("#preamble", TextArea).text = DEFAULT_PREAMBLE

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(self._text)

    @on(Button.Pressed, "#save")
    def action_save(self) -> None:
        self.dismiss(self.query_one("#preamble", TextArea).text)


class JsonWalkApp(App):
    """Enter a field name, watch its likely values, read the verdicts."""

    CSS = CSS
    TITLE = "jsonwalk"
    # Every action has a ctrl binding and a button, so nothing needs a
    # function key: F1/F2 are aliases, not the only way in. ctrl+a/e/d/f/k/u/w
    # are all consumed by Input, and ctrl+p is the command palette.
    BINDINGS = [
        ("ctrl+r", "run", "Run"),
        ("ctrl+s", "sort", "Sort"),
        ("ctrl+y", "copy", "Copy JSON"),
        ("ctrl+b,f2", "preamble", "Preamble"),
        ("ctrl+g,f1", "help", "Help"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model_id: str = DEFAULT_MODEL, k: int = 20) -> None:
        super().__init__()
        self.model_id = model_id
        self.k = k
        self.lm = None  # HFLanguageModel, built on first run inside the worker
        self.result: RunResult | None = None
        self.preamble = DEFAULT_PREAMBLE
        self.sort_mode = DEFAULT_SORT
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="fields"):
            with Horizontal(classes="row"):
                yield Label("string field")
                yield Input(value="startup_name", id="field")
            with Horizontal(classes="row"):
                yield Label("bool field")
                yield Input(value="good_sounding_name", id="bool_field")
            with Horizontal(classes="row"):
                yield Label("values (k)")
                yield Input(value=str(self.k), id="k", type="integer")
            # Clickable and tab-reachable, so no keyboard shortcut is required.
            with Horizontal(id="actions"):
                yield Button("Run", id="run", variant="primary")
                yield Button("Sort", id="sortbtn")
                yield Button("Preamble", id="preamblebtn")
                yield Button("Help", id="helpbtn")
        yield Static("", id="status")
        yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
        # markup=False: token pieces are rendered as [Hel][ium]["], and Rich
        # would parse those brackets as style tags and silently eat them.
        yield Static("", id="detail", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        for name, width in COLUMNS:
            table.add_column(name, width=width, key=name)
        self.set_status("enter or ctrl+r to run   -   F1 explains every column")

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # -- actions ---------------------------------------------------------
    @on(Input.Submitted)
    def _on_submit(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#run")
    def _btn_run(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#sortbtn")
    def _btn_sort(self) -> None:
        self.action_sort()

    @on(Button.Pressed, "#preamblebtn")
    def _btn_preamble(self) -> None:
        self.action_preamble()

    @on(Button.Pressed, "#helpbtn")
    def _btn_help(self) -> None:
        self.action_help()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_preamble(self) -> None:
        def store(text: str | None) -> None:
            if text is not None and text != self.preamble:
                self.preamble = text
                self.set_status("preamble updated - ctrl+r to re-run")

        self.push_screen(PreambleScreen(self.preamble), store)

    def action_run(self) -> None:
        if self.busy:
            self.set_status("already running - wait for the current walk")
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
            raise ValueError("values (k) must be a positive integer")
        return RunConfig(
            field=field,
            bool_field=bool_field,
            preamble=self.preamble,
            k=int(raw_k),
            walk=WalkConfig(),
        )

    def action_sort(self) -> None:
        if self.result is None:
            return
        i = SORT_MODES.index(self.sort_mode)
        self.sort_mode = SORT_MODES[(i + 1) % len(SORT_MODES)]
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
            f"{len(result.rows)} scored of {s.distinct_values} found, "
            f"{s.as_dict()['found_mass']:.1%} of the mass, "
            f"{s.expansions} expansions"
        )
        shown = result.select(self.sort_mode)
        if shown and shown[0].bool_score is not None:
            worst = min(r.bool_score.bool_mass for r in shown)
            if worst < 0.5:
                note += (
                    f"  |  WARNING bool_mass down to {worst:.2f}: the model is "
                    "not reliably writing a boolean - edit the preamble (F2)"
                )
        self.set_status(note)

    def fill_table(self) -> None:
        if self.result is None:
            return
        table = self.query_one("#table", DataTable)
        table.clear()
        # select() re-picks the k from the whole scored pool for this mode,
        # rather than reshuffling a list that was already cut down by
        # likelihood. No model call: everything is scored already.
        for row in self.result.select(self.sort_mode):
            b = row.bool_score
            table.add_row(
                str(row.rank),
                row.value,
                f"{row.signal:+.4f}" if b else "-",
                f"{row.candidate.prob:.5f}",
                f"{row.joint_prob:.5f}" if b else "-",
                f"{b.delta:+.2f}" if b else "-",
                f"{b.p_true:.2f}" if b else "-",
                f"{b.bool_mass:.3f}" if b else "-",
                str(len(row.candidate.best_path)),
                str(row.candidate.n_paths),
                "*" if row.echoed else "",
                key=str(row.rank),
            )
        self.sub_title = f"sorted by {SORT_LABELS[self.sort_mode]}"

    @on(DataTable.RowHighlighted)
    def _on_row(self, event: DataTable.RowHighlighted) -> None:
        if self.result is None or event.row_key.value is None:
            return
        rank = int(event.row_key.value)
        row = next((r for r in self.result.rows if r.rank == rank), None)
        if row is None:
            return
        detail = json.dumps(row.as_object(self.result.schema), ensure_ascii=False)
        if row.echoed:
            detail += "   [appears in the preamble]"
        if row.bool_score is not None:
            words = "  ".join(
                f"{w}={lp:.2f}" for w, lp in row.bool_score.per_word.items()
            )
            detail += f"\n{words}"
        detail += "\n" + self.describe_paths(row)
        self.query_one("#detail", Static).update(detail)

    def describe_paths(self, row) -> str:
        """Spell out the tokenizations behind the `paths` count.

        Nearly always the value body is tokenized one way and the variants are
        different *closing* tokens -- `",` `"}` `"},` are each single tokens --
        so this is what makes that visible rather than mysterious.
        """
        toks = row.candidate.tokenizations
        if self.lm is None or not toks:
            return ""
        parts = []
        for t in toks[:3]:
            pieces = "".join(f"[{self.lm.decode([i])}]" for i in t.tokens)
            parts.append(f"{t.prob / row.candidate.prob:.0%} {pieces}")
        more = f"  +{len(toks) - 3} more" if len(toks) > 3 else ""
        return f"paths: {'  '.join(parts)}{more}"


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
