"""Headless checks that the TUI mounts and renders results.

Textual can drive an app with no terminal, which catches the errors that
otherwise only show up as a blank screen: a bad CSS selector, a widget id
that does not exist, a column count that does not match the row.
"""

import asyncio

from fake_lm import FakeLM
from jsonwalk.engine import RunConfig, run
from jsonwalk.tui import JsonWalkApp
from jsonwalk.walk import WalkConfig
from test_engine import TABLE, VOCAB
from textual.widgets import DataTable, Input, Static


def make_result():
    lm = FakeLM(VOCAB, TABLE)
    cfg = RunConfig(
        field="n",
        bool_field="b",
        preamble="",
        walk=WalkConfig(k=10, max_tokens=6, child_top_k=10),
        true_words=(" true",),
        false_words=(" false",),
    )
    return run(lm, cfg)


def drive(coro_factory):
    async def go():
        app = JsonWalkApp()
        async with app.run_test() as pilot:
            await coro_factory(app, pilot)

    asyncio.run(go())


def test_app_mounts_with_all_expected_widgets():
    async def check(app, pilot):
        assert app.query_one("#field", Input).value == "startup_name"
        assert app.query_one("#bool_field", Input).value == "good_sounding_name"
        assert len(app.query_one("#table", DataTable).columns) == 8

    drive(check)


def test_results_render_one_row_per_value():
    result = make_result()

    async def check(app, pilot):
        app.show_result(result)
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert table.row_count == len(result.rows)
        assert "distinct values" in str(app.query_one("#status", Static).render())

    drive(check)


def test_sort_toggle_reorders_by_delta():
    result = make_result()

    async def check(app, pilot):
        app.show_result(result)
        app.action_sort()
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        first = table.get_row_at(0)
        # "ab" has the highest delta as well as the highest probability, so
        # check the tail instead: "c" is the most-false and must sink.
        last = table.get_row_at(table.row_count - 1)
        assert first[1] == "ab"
        assert last[1] == "c"

    drive(check)


def test_invalid_k_is_reported_instead_of_crashing():
    async def check(app, pilot):
        app.query_one("#k", Input).value = "0"
        app.action_run()
        await pilot.pause()
        assert "invalid input" in str(app.query_one("#status", Static).render())
        assert not app.busy

    drive(check)
