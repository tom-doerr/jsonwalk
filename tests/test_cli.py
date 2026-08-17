import subprocess
import sys
from pathlib import Path

import pytest
from jsonwalk import __main__ as entry
from jsonwalk.cli import build_parser, resolve_preamble
from jsonwalk.prompt import DEFAULT_PREAMBLE, PREAMBLE_STYLES

SRC = str(Path(__file__).resolve().parents[1] / "src")


def parse(args):
    return build_parser().parse_args(args)


def test_field_names_are_positional():
    args = parse(["city_name", "is_in_europe", "-k", "5"])
    assert (args.field, args.bool_field, args.k) == ("city_name", "is_in_europe", 5)


def test_preamble_defaults_and_overrides(tmp_path):
    assert resolve_preamble(parse([])) == DEFAULT_PREAMBLE
    assert resolve_preamble(parse(["--preamble", "hi\n"])) == "hi\n"
    for style, text in PREAMBLE_STYLES.items():
        assert resolve_preamble(parse(["--preamble-style", style])) == text

    path = tmp_path / "p.txt"
    path.write_text("from a file\n", encoding="utf-8")
    assert resolve_preamble(parse(["--preamble-file", str(path)])) == "from a file\n"


def test_conflicting_preamble_sources_are_rejected():
    with pytest.raises(SystemExit):
        resolve_preamble(parse(["--preamble", "x", "--preamble-style", "comment"]))


def test_empty_preamble_is_honoured_not_treated_as_unset():
    # `--preamble ""` is a real choice: no preamble at all.
    assert resolve_preamble(parse(["--preamble", ""])) == ""


def test_bare_arguments_go_to_the_cli(monkeypatch):
    seen = {}
    monkeypatch.setattr("jsonwalk.cli.main", lambda argv: seen.setdefault("cli", argv))
    monkeypatch.setattr("jsonwalk.tui.main", lambda argv: seen.setdefault("tui", argv))

    entry.main(["city_name", "is_in_europe"])
    assert seen == {"cli": ["city_name", "is_in_europe"]}


def test_explicit_subcommands_strip_themselves(monkeypatch):
    seen = {}
    monkeypatch.setattr("jsonwalk.cli.main", lambda argv: seen.setdefault("cli", argv))
    monkeypatch.setattr("jsonwalk.tui.main", lambda argv: seen.setdefault("tui", argv))

    entry.main(["run", "-k", "3"])
    assert seen["cli"] == ["-k", "3"]

    seen.clear()
    entry.main(["tui", "--model", "x"])
    assert seen["tui"] == ["--model", "x"]


def test_no_arguments_opens_the_tui(monkeypatch):
    seen = {}
    monkeypatch.setattr("jsonwalk.tui.main", lambda argv: seen.setdefault("tui", argv))
    entry.main([])
    assert seen == {"tui": []}


def test_help_does_not_import_torch():
    # Loading torch costs seconds and a CUDA context. Argument parsing must
    # not trigger it, which is why DEFAULT_MODEL lives in lm.py and the
    # backend is imported inside main().
    code = "import sys, jsonwalk.cli; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        check=True,
    )
    assert out.stdout.strip() == "False"
