"""``jsonwalk`` launches the TUI; ``jsonwalk run ...`` is the headless path."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        from .cli import main as cli_main

        return cli_main(argv[1:])
    if argv and argv[0] == "tui":
        argv = argv[1:]
    from .tui import main as tui_main

    return tui_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
