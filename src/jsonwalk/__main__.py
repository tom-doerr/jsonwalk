"""Command dispatch.

    jsonwalk city_name is_in_europe   headless, the common case
    jsonwalk run city_name ...        same thing, explicit
    jsonwalk                          interactive TUI
    jsonwalk tui                      TUI, explicit

Bare arguments go to the CLI so the tool behaves like a normal command;
``run`` and ``tui`` are reserved as the first word, which is why a field
literally named "run" needs the explicit ``jsonwalk run run ...`` form.
"""

from __future__ import annotations

import sys

RESERVED = {"run", "tui"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "tui":
        from .tui import main as tui_main

        return tui_main(argv[1:])

    if argv and argv[0] == "run":
        from .cli import main as cli_main

        return cli_main(argv[1:])

    if argv:
        from .cli import main as cli_main

        return cli_main(argv)

    from .tui import main as tui_main

    return tui_main([])


if __name__ == "__main__":
    raise SystemExit(main())
