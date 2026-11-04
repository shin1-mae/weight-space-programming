"""Command line entry point: ``wsp <exp1|exp2|exp3|all> [options]``.

Examples::

    uv run wsp all                         # every experiment with the offline scripted agent
    uv run wsp exp1 --agent claude         # Claude reads the part networks (needs credentials)
    uv run wsp exp3 --agent claude --seeds 0
    uv run wsp exp2 --no-retrain --cases ecn2_readme_example
"""

from __future__ import annotations

import sys

COMMANDS = {
    "exp1": "wsp.experiments.exp1_read_3d_semantics",
    "exp2": "wsp.experiments.exp2_write_ecn_edit",
    "exp3": "wsp.experiments.exp3_fix_tolerance_bias",
}


def main(argv: list[str] | None = None) -> None:
    import importlib

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("commands:", ", ".join([*COMMANDS, "all"]))
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "all":
        for name in COMMANDS:
            print(f"\n=== {name} ===", flush=True)
            importlib.import_module(COMMANDS[name]).main(rest)
        return
    if cmd not in COMMANDS:
        sys.exit(f"unknown command {cmd!r}; choose from {', '.join([*COMMANDS, 'all'])}")
    importlib.import_module(COMMANDS[cmd]).main(rest)


if __name__ == "__main__":
    main()
