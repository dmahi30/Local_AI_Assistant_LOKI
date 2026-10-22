"""CLI entry point to launch the LOKI desktop UI."""

import sys
from .launcher import run_desktop_ui


def main() -> None:
    success = run_desktop_ui()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
