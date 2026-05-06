from __future__ import annotations

import asyncio
import sys

from weakness_driven_problem_synthesis.run import main_with_args


def main() -> int:
    return asyncio.run(main_with_args(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
