from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cr_bot.app.main import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    from cr_bot.app.cli import main

    main()
