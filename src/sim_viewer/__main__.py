"""Launch with ``python -m sim_viewer`` from the project environment."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone local V4 Arena Lab")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8010, type=int)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("sim_viewer.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
