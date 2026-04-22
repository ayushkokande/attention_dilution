"""Thin wrapper that starts a local ``circuit-tracer`` visualization server.

After running ``src/trace_refusal_circuit.py`` you will have pruned JSON graph
files in ``results/<slug>/graph_files/``. Point this script at that directory
to browse and annotate the graphs.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

from utils import results_dir

from circuit_tracer.frontend.local_server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing pruned graph JSON files. "
        "Defaults to results/<model_slug>/graph_files/.",
    )
    parser.add_argument("--port", type=int, default=8046)
    args = parser.parse_args()

    data_dir = args.data_dir or (results_dir() / "graph_files")
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"{data_dir} not found. Run `python src/trace_refusal_circuit.py` first."
        )

    server = serve(data_dir=str(data_dir), port=args.port)
    print(f"\nServing {data_dir} at http://localhost:{args.port}/index.html")
    print("Press Ctrl+C to stop.")

    stopped = {"v": False}

    def _stop(*_):
        if not stopped["v"]:
            stopped["v"] = True
            try:
                server.stop()
            except Exception:  # noqa: BLE001
                pass

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stopped["v"]:
            time.sleep(1)
    finally:
        _stop()


if __name__ == "__main__":
    main()
