"""Run one of the four maintained experiment scripts."""

import argparse
import importlib
import sys

COMMANDS = {
    "baseline": "experiment_1.baseline_benchmark",
    "direction": "experiment_2.refusal_direction",
    "context": "experiment_8.context_sweep",
    "projection": "experiment_9.projection_sweep",
}


def main():
    parser = argparse.ArgumentParser(description="Controlled context-length study")
    parser.add_argument("stage", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.argv = [COMMANDS[args.stage], *args.arguments]
    importlib.import_module(COMMANDS[args.stage]).main()


if __name__ == "__main__":
    main()
