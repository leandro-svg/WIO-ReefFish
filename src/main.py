"""CLI entry point.

Run from the repository's ``src/`` folder:

    python main.py --help
    python main.py list-baselines
    python main.py stats --dataset ../data/WIO-ReefFish
    python main.py train-baseline --baseline yolo26 --data ../data/WIO-ReefFish/data.yaml

Equivalent to ``python -m fish_monitoring`` when ``src/`` is on ``PYTHONPATH``.
"""

from __future__ import annotations

import sys

from fish_monitoring.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
