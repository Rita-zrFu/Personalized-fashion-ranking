"""Small CLI for this dataset: prepare, EDA, and the ranking experiment."""
import argparse
import json
from pathlib import Path

from .data import prepare, load_tables


def main():
    parser = argparse.ArgumentParser(description="H&M recommendation / ranking project")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prep.add_argument("--output", type=Path, default=Path("data/processed/hm_sample"))
    prep.add_argument("--max-users", type=int, default=50000, help="0 retains all customers")
    prep.add_argument("--seed", type=int, default=20261013)
    exploratory = sub.add_parser("eda")
    exploratory.add_argument("--data-dir", type=Path, default=Path("data/processed/hm_sample"))
    exploratory.add_argument("--output", type=Path, default=Path("reports/runs/hm_sample/eda"))
    experiment = sub.add_parser("run")
    experiment.add_argument("--config", type=Path, default=Path("configs/hm_sample.json"))
    folds = sub.add_parser("rolling")
    folds.add_argument("--config", type=Path, default=Path("configs/hm_sample.json"))
    report = sub.add_parser("publish")
    report.add_argument("--run-dir", type=Path, default=Path("reports/runs/hm_sample"))
    report.add_argument("--output", type=Path, default=Path("reports/hm_sample"))
    applied = sub.add_parser("applied", help="Build product-facing figures from saved H&M predictions")
    applied.add_argument("--config", type=Path, default=Path("configs/hm_sample.json"))
    applied.add_argument("--project-root", type=Path)
    applied.add_argument("--image-dir", type=Path, help="Optional local H&M image directory; no downloads")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.raw_dir, args.output, args.max_users or None, args.seed), indent=2))
    elif args.command == "eda":
        from .analysis import eda
        print(json.dumps(eda(*load_tables(args.data_dir), args.output), indent=2, default=int))
    elif args.command == "run":
        from .pipeline import run
        run(args.config)
    elif args.command == "rolling":
        from .rolling import rolling
        rolling(args.config)
    elif args.command == "publish":
        from .reporting import publish
        publish(args.run_dir, args.output)
    elif args.command == "applied":
        from .applied import generate_applied
        generate_applied(args.config, args.project_root, args.image_dir)


if __name__ == "__main__":
    main()
