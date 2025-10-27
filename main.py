import sys

from src.train import parse_training_args, run_training


def main() -> None:
    """Entry point for launching a training run from the project root."""

    try:
        training_config, logger_config = parse_training_args()
        run_training(training_config, logger_config)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
