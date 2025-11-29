import argparse

from rich.console import Console
from rich.table import Table

from datasets.registry import DatasetRegistry
from models.registry import ModuleRegistry


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNN Architecture")

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str.lower,
        required=True,
        choices=DatasetRegistry.list_datasets(),
        help="Dataset name registered in DatasetRegistry",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download/load datasets",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Global batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader worker processes",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation for training",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str.lower,
        required=True,
        choices=ModuleRegistry.list_models(),
        help="Model name registered in ModuleRegistry",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override number of output classes (defaults to dataset value)",
    )

    # Optimization / training loop
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Base learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay or AdamW decoupled weight decay",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD-based optimizers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization and data shuffling",
    )

    # Runtime
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device selection (auto/cpu/gpu/tpu)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Steps between logging training metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation",
    )

    return parser


def print_args_table(args: argparse.Namespace) -> None:
    console = Console()
    table = Table(title="Run Configuration")
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in vars(args).items():
        table.add_row(key.replace("_", "-"), str(value))

    console.print(table)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print_args_table(args)
