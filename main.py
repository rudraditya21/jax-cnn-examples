import argparse
import os
from collections.abc import Iterable

from rich.console import Console
from rich.table import Table

from datasets.registry import DatasetRegistry
from models.registry import ModuleRegistry


def _console() -> Console:
    return Console()


def _print_table(title: str, rows: Iterable[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in rows:
        table.add_row(key, value)
    _console().print(table)


def list_items(section: str) -> None:
    if section in ("models", "all"):
        _print_table(
            "Available Models",
            [(str(idx + 1), name) for idx, name in enumerate(ModuleRegistry.list_models())],
        )
    if section in ("datasets", "all"):
        _print_table(
            "Available Datasets",
            [(str(idx + 1), name) for idx, name in enumerate(DatasetRegistry.list_datasets())],
        )


def _set_device(device: str) -> None:
    """
    Hint JAX which platform to target (CPU/GPU/TPU) for Colab/TPU use.
    """
    if device and device != "auto":
        os.environ["JAX_PLATFORM_NAME"] = device


def run_model_check(
    model_name: str,
    num_classes: int,
    batch_size: int,
    image_size: tuple[int, int],
    channels: int,
    device: str,
    seed: int,
    echo_config: bool,
) -> None:
    _set_device(device)

    import jax
    import jax.numpy as jnp

    init_fun, apply_fun = ModuleRegistry.get(model_name)(num_classes=num_classes)

    h, w = image_size
    dummy_shape = (batch_size, h, w, channels)
    init_rng, apply_rng = jax.random.split(jax.random.PRNGKey(seed))

    out_shape, params = init_fun(init_rng, dummy_shape)
    dummy = jnp.ones(dummy_shape, jnp.float32)
    output = apply_fun(params, dummy, rng=apply_rng)

    device_used = jax.devices()[0].platform
    if echo_config:
        _print_table(
            f"Model Check: {model_name}",
            [
                ("num-classes", str(num_classes)),
                ("batch-size", str(batch_size)),
                ("image-size", f"{h}x{w}"),
                ("channels", str(channels)),
                ("device", device_used),
                ("output-shape", str(output.shape)),
            ],
        )
    else:
        _console().print(
            f"[green]Success[/green]: {model_name} ran on {device_used} with output shape {output.shape}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model management utilities (list, sanity-check) for JAX CNN examples"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered models and datasets")
    list_parser.add_argument(
        "--section",
        choices=("models", "datasets", "all"),
        default="all",
        help="What to list",
    )

    check_parser = subparsers.add_parser("check-model", help="Run a dummy forward pass")
    check_parser.add_argument(
        "--model",
        type=str.lower,
        choices=ModuleRegistry.list_models(),
        help="Model name registered in ModuleRegistry",
    )
    check_parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run the check on every registered model (ignores --model)",
    )
    check_parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of output classes to configure the model with",
    )
    check_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used for the dummy input",
    )
    check_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(224, 224),
        help="Spatial size (H W) for the dummy input",
    )
    check_parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Channel count for the dummy input (3 for RGB, 1 for grayscale)",
    )
    check_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="auto",
        help="Preferred device; set to tpu in Colab TPU runtimes",
    )
    check_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parameter init and dropout RNG",
    )
    check_parser.add_argument(
        "--echo-config",
        action="store_true",
        help="Print a configuration table instead of a one-line success message",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "list":
        list_items(args.section)
        return

    if args.command == "check-model":
        if not args.all_models and not args.model:
            parser.error("No model specified. Provide --model or use --all-models.")

        target_models = ModuleRegistry.list_models() if args.all_models else [args.model]

        for name in target_models:
            run_model_check(
                model_name=name,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                image_size=tuple(args.image_size),
                channels=args.channels,
                device=args.device,
                seed=args.seed,
                echo_config=args.echo_config,
            )


if __name__ == "__main__":
    main()
