import argparse
import os
from collections.abc import Iterable

import torch
from rich.console import Console
from rich.table import Table

from datasets.registry import DatasetRegistry
from models.registry import ModuleRegistry


def _console() -> Console:
    return Console(highlight=False)


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


def _one_hot(labels, num_classes: int):
    import jax.numpy as jnp

    return jnp.eye(num_classes, dtype=jnp.float32)[labels]


def train_model(
    dataset: str,
    model_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    augment: bool,
    num_classes_override: int | None,
    epochs: int,
    learning_rate: float,
    max_steps: int | None,
    resize_to: tuple[int, int] | None,
    device: str,
    seed: int,
) -> None:
    """
    Minimal JAX training loop to sanity-check model/dataset wiring.
    Keeps things intentionally lightweight (SGD, small steps) for quick verification.
    """
    _set_device(device)

    import jax
    import jax.numpy as jnp
    import numpy as np

    train_loader, eval_loader, num_classes_from_ds = DatasetRegistry.get(dataset)(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augment=augment,
    )

    num_classes = num_classes_override or num_classes_from_ds
    init_fun, apply_fun = ModuleRegistry.get(model_name)(num_classes=num_classes)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Infer input shape from one batch.
    sample_x, _ = next(iter(train_loader))
    sample_np = sample_x.numpy()
    if sample_np.ndim != 4:
        raise ValueError(f"Expected 4D input (NCHW), got shape {sample_np.shape}")
    if resize_to is not None:
        import torch.nn.functional as F

        sample_tensor = torch.from_numpy(sample_np)
        sample_tensor = F.interpolate(
            sample_tensor,
            size=resize_to,
            mode="bilinear",
            align_corners=False,
        )
        sample_np = sample_tensor.numpy()
    sample_np = np.transpose(sample_np, (0, 2, 3, 1))
    input_shape = sample_np.shape

    out_shape, params = init_fun(init_rng, input_shape)
    _console().print(
        f"[cyan]Initialized[/cyan] {model_name} for {dataset} -> input {input_shape}, output {out_shape}"
    )

    def _to_array(batch):
        imgs, labels = batch
        x = imgs.numpy()
        if resize_to is not None:
            import torch.nn.functional as F

            # Upsample NCHW to desired spatial size for models expecting larger inputs.
            x_tensor = torch.from_numpy(x)
            x_tensor = F.interpolate(
                x_tensor,
                size=resize_to,
                mode="bilinear",
                align_corners=False,
            )
            x = x_tensor.numpy()
        x = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        y = labels.numpy().astype(np.int32)
        return x, y

    @jax.jit
    def loss_and_grads(p, rng_key, x_np, y_np):
        x = jnp.asarray(x_np)
        y = jnp.asarray(y_np)
        onehot = _one_hot(y, num_classes)

        def forward(params):
            logits = apply_fun(params, x, rng=rng_key)
            log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
            loss = -jnp.mean(jnp.sum(onehot * log_probs, axis=-1))
            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.mean(preds == y)
            return loss, acc

        (loss, acc), grads = jax.value_and_grad(forward, has_aux=True)(p)
        return loss, acc, grads

    @jax.jit
    def sgd_update(p, g):
        return jax.tree_util.tree_map(lambda param, grad: param - learning_rate * grad, p, g)

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        steps = 0
        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break
            rng, step_rng = jax.random.split(rng)
            x_np, y_np = _to_array(batch)
            loss, acc, grads = loss_and_grads(params, step_rng, x_np, y_np)
            params = sgd_update(params, grads)
            train_loss += float(loss)
            train_acc += float(acc)
            steps += 1

        train_loss /= max(1, steps)
        train_acc = (train_acc / max(1, steps)) * 100.0

        # Quick eval on a single batch for speed.
        eval_batch = next(iter(eval_loader))
        x_np, y_np = _to_array(eval_batch)
        rng, eval_rng = jax.random.split(rng)
        logits = apply_fun(params, x_np, rng=eval_rng)
        preds = jnp.argmax(logits, axis=-1)
        eval_acc = float(jnp.mean(preds == jnp.asarray(y_np)) * 100.0)

        _console().print(
            f"[magenta]Epoch {epoch+1}/{epochs}[/magenta] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% eval_acc~{eval_acc:.2f}%"
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

    train_parser = subparsers.add_parser("train", help="Run a minimal training sanity-check")
    train_parser.add_argument(
        "--dataset",
        type=str.lower,
        choices=DatasetRegistry.list_datasets(),
        required=True,
        help="Dataset name registered in DatasetRegistry",
    )
    train_parser.add_argument(
        "--model",
        type=str.lower,
        choices=ModuleRegistry.list_models(),
        required=True,
        help="Model name registered in ModuleRegistry",
    )
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download/load datasets",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader worker processes",
    )
    train_parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation for training",
    )
    train_parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override number of output classes (defaults to dataset value)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for SGD",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Limit training steps per epoch (use None for full pass)",
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional spatial resize (e.g., 224 224 for ImageNet-style models)",
    )
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="auto",
        help="Preferred device; set to tpu in Colab TPU runtimes",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization and dropout RNG",
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

    if args.command == "train":
        max_steps = args.max_steps if args.max_steps >= 0 else None
        train_model(
            dataset=args.dataset,
            model_name=args.model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=not args.no_augment,
            num_classes_override=args.num_classes,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_steps=max_steps,
            resize_to=tuple(args.image_size) if args.image_size else None,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
