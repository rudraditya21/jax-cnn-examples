from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _basic_block(
    channels: int, stride: int = 1, use_projection: bool = False, dropout_rate: float | None = None
):
    """
    WideResNet basic residual block with optional dropout and projection.
    """
    shortcut = (
        stax.serial(
            stax.Conv(channels, (1, 1), strides=(stride, stride), padding="SAME"),
            stax.BatchNorm(),
        )
        if use_projection
        else stax.Identity
    )

    main_layers = [
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(channels, (3, 3), strides=(stride, stride), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
    ]
    if dropout_rate:
        main_layers.append(stax.Dropout(dropout_rate))
    main_layers.append(stax.Conv(channels, (3, 3), padding="SAME"))

    main = stax.serial(*main_layers)

    return stax.serial(
        stax.FanOut(2),
        stax.parallel(main, shortcut),
        stax.FanInSum,
    )


def _make_layer(
    channels: int,
    blocks: int,
    stride: int,
    dropout_rate: float | None,
):
    """
    Stacks WideResNet basic blocks for a stage.
    """
    layers: list = [
        _basic_block(channels, stride=stride, use_projection=True, dropout_rate=dropout_rate)
    ]
    layers.extend(
        _basic_block(channels, stride=1, dropout_rate=dropout_rate) for _ in range(blocks - 1)
    )
    return stax.serial(*layers)


def _wideresnet(
    depth: int,
    widen_factor: int,
    num_classes: int,
    dropout_rate: float | None = 0.0,
):
    """
    Builds a WideResNet model (Zagoruyko & Komodakis).

    Args:
        depth: Total depth of the network (must satisfy depth = 6n + 4).
        widen_factor: Width multiplier for channels.
        num_classes: Number of output classes.
        dropout_rate: Dropout inside residual blocks; set None/0.0 to disable.

    Returns:
        `(init_fn, apply_fn)` tuple for WideResNet.
    """
    if (depth - 4) % 6 != 0:
        raise ValueError("Depth should be of the form 6n+4 for WideResNet.")
    n = (depth - 4) // 6
    channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

    return stax.serial(
        stax.Conv(channels[0], (3, 3), padding="SAME"),
        _make_layer(channels[1], blocks=n, stride=1, dropout_rate=dropout_rate),
        _make_layer(channels[2], blocks=n, stride=2, dropout_rate=dropout_rate),
        _make_layer(channels[3], blocks=n, stride=2, dropout_rate=dropout_rate),
        stax.BatchNorm(),
        stax.Relu,
        stax.AvgPool((8, 8), strides=(1, 1), padding="VALID"),
        stax.Flatten,
        stax.Dense(num_classes),
    )


@ModuleRegistry.register("wideresnet28-10")
def wideresnet28_10(num_classes: int = 10, dropout_rate: float | None = 0.0):
    """
    WideResNet-28-10 (depth 28, widen factor 10).
    """
    return _wideresnet(
        depth=28, widen_factor=10, num_classes=num_classes, dropout_rate=dropout_rate
    )


@ModuleRegistry.register("wideresnet40-2")
def wideresnet40_2(num_classes: int = 10, dropout_rate: float | None = 0.0):
    """
    WideResNet-40-2 (depth 40, widen factor 2).
    """
    return _wideresnet(depth=40, widen_factor=2, num_classes=num_classes, dropout_rate=dropout_rate)


@ModuleRegistry.register("wideresnet16-8")
def wideresnet16_8(num_classes: int = 10, dropout_rate: float | None = 0.0):
    """
    WideResNet-16-8 (depth 16, widen factor 8).
    """
    return _wideresnet(depth=16, widen_factor=8, num_classes=num_classes, dropout_rate=dropout_rate)
