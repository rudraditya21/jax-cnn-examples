from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _resnext_block(
    planes: int, stride: int, base_width: int, cardinality: int, use_projection: bool = False
):
    """
    ResNeXt bottleneck block with grouped 3x3 convolution.

    Args:
        planes: Base channel size for the block.
        stride: Spatial stride (1 keeps resolution, 2 downsamples).
        base_width: Base width multiplier (e.g., 4 for 32x4d).
        cardinality: Number of groups in the grouped convolution.
        use_projection: Whether to apply a 1x1 projection to the shortcut.

    Returns:
        A stax block implementing the residual unit.
    """
    width = int(planes * (base_width / 64.0)) * cardinality
    out_channels = planes * 4

    shortcut = (
        stax.serial(
            stax.Conv(out_channels, (1, 1), strides=(stride, stride), padding="SAME"),
            stax.BatchNorm(),
        )
        if use_projection
        else stax.Identity
    )

    main = stax.serial(
        stax.Conv(width, (1, 1), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(
            width,
            (3, 3),
            strides=(stride, stride),
            padding="SAME",
            feature_group_count=cardinality,
        ),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(out_channels, (1, 1), padding="SAME"),
        stax.BatchNorm(),
    )

    return stax.serial(
        stax.FanOut(2),
        stax.parallel(main, shortcut),
        stax.FanInSum,
        stax.Relu,
    )


def _make_layer(
    planes: int,
    blocks: int,
    stride: int,
    base_width: int,
    cardinality: int,
):
    """
    Stacks ResNeXt blocks for one stage.
    """
    layers: list = [
        _resnext_block(
            planes,
            stride=stride,
            base_width=base_width,
            cardinality=cardinality,
            use_projection=True,
        )
    ]
    layers.extend(
        _resnext_block(planes, stride=1, base_width=base_width, cardinality=cardinality)
        for _ in range(blocks - 1)
    )
    return stax.serial(*layers)


def _resnext(
    layers: Sequence[int],
    base_width: int,
    cardinality: int,
    num_classes: int = 1000,
):
    """
    Builds a ResNeXt network.

    Args:
        layers: Block counts per stage (e.g., (3, 4, 6, 3) for 50-layer).
        base_width: Base width multiplier (e.g., 4 for 32x4d, 2 for 32x2d).
        cardinality: Number of groups in the grouped convolution.
        num_classes: Number of output classes.

    Returns:
        `(init_fn, apply_fn)` tuple for ResNeXt.
    """
    return stax.serial(
        stax.Conv(64, (7, 7), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _make_layer(64, layers[0], stride=1, base_width=base_width, cardinality=cardinality),
        _make_layer(128, layers[1], stride=2, base_width=base_width, cardinality=cardinality),
        _make_layer(256, layers[2], stride=2, base_width=base_width, cardinality=cardinality),
        _make_layer(512, layers[3], stride=2, base_width=base_width, cardinality=cardinality),
        stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
        stax.Flatten,
        stax.Dense(num_classes),
    )


@ModuleRegistry.register("resnext50-32x4d")
def resnext50_32x4d(num_classes: int = 1000):
    """
    ResNeXt-50 with cardinality 32 and base width 4.
    """
    return _resnext((3, 4, 6, 3), base_width=4, cardinality=32, num_classes=num_classes)


@ModuleRegistry.register("resnext101-32x4d")
def resnext101_32x4d(num_classes: int = 1000):
    """
    ResNeXt-101 with cardinality 32 and base width 4.
    """
    return _resnext((3, 4, 23, 3), base_width=4, cardinality=32, num_classes=num_classes)


@ModuleRegistry.register("resnext101-64x4d")
def resnext101_64x4d(num_classes: int = 1000):
    """
    ResNeXt-101 with cardinality 64 and base width 4 (wider variant).
    """
    return _resnext((3, 4, 23, 3), base_width=4, cardinality=64, num_classes=num_classes)
