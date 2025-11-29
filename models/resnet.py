from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _basic_block(channels: int, stride: int = 1, use_projection: bool = False):
    shortcut = (
        stax.serial(
            stax.Conv(channels, (1, 1), strides=(stride, stride), padding="SAME"),
            stax.BatchNorm(),
        )
        if use_projection
        else stax.Identity
    )

    main = stax.serial(
        stax.Conv(channels, (3, 3), strides=(stride, stride), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(channels, (3, 3), padding="SAME"),
        stax.BatchNorm(),
    )

    return stax.serial(
        stax.FanOut(2),
        stax.parallel(main, shortcut),
        stax.FanInSum,
        stax.Relu,
    )


def _bottleneck_block(channels: int, stride: int = 1, use_projection: bool = False):
    out_channels = channels * 4
    shortcut = (
        stax.serial(
            stax.Conv(out_channels, (1, 1), strides=(stride, stride), padding="SAME"),
            stax.BatchNorm(),
        )
        if use_projection
        else stax.Identity
    )

    main = stax.serial(
        stax.Conv(channels, (1, 1), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(channels, (3, 3), strides=(stride, stride), padding="SAME"),
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
    block_fn,
    channels: int,
    blocks: int,
    stride: int,
):
    layers: list = [block_fn(channels, stride=stride, use_projection=True)]
    layers.extend(block_fn(channels, stride=1) for _ in range(blocks - 1))
    return stax.serial(*layers)


def _resnet(
    block_fn,
    layers: Sequence[int],
    num_classes: int = 1000,
):
    return stax.serial(
        stax.Conv(64, (7, 7), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _make_layer(block_fn, 64, layers[0], stride=1),
        _make_layer(block_fn, 128, layers[1], stride=2),
        _make_layer(block_fn, 256, layers[2], stride=2),
        _make_layer(block_fn, 512, layers[3], stride=2),
        stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
        stax.Flatten,
        stax.Dense(num_classes),
    )


@ModuleRegistry.register("resnet18")
def resnet18(num_classes: int = 1000):
    """
    ResNet-18 (He et al., 2015) using basic residual blocks.
    """
    return _resnet(_basic_block, (2, 2, 2, 2), num_classes=num_classes)


@ModuleRegistry.register("resnet34")
def resnet34(num_classes: int = 1000):
    """
    ResNet-34 (He et al., 2015) using basic residual blocks.
    """
    return _resnet(_basic_block, (3, 4, 6, 3), num_classes=num_classes)


@ModuleRegistry.register("resnet50")
def resnet50(num_classes: int = 1000):
    """
    ResNet-50 (He et al., 2015) using bottleneck residual blocks.
    """
    return _resnet(_bottleneck_block, (3, 4, 6, 3), num_classes=num_classes)


@ModuleRegistry.register("resnet101")
def resnet101(num_classes: int = 1000):
    """
    ResNet-101 (He et al., 2015) using bottleneck residual blocks.
    """
    return _resnet(_bottleneck_block, (3, 4, 23, 3), num_classes=num_classes)


@ModuleRegistry.register("resnet152")
def resnet152(num_classes: int = 1000):
    """
    ResNet-152 (He et al., 2015) using bottleneck residual blocks.
    """
    return _resnet(_bottleneck_block, (3, 8, 36, 3), num_classes=num_classes)
