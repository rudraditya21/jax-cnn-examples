from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _regnet_block(
    out_channels: int,
    stride: int,
    bottleneck_ratio: float,
    group_width: int,
):
    """
    RegNet bottleneck block with grouped 3x3 convolution.

    Args:
        out_channels: Output channels after the block.
        stride: Spatial stride for the 3x3 convolution.
        bottleneck_ratio: Expansion ratio inside the bottleneck.
        group_width: Channel count per group for the grouped convolution.
    """

    def block(input_shape):
        _, _, _, in_channels = input_shape
        bottleneck_channels = max(
            group_width, int(round(out_channels * bottleneck_ratio / group_width)) * group_width
        )
        groups = max(1, bottleneck_channels // group_width)

        shortcut = (
            stax.serial(
                stax.Conv(out_channels, (1, 1), strides=(stride, stride), padding="SAME"),
                stax.BatchNorm(),
            )
            if stride != 1 or in_channels != out_channels
            else stax.Identity
        )

        main = stax.serial(
            stax.Conv(bottleneck_channels, (1, 1), padding="SAME"),
            stax.BatchNorm(),
            stax.Relu,
            stax.Conv(
                bottleneck_channels,
                (3, 3),
                strides=(stride, stride),
                padding="SAME",
                feature_group_count=groups,
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

    return stax.shape_dependent(block)


def _make_stage(
    width: int,
    depth: int,
    stride: int,
    bottleneck_ratio: float,
    group_width: int,
):
    layers: list = [
        _regnet_block(
            width,
            stride=stride,
            bottleneck_ratio=bottleneck_ratio,
            group_width=group_width,
        )
    ]
    layers.extend(
        _regnet_block(
            width,
            stride=1,
            bottleneck_ratio=bottleneck_ratio,
            group_width=group_width,
        )
        for _ in range(depth - 1)
    )
    return stax.serial(*layers)


def _regnet(
    widths: Sequence[int],
    depths: Sequence[int],
    num_classes: int,
    stem_width: int = 32,
    bottleneck_ratio: float = 1.0,
    group_width: int = 16,
):
    """
    Builds a RegNet backbone with configurable widths and depths.
    """
    layers: list = [
        stax.Conv(stem_width, (3, 3), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
    ]

    for idx, (width, depth) in enumerate(zip(widths, depths, strict=False)):
        stride = 1 if idx == 0 else 2
        layers.append(
            _make_stage(
                width,
                depth,
                stride=stride,
                bottleneck_ratio=bottleneck_ratio,
                group_width=group_width,
            )
        )

    layers.extend(
        [
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


@ModuleRegistry.register("regnetx-400mf")
def regnetx_400mf(num_classes: int = 1000):
    """
    RegNetX-400MF configuration (Radosavovic et al., 2020).
    """
    return _regnet(
        widths=(24, 56, 152, 368),
        depths=(1, 3, 6, 2),
        bottleneck_ratio=1.0,
        group_width=16,
        num_classes=num_classes,
    )


@ModuleRegistry.register("regnetx-800mf")
def regnetx_800mf(num_classes: int = 1000):
    """
    RegNetX-800MF configuration (Radosavovic et al., 2020).
    """
    return _regnet(
        widths=(64, 128, 288, 672),
        depths=(2, 4, 10, 2),
        bottleneck_ratio=1.0,
        group_width=16,
        num_classes=num_classes,
    )
