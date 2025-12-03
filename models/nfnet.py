from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _nfnet_block(out_channels: int, stride: int = 1, expansion: int = 2):
    """
    Simplified NFNet-style bottleneck block with pre-activation and residual add.
    """

    def block(input_shape):
        _, _, _, in_channels = input_shape
        hidden = out_channels * expansion

        shortcut = (
            stax.serial(
                stax.Conv(out_channels, (1, 1), strides=(stride, stride), padding="SAME"),
                stax.BatchNorm(),
            )
            if stride != 1 or in_channels != out_channels
            else stax.Identity
        )

        main = stax.serial(
            stax.BatchNorm(),
            stax.Relu,
            stax.Conv(hidden, (1, 1), padding="SAME"),
            stax.BatchNorm(),
            stax.Relu,
            stax.Conv(hidden, (3, 3), strides=(stride, stride), padding="SAME"),
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
    expansion: int,
):
    layers: list = [_nfnet_block(width, stride=stride, expansion=expansion)]
    layers.extend(_nfnet_block(width, stride=1, expansion=expansion) for _ in range(depth - 1))
    return stax.serial(*layers)


def _nfnet(
    widths: Sequence[int],
    depths: Sequence[int],
    num_classes: int,
    stem_width: int = 32,
    expansion: int = 2,
):
    """
    Builds a lightweight NFNet-style network with configurable stage widths.
    """
    layers: list = [
        stax.Conv(stem_width, (3, 3), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(stem_width * 2, (3, 3), strides=(1, 1), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(stem_width * 4, (3, 3), strides=(1, 1), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
    ]

    for idx, (width, depth) in enumerate(zip(widths, depths, strict=False)):
        stride = 1 if idx == 0 else 2
        layers.append(_make_stage(width, depth, stride=stride, expansion=expansion))

    layers.extend(
        [
            stax.BatchNorm(),
            stax.Relu,
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


@ModuleRegistry.register("nfnet-f0")
def nfnet_f0(num_classes: int = 1000):
    """
    NFNet-F0 inspired configuration (channels scaled down for lighter compute).
    """
    return _nfnet(
        widths=(256, 512, 1536, 1536),
        depths=(1, 2, 6, 3),
        stem_width=32,
        expansion=2,
        num_classes=num_classes,
    )


@ModuleRegistry.register("nfnet-f1")
def nfnet_f1(num_classes: int = 1000):
    """
    NFNet-F1 inspired configuration with wider stem and deeper middle stage.
    """
    return _nfnet(
        widths=(256, 512, 1536, 2048),
        depths=(2, 4, 12, 4),
        stem_width=48,
        expansion=2,
        num_classes=num_classes,
    )
