from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _repvgg_block(out_channels: int, stride: int = 1):
    """
    RepVGG block with 3x3 and 1x1 conv branches plus optional identity.
    """

    def block(input_shape):
        _, _, _, in_channels = input_shape
        branches = [
            stax.serial(
                stax.Conv(out_channels, (3, 3), strides=(stride, stride), padding="SAME"),
                stax.BatchNorm(),
            ),
            stax.serial(
                stax.Conv(out_channels, (1, 1), strides=(stride, stride), padding="SAME"),
                stax.BatchNorm(),
            ),
        ]

        if stride == 1 and in_channels == out_channels:
            branches.append(stax.Identity)

        return stax.serial(
            stax.FanOut(len(branches)),
            stax.parallel(*branches),
            stax.FanInSum,
            stax.Relu,
        )

    return stax.shape_dependent(block)


def _make_stage(width: int, depth: int, stride: int):
    layers: list = [_repvgg_block(width, stride=stride)]
    layers.extend(_repvgg_block(width, stride=1) for _ in range(depth - 1))
    return stax.serial(*layers)


def _repvgg(
    widths: Sequence[int],
    depths: Sequence[int],
    num_classes: int,
    stem_width: int = 64,
):
    """
    Builds a RepVGG-style plain CNN.
    """
    layers: list = [
        stax.Conv(stem_width, (3, 3), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
    ]

    for idx, (width, depth) in enumerate(zip(widths, depths, strict=False)):
        stride = 1 if idx == 0 else 2
        layers.append(_make_stage(width, depth, stride=stride))

    layers.extend(
        [
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


@ModuleRegistry.register("repvgg-a0")
def repvgg_a0(num_classes: int = 1000):
    """
    RepVGG-A0 (Ding et al., 2021) configuration.
    """
    return _repvgg(
        widths=(48, 96, 192, 1280),
        depths=(2, 4, 14, 1),
        stem_width=48,
        num_classes=num_classes,
    )


@ModuleRegistry.register("repvgg-b1")
def repvgg_b1(num_classes: int = 1000):
    """
    RepVGG-B1 (Ding et al., 2021) configuration with wider stages.
    """
    return _repvgg(
        widths=(64, 128, 256, 2048),
        depths=(2, 4, 16, 1),
        stem_width=64,
        num_classes=num_classes,
    )
