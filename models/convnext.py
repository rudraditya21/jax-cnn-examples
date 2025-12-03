from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _depthwise_conv7x7():
    """
    Depthwise 7x7 convolution that adapts to input channel count.
    """

    def block(input_shape):
        _, _, _, in_channels = input_shape
        return stax.Conv(
            in_channels,
            (7, 7),
            padding="SAME",
            feature_group_count=in_channels,
        )

    return stax.shape_dependent(block)


def _convnext_block(dim: int, expansion: int = 4):
    """
    ConvNeXt block: depthwise conv -> pointwise MLP -> residual add.
    """

    def block(input_shape):
        _, _, _, in_channels = input_shape
        hidden_dim = dim * expansion

        main = stax.serial(
            _depthwise_conv7x7(),
            stax.BatchNorm(),
            stax.Conv(hidden_dim, (1, 1), padding="SAME"),
            stax.Relu,
            stax.Conv(dim, (1, 1), padding="SAME"),
        )

        return stax.serial(
            stax.FanOut(2),
            stax.parallel(main, stax.Identity),
            stax.FanInSum,
        )

    return stax.shape_dependent(block)


def _convnext(
    depths: Sequence[int],
    dims: Sequence[int],
    num_classes: int,
):
    """
    Builds a ConvNeXt-style model with stage-wise downsampling.
    """
    layers: list = [
        stax.Conv(dims[0], (4, 4), strides=(4, 4), padding="SAME"),
        stax.BatchNorm(),
    ]

    for idx, (depth, dim) in enumerate(zip(depths, dims, strict=False)):
        if idx > 0:
            layers.extend(
                [
                    stax.Conv(dim, (2, 2), strides=(2, 2), padding="SAME"),
                    stax.BatchNorm(),
                ]
            )

        layers.extend(_convnext_block(dim) for _ in range(depth))

    layers.extend(
        [
            stax.Relu,
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


@ModuleRegistry.register("convnext-t")
def convnext_t(num_classes: int = 1000):
    """
    ConvNeXt-Tiny (Liu et al., 2022) with reduced MLP depth.
    """
    return _convnext(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), num_classes=num_classes)


@ModuleRegistry.register("convnext-s")
def convnext_s(num_classes: int = 1000):
    """
    ConvNeXt-Small (Liu et al., 2022) with deeper third stage.
    """
    return _convnext(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), num_classes=num_classes)
