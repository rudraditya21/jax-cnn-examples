from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.nn.initializers import glorot_uniform

from .registry import ModuleRegistry


def _depthwise_conv(
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: str = "SAME",
):
    """
    Depthwise convolution helper for NHWC tensors.

    Args:
        kernel_size: Spatial kernel size.
        stride: Spatial stride.
        padding: Padding mode, `SAME` or `VALID`.

    Returns:
        A `(init_fn, apply_fn)` pair implementing depthwise convolution.
    """

    def init_fun(rng, input_shape):
        if len(input_shape) != 4:
            raise ValueError("DepthwiseConv expects NHWC input.")
        _, h, w, c = input_shape
        k_h, k_w = kernel_size
        # Depthwise: one input channel per group, output channels equal to input channels.
        w_shape = (k_h, k_w, 1, c)
        k_rng, _ = jax.random.split(rng)
        W = glorot_uniform()(k_rng, w_shape)
        b = jnp.zeros((c,), dtype=W.dtype)

        def out_dim(in_size: int, k: int, s: int) -> int:
            if padding == "SAME":
                return math.ceil(in_size / s)
            if padding == "VALID":
                return math.ceil((in_size - k + 1) / s)
            raise ValueError(f"Unsupported padding: {padding}")

        out_h = out_dim(h, k_h, stride[0])
        out_w = out_dim(w, k_w, stride[1])
        return (input_shape[0], out_h, out_w, c), (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        y = jax.lax.conv_general_dilated(
            inputs,
            W,
            window_strides=stride,
            padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=inputs.shape[-1],
        )
        return y + b

    return init_fun, apply_fun


def _channel_shuffle(groups: int):
    """
    Rearranges channels to enable cross-group information flow.

    Args:
        groups: Number of channel groups.

    Returns:
        A `(init_fn, apply_fn)` pair performing channel shuffle.
    """

    def init_fun(rng, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Channel shuffle expects NHWC input.")
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        n, h, w, c = inputs.shape
        if c % groups != 0 or groups <= 1:
            return inputs
        x = inputs.reshape(n, h, w, groups, c // groups)
        x = jnp.transpose(x, (0, 1, 2, 4, 3))
        return x.reshape(n, h, w, c)

    return init_fun, apply_fun


def _shuffle_unit(out_channels: int, stride: int, groups: int):
    """
    ShuffleNetV2 unit with channel shuffle and optional projection.

    Args:
        out_channels: Output channels from the unit.
        stride: Spatial stride (1 keeps resolution, 2 downsamples).
        groups: Number of groups for shuffle.

    Returns:
        Shape-dependent stax block.
    """

    def block(input_shape):
        _, _, _, in_channels = input_shape
        mid_channels = out_channels // 4

        proj_channels = max(1, out_channels - in_channels) if stride != 1 else out_channels

        branch_main = stax.serial(
            stax.Conv(mid_channels, (1, 1), padding="SAME"),
            stax.BatchNorm(),
            stax.Relu,
            _channel_shuffle(groups),
            _depthwise_conv(stride=(stride, stride)),
            stax.BatchNorm(),
            stax.Conv(proj_channels, (1, 1), padding="SAME"),
            stax.BatchNorm(),
            stax.Relu,
        )

        if stride == 1:
            return stax.serial(
                stax.FanOut(2),
                stax.parallel(branch_main, stax.Identity),
                stax.FanInConcat(axis=-1),
            )

        branch_proj = stax.serial(
            _depthwise_conv(stride=(stride, stride)),
            stax.BatchNorm(),
            stax.Conv(in_channels, (1, 1), padding="SAME"),
            stax.BatchNorm(),
            stax.Relu,
        )

        return stax.serial(
            stax.FanOut(2),
            stax.parallel(branch_main, branch_proj),
            stax.FanInConcat(axis=-1),
        )

    return stax.shape_dependent(block)


def _make_shufflenet_v2(
    stages_repeats: Sequence[int],
    stages_out_channels: Sequence[int],
    num_classes: int,
    groups: int,
):
    """
    Builds ShuffleNetV2 with configurable channel scales.

    Args:
        stages_repeats: Number of units per stage.
        stages_out_channels: Output channel sizes per stage (including final 1x1).
        num_classes: Number of output classes.
        groups: Shuffle groups (fixed at 2 for these variants).

    Returns:
        `(init_fn, apply_fn)` tuple for ShuffleNetV2.
    """
    layers: list = [
        stax.Conv(24, (3, 3), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
    ]

    for repeats, out_ch in zip(stages_repeats, stages_out_channels, strict=False):
        for i in range(repeats):
            stride = 2 if i == 0 else 1
            layers.append(_shuffle_unit(out_ch, stride=stride, groups=groups))

    layers.extend(
        [
            stax.Conv(stages_out_channels[-1], (1, 1), padding="SAME"),
            stax.BatchNorm(),
            stax.Relu,
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


@ModuleRegistry.register("shufflenetv2-0.5")
def shufflenetv2_05(num_classes: int = 1000):
    """ShuffleNetV2 channel scale 0.5x."""
    return _make_shufflenet_v2((4, 8, 4), (48, 96, 192, 1024), num_classes=num_classes, groups=2)


@ModuleRegistry.register("shufflenetv2-1.0")
def shufflenetv2_10(num_classes: int = 1000):
    """ShuffleNetV2 channel scale 1.0x."""
    return _make_shufflenet_v2((4, 8, 4), (116, 232, 464, 1024), num_classes=num_classes, groups=2)


@ModuleRegistry.register("shufflenetv2-1.5")
def shufflenetv2_15(num_classes: int = 1000):
    """ShuffleNetV2 channel scale 1.5x."""
    return _make_shufflenet_v2((4, 8, 4), (176, 352, 704, 1024), num_classes=num_classes, groups=2)


@ModuleRegistry.register("shufflenetv2-2.0")
def shufflenetv2_20(num_classes: int = 1000):
    """ShuffleNetV2 channel scale 2.0x."""
    return _make_shufflenet_v2((4, 8, 4), (244, 488, 976, 2048), num_classes=num_classes, groups=2)
