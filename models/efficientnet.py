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
    Depthwise convolution helper for NHWC layouts.

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
        w_shape = (k_h, k_w, c, 1)
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


def _round_filters(
    filters: int, width_mult: float, divisor: int = 8, min_depth: int | None = None
) -> int:
    """
    Scales channel counts using EfficientNet rounding rules.

    Args:
        filters: Base channel count.
        width_mult: Width multiplier for the variant.
        divisor: Channels are rounded to be divisible by this value.
        min_depth: Optional lower bound for channels.

    Returns:
        Rounded channel count.
    """
    filters *= width_mult
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats: int, depth_mult: float) -> int:
    """Scales block repeat counts using the depth multiplier."""
    return int(math.ceil(repeats * depth_mult))


def _mbconv_block(out_ch: int, expand_ratio: int, stride: int, kernel_size: int):
    """
    MBConv block (without squeeze-excitation or stochastic depth; stax lacks native support).

    Args:
        out_ch: Output channels after projection.
        expand_ratio: Expansion multiplier for hidden channels.
        stride: Spatial stride (1 keeps resolution, 2 downsamples).
        kernel_size: Depthwise convolution kernel size.

    Returns:
        Shape-dependent stax block with optional residual.
    """

    def block(input_shape):
        _, _, _, in_ch = input_shape
        mid_ch = in_ch * expand_ratio
        layers: list = []

        if expand_ratio != 1:
            layers.extend(
                [
                    stax.Conv(mid_ch, (1, 1), padding="SAME"),
                    stax.BatchNorm(),
                    stax.Relu,
                ]
            )

        layers.extend(
            [
                _depthwise_conv(kernel_size=(kernel_size, kernel_size), stride=(stride, stride)),
                stax.BatchNorm(),
                stax.Relu,
                stax.Conv(out_ch, (1, 1), padding="SAME"),
                stax.BatchNorm(),
            ]
        )

        core = stax.serial(*layers)

        if stride == 1 and in_ch == out_ch:
            return stax.serial(
                stax.FanOut(2),
                stax.parallel(core, stax.Identity),
                stax.FanInSum,
            )
        return core

    return stax.shape_dependent(block)


def _make_efficientnet(
    width_mult: float,
    depth_mult: float,
    num_classes: int,
    dropout_rate: float,
):
    """
    Builds an EfficientNet model with scaled width/depth.

    Args:
        width_mult: Width scaling factor.
        depth_mult: Depth scaling factor.
        num_classes: Number of output classes.
        dropout_rate: Dropout applied before the classifier.

    Returns:
        `(init_fn, apply_fn)` tuple for EfficientNet.
    """
    base_config: Sequence[tuple[int, int, int, int, int]] = (
        # expand, out, repeats, stride, kernel
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    )

    layers: list = [
        stax.Conv(_round_filters(32, width_mult), (3, 3), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
    ]

    for expand, out_c, repeats, stride, k in base_config:
        out = _round_filters(out_c, width_mult)
        reps = _round_repeats(repeats, depth_mult)
        for i in range(reps):
            s = stride if i == 0 else 1
            layers.append(_mbconv_block(out, expand_ratio=expand, stride=s, kernel_size=k))

    layers.extend(
        [
            stax.Conv(_round_filters(1280, width_mult), (1, 1), padding="SAME"),
            stax.BatchNorm(),
            stax.Relu,
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
        ]
    )

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.extend([stax.Flatten, stax.Dense(num_classes)])

    return stax.serial(*layers)


_VARIANTS = {
    "efficientnet-b0": (1.0, 1.0, 0.2),
    "efficientnet-b1": (1.0, 1.1, 0.2),
    "efficientnet-b2": (1.1, 1.2, 0.3),
    "efficientnet-b3": (1.2, 1.4, 0.3),
    "efficientnet-b4": (1.4, 1.8, 0.4),
    "efficientnet-b5": (1.6, 2.2, 0.4),
    "efficientnet-b6": (1.8, 2.6, 0.5),
    "efficientnet-b7": (2.0, 3.1, 0.5),
}


def _register_variant(name: str, width: float, depth: float, dropout: float):
    @ModuleRegistry.register(name)
    def factory(num_classes: int = 1000):
        return _make_efficientnet(
            width_mult=width, depth_mult=depth, num_classes=num_classes, dropout_rate=dropout
        )

    return factory


for _name, (w, d, p) in _VARIANTS.items():
    _register_variant(_name, w, d, p)
