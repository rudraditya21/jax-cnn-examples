from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.nn.initializers import glorot_uniform

from .registry import ModuleRegistry

# stax lacks a built-in ReLU6; define a simple variant here.
RELU6 = stax.elementwise(lambda x: jnp.minimum(jnp.maximum(x, 0.0), 6.0))


def _depthwise_conv(
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: str = "SAME",
    channel_multiplier: int = 1,
):
    """
    Creates a depthwise convolution layer for NHWC inputs.

    Args:
        kernel_size: Convolutional kernel size.
        stride: Stride for spatial dimensions.
        padding: Padding mode, `SAME` or `VALID`.
        channel_multiplier: Multiplier for per-channel filters.

    Returns:
        A `(init_fn, apply_fn)` pair compatible with stax.serial.
    """

    def init_fun(rng, input_shape):
        if len(input_shape) != 4:
            raise ValueError("DepthwiseConv expects NHWC input.")
        _, h, w, c = input_shape
        k_h, k_w = kernel_size
        # Depthwise: one input channel per group, channel_multiplier outputs per group.
        w_shape = (k_h, k_w, 1, c * channel_multiplier)
        k_rng, _ = jax.random.split(rng)
        W = glorot_uniform()(k_rng, w_shape)
        b = jnp.zeros((c * channel_multiplier,), dtype=W.dtype)

        def out_dim(in_size: int, k: int, s: int) -> int:
            if padding == "SAME":
                return math.ceil(in_size / s)
            if padding == "VALID":
                return math.ceil((in_size - k + 1) / s)
            raise ValueError(f"Unsupported padding: {padding}")

        out_h = out_dim(h, k_h, stride[0])
        out_w = out_dim(w, k_w, stride[1])
        return (input_shape[0], out_h, out_w, c * channel_multiplier), (W, b)

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


def _conv_bn_relu(out_ch: int, kernel: tuple[int, int], stride: tuple[int, int] = (1, 1)):
    """Conv-BatchNorm-ReLU convenience stack."""
    return stax.serial(
        stax.Conv(out_ch, kernel, strides=stride, padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
    )


def _depthwise_separable(out_ch: int, stride: tuple[int, int] = (1, 1)):
    """Depthwise separable block used in MobileNetV1."""
    return stax.serial(
        _depthwise_conv(stride=stride),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(out_ch, (1, 1), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
    )


def _make_mobilenet_v1(width_mult: float, num_classes: int):
    """
    Builds a MobileNetV1 model.

    Args:
        width_mult: Width scaling factor (e.g., 1.0, 0.75).
        num_classes: Number of output classes.

    Returns:
        `(init_fn, apply_fn)` tuple for MobileNetV1.
    """

    def _c(channels: int) -> int:
        return max(1, int(round(channels * width_mult)))

    layers: list = [
        _conv_bn_relu(_c(32), (3, 3), stride=(2, 2)),
        _depthwise_separable(_c(64)),
        _depthwise_separable(_c(128), stride=(2, 2)),
        _depthwise_separable(_c(128)),
        _depthwise_separable(_c(256), stride=(2, 2)),
        _depthwise_separable(_c(256)),
        _depthwise_separable(_c(512), stride=(2, 2)),
    ]

    layers.extend(_depthwise_separable(_c(512)) for _ in range(5))
    layers.extend(
        [
            _depthwise_separable(_c(1024), stride=(2, 2)),
            _depthwise_separable(_c(1024)),
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


def _inverted_residual(out_ch: int, stride: int, expand_ratio: int):
    """
    Inverted residual block from MobileNetV2 with optional skip.

    Args:
        out_ch: Output channel count after projection.
        stride: Spatial stride (1 keeps resolution, 2 downsamples).
        expand_ratio: Expansion multiplier for hidden channels.

    Returns:
        Shape-dependent stax block.
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
                    RELU6,
                ]
            )

        layers.extend(
            [
                _depthwise_conv(stride=(stride, stride)),
                stax.BatchNorm(),
                RELU6,
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


def _make_mobilenet_v2(width_mult: float, num_classes: int):
    """
    Builds a MobileNetV2 model.

    Args:
        width_mult: Width scaling factor (e.g., 1.0, 0.75).
        num_classes: Number of output classes.

    Returns:
        `(init_fn, apply_fn)` tuple for MobileNetV2.
    """

    def _c(ch: int) -> int:
        return max(8, int(round(ch * width_mult)))

    settings = [
        # t, c, n, s
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    layers: list = [
        _conv_bn_relu(_c(32), (3, 3), stride=(2, 2)),
    ]

    for t, c, n, s in settings:
        out_ch = _c(c)
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(_inverted_residual(out_ch, stride=stride, expand_ratio=t))

    layers.extend(
        [
            stax.Conv(_c(1280), (1, 1), padding="SAME"),
            stax.BatchNorm(),
            RELU6,
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


@ModuleRegistry.register("mobilenetv1-1.0")
def mobilenetv1_1(num_classes: int = 1000):
    """MobileNetV1 width multiplier 1.0."""
    return _make_mobilenet_v1(1.0, num_classes)


@ModuleRegistry.register("mobilenetv1-0.75")
def mobilenetv1_075(num_classes: int = 1000):
    """MobileNetV1 width multiplier 0.75."""
    return _make_mobilenet_v1(0.75, num_classes)


@ModuleRegistry.register("mobilenetv1-0.5")
def mobilenetv1_05(num_classes: int = 1000):
    """MobileNetV1 width multiplier 0.5."""
    return _make_mobilenet_v1(0.5, num_classes)


@ModuleRegistry.register("mobilenetv1-0.25")
def mobilenetv1_025(num_classes: int = 1000):
    """MobileNetV1 width multiplier 0.25."""
    return _make_mobilenet_v1(0.25, num_classes)


@ModuleRegistry.register("mobilenetv2-1.0")
def mobilenetv2_1(num_classes: int = 1000):
    """MobileNetV2 width multiplier 1.0."""
    return _make_mobilenet_v2(1.0, num_classes)


@ModuleRegistry.register("mobilenetv2-0.75")
def mobilenetv2_075(num_classes: int = 1000):
    """MobileNetV2 width multiplier 0.75."""
    return _make_mobilenet_v2(0.75, num_classes)


@ModuleRegistry.register("mobilenetv2-0.5")
def mobilenetv2_05(num_classes: int = 1000):
    """MobileNetV2 width multiplier 0.5."""
    return _make_mobilenet_v2(0.5, num_classes)


@ModuleRegistry.register("mobilenetv2-0.35")
def mobilenetv2_035(num_classes: int = 1000):
    """MobileNetV2 width multiplier 0.35."""
    return _make_mobilenet_v2(0.35, num_classes)
