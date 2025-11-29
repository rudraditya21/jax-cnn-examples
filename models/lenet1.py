from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


@ModuleRegistry.register("lenet1")
def lenet1(num_classes: int = 10):
    """
    LeNet-1 as described by LeCun et al. for 32x32 inputs.
    Conv(4,5x5) -> AvgPool -> Conv(12,5x5) -> AvgPool -> Flatten -> Dense(num_classes)
    """
    return stax.serial(
        stax.Conv(4, (5, 5), padding="VALID"),
        stax.Tanh,
        stax.AvgPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(12, (5, 5), padding="VALID"),
        stax.Tanh,
        stax.AvgPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Flatten,
        stax.Dense(num_classes),
    )
