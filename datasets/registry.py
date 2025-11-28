from __future__ import annotations

from collections.abc import Callable
from typing import Any

DatasetFn = Callable[..., tuple[Any, Any, int]]


class DatasetRegistry:
    _registry: dict[str, DatasetFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[DatasetFn], DatasetFn]:
        def decorator(fn: DatasetFn) -> DatasetFn:
            if name in cls._registry:
                raise ValueError(f"Dataset '{name}' is already registered.")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> DatasetFn:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def list_datasets(cls) -> tuple[str, ...]:
        return tuple(cls._registry.keys())
