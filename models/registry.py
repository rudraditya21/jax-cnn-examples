from __future__ import annotations

from collections.abc import Callable
from typing import Any

ModelFn = Callable[..., Any]


class ModuleRegistry:
    _registry: dict[str, ModelFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[ModelFn], ModelFn]:
        def decorator(fn: ModelFn) -> ModelFn:
            if name in cls._registry:
                raise ValueError(f"Model '{name}' is already registered.")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> ModelFn:
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> tuple[str, ...]:
        return tuple(cls._registry.keys())
