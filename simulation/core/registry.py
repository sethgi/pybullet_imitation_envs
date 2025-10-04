from enum import Enum


class Registry:
    def __init__(self):
        self._map = {}

    def register(self, name):
        def inner(cls):
            self._map[name] = cls
            return cls
        return inner

    def build(self, name, *args, **kwargs):
        if name not in self._map:
            raise ValueError(f"{name} not found in registry. Available: {list(self._map.keys())}")
        cls = self._map[name]

        # If first arg is an Enum and class has .create(), use that
        if hasattr(cls, "create"):
            return cls.create(*args, **kwargs)
        else:
            return cls(*args, **kwargs)
ROBOT_REGISTRY = Registry()
TASK_REGISTRY = Registry()
TELEOP_REGISTRY = Registry()