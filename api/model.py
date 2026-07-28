import threading
from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class LazyModel(Generic[T]):
    """Load a model on first use and share it across subsequent requests."""

    def __init__(self, load_func: Callable[[], T]):
        self._load_func = load_func
        self._model: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_func()
        return self._model
