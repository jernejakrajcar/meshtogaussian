from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


class StageError(RuntimeError):
    """Exception that keeps the pipeline stage in the error message."""


@dataclass
class StageLogger:
    enabled: bool = True
    verbose: bool = True

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = perf_counter()
        succeeded = False
        if self.enabled and self.verbose:
            print(f"[stage] {name}...")
        try:
            yield
            succeeded = True
        except Exception as exc:
            message = f"Stage failed: {name}. Reason: {exc}"
            if self.enabled:
                print(f"[error] {message}")
            raise StageError(message) from exc
        finally:
            if succeeded and self.enabled and self.verbose:
                elapsed = perf_counter() - start
                print(f"[stage] {name} done in {elapsed:.2f}s")

    def iter(self, items: Iterable[T], description: str, total: int | None = None) -> Iterable[T]:
        if not self.enabled:
            return items

        try:
            from tqdm import tqdm
        except Exception:
            return self._plain_iter(items, description, total)

        return tqdm(items, desc=description, total=total)

    def _plain_iter(self, items: Iterable[T], description: str, total: int | None = None) -> Iterator[T]:
        if self.verbose:
            suffix = f" ({total} items)" if total is not None else ""
            print(f"[progress] {description}{suffix}")
        yield from items
