from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def progress(iterable: Iterable[T], total: int | None = None, desc: str | None = None) -> Iterator[T]:
    try:
        from tqdm.auto import tqdm

        yield from tqdm(iterable, total=total, desc=desc)
    except Exception:
        if desc:
            print(desc)
        yield from iterable


def log_step(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)

