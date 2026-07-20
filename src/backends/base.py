"""Shared protocol and result types that every ASR backend conforms to."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class TimestampItem:
    text: str
    start_time: float
    end_time: float


@dataclass
class TimestampResult:
    items: List[TimestampItem] = field(default_factory=list)


@dataclass
class ASRResult:
    language: str
    text: str
    time_stamps: Optional[TimestampResult] = None


class ASRBackend(Protocol):
    """Minimal contract that every backend satisfies.

    ModelManager holds exactly one backend at any time and routes
    ``transcribe_one`` / ``load`` / ``unload`` through it.
    """

    @property
    def is_loaded(self) -> bool: ...

    @property
    def device_name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def load(self, **kwargs) -> None: ...

    def transcribe(
        self,
        wav: np.ndarray,
        language: Optional[str],
        return_time_stamps: bool,
    ) -> ASRResult: ...

    def unload(self) -> None: ...
