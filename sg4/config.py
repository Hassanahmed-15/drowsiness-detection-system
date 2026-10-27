"""Configuration for the SG-4 temporal analyzer.

Every threshold used by the analysis logic lives in :class:`SG4Config`.
The defaults are **unvalidated placeholders** (see README); tune them from
labelled recordings before drawing any conclusions.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Union

PathLike = Union[str, Path]


def _is_number(value: Any) -> bool:
    """Return True for finite int/float values (bool is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


@dataclass(frozen=True)
class SG4Config:
    """All tunable parameters of SG-4 (placeholder defaults)."""

    window_ms: int = 30000
    warmup_ms: int = 5000
    min_valid_fraction: float = 0.6
    max_gap_ms: int = 200
    t_blink_min_ms: int = 50
    t_blink_max_ms: int = 500
    yawn_smooth_n: int = 3
    yawn_on_threshold: float = 0.6
    yawn_off_threshold: float = 0.4
    yawn_min_ms: int = 1500
    yawn_refractory_ms: int = 3000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if any parameter is out of range."""
        for f in fields(self):
            if not _is_number(getattr(self, f.name)):
                raise ValueError(f"config '{f.name}' must be a finite number, "
                                 f"got {getattr(self, f.name)!r}")
        if self.window_ms <= 0:
            raise ValueError("window_ms must be > 0")
        if self.warmup_ms < 0:
            raise ValueError("warmup_ms must be >= 0")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        if self.max_gap_ms < 0:
            raise ValueError("max_gap_ms must be >= 0")
        if not 0 <= self.t_blink_min_ms <= self.t_blink_max_ms:
            raise ValueError("need 0 <= t_blink_min_ms <= t_blink_max_ms")
        if int(self.yawn_smooth_n) != self.yawn_smooth_n or self.yawn_smooth_n < 1:
            raise ValueError("yawn_smooth_n must be an integer >= 1")
        if not 0.0 <= self.yawn_off_threshold <= self.yawn_on_threshold <= 1.0:
            raise ValueError("need 0 <= yawn_off_threshold <= yawn_on_threshold <= 1")
        if self.yawn_min_ms < 0 or self.yawn_refractory_ms < 0:
            raise ValueError("yawn_min_ms and yawn_refractory_ms must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SG4Config":
        """Build a config from a mapping. Unknown keys raise ``ValueError``
        (a typo in a threshold name must not be silently ignored)."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown SG4 config keys: {unknown}")
        return cls(**dict(data))

    def to_dict(self) -> Dict[str, Any]:
        """Return the config as a plain dict."""
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> "SG4Config":
        """Return a copy with some fields changed (validated)."""
        known = {f.name for f in fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown SG4 config keys: {unknown}")
        return replace(self, **overrides)

    @classmethod
    def load(cls, path: PathLike) -> "SG4Config":
        """Load from a ``.json`` file, or ``.yaml``/``.yml`` if PyYAML exists."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError("PyYAML is required to read YAML configs; "
                                   "use JSON or `pip install pyyaml`") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"config file {p} must contain a mapping")
        return cls.from_dict(data)

    def save(self, path: PathLike) -> None:
        """Write the config to JSON (or YAML if the suffix says so)."""
        p = Path(path)
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError("PyYAML is required to write YAML configs") from exc
            p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False),
                         encoding="utf-8")
        else:
            p.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                         encoding="utf-8")
