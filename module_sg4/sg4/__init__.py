"""SG-4 temporal behaviour analysis for driver drowsiness monitoring."""
from .analyzer import MEASUREMENT_KEYS, OUTPUT_KEYS, TemporalAnalyzer, run_stream
from .closure import ClosureRun, ClosureTracker
from .config import SG4Config
from .io_utils import (flatten_packet, nest_packet, normalize_record,
                       read_jsonl, write_jsonl)
from .yawn import YawnTracker

__all__ = [
    "MEASUREMENT_KEYS", "OUTPUT_KEYS", "TemporalAnalyzer", "run_stream",
    "ClosureRun", "ClosureTracker", "SG4Config", "YawnTracker",
    "flatten_packet", "nest_packet", "normalize_record",
    "read_jsonl", "write_jsonl",
]
__version__ = "0.1.0"
