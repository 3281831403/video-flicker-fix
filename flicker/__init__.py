"""Flicker correction package for steady-brightness video frames."""

from .correct import (
    apply_gain,
    correct_sequence,
    detect_cuts,
    frame_hist,
    frame_mean,
    global_gain,
    match_hist,
    validate,
)

__all__ = [
    "apply_gain",
    "correct_sequence",
    "detect_cuts",
    "frame_hist",
    "frame_mean",
    "global_gain",
    "match_hist",
    "validate",
]
