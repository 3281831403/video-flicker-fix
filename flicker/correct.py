"""Flicker correction primitives for steady-brightness video sequences.

A frame is a non-empty rectangular ``list[list[float]]`` of linear-light
intensity values in ``[0.0, 1.0]``.  Functions return new matrices and never
mutate their inputs.  Tests compare values with an absolute tolerance of
``1e-9`` unless a docstring states a wider bound.
"""

import math

HIST_TOTAL = 1000
DEFAULT_CUT_THRESHOLD = 0.25
GAIN_MIN = 0.5
GAIN_MAX = 2.0
GAIN_MEAN_TOL = 0.01


def _median(values: list[float]) -> float:
    """Return the median of a non-empty sequence (mean of middles if even)."""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bin_index(value: float, bins: int) -> int:
    """Bin ``value`` uniformly over ``[0, 1]``; last bin closed on the right."""
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    index = int(value * bins)
    if index >= bins:
        index = bins - 1
    if index < 0:
        index = 0
    return index


def _relative_change(current: float, previous: float) -> float:
    """Return ``abs(current - previous) / previous`` without dividing by zero."""
    if previous == 0.0:
        return 0.0 if current == 0.0 else math.inf
    return abs(current - previous) / abs(previous)


def frame_mean(frame: list[list[float]]) -> float:
    """Return the arithmetic mean of all pixels in ``frame``.

    ``frame`` is a non-empty rectangular matrix of floats.  Values are linear
    light and are not clamped or otherwise normalised before averaging.  The
    result is exact to ``1e-9``.  Inputs are not modified.
    """
    total = 0.0
    count = 0
    for row in frame:
        for value in row:
            total += value
            count += 1
    return total / count


def frame_hist(frame: list[list[float]], bins: int) -> list[int]:
    """Return a resolution-independent histogram of ``frame``.

    ``frame`` is a non-empty rectangular matrix and ``bins`` a positive
    integer.  Pixel values are clamped to ``[0, 1]`` and binned uniformly over
    that range: bin ``b`` covers ``[b / bins, (b + 1) / bins)`` and the last
    bin is closed on the right.  Raw counts are then normalised to exactly
    ``HIST_TOTAL`` (1000) units using largest-remainder apportionment, with
    ties resolved towards the lower bin index.

    Therefore two frames with the same intensity proportions but different
    pixel counts return identical histograms, and histogram matching can
    never depend on image resolution.  Return a list of ``bins`` integers
    summing to ``HIST_TOTAL``.
    """
    counts = [0] * bins
    for row in frame:
        for value in row:
            counts[_bin_index(value, bins)] += 1

    total = sum(counts)
    quotas = [count * HIST_TOTAL / total for count in counts]
    result = [int(quota) for quota in quotas]

    remaining = HIST_TOTAL - sum(result)
    if remaining > 0:
        # Largest fractional part first; ties towards the lower bin index.
        order = sorted(
            range(bins),
            key=lambda index: (-(quotas[index] - int(quotas[index])), index),
        )
        for index in order[:remaining]:
            result[index] += 1
    return result


def global_gain(means: list[float], window: int) -> list[float]:
    """Return one multiplicative correction gain per frame.

    ``means`` is a non-empty sequence of positive per-frame mean brightness
    values and ``window`` is a positive integer.  Let ``half = window // 2``;
    the symmetric window around frame ``i`` spans ``i - half .. i + half``
    including ``i`` itself (for an even ``window`` that is ``window + 1``
    frames).  The local level for frame ``i`` is the median of the *other*
    frames in that window, i.e. the ``2 * half`` samples with ``j != i``.  The
    current frame is deliberately excluded: its own flicker must not pull the
    level it is compared against, otherwise the gain tracks the flicker
    instead of cancelling it.

    When the full symmetric window is unavailable -- the first and last
    ``half`` frames, or ``window == 1`` -- the local level is the median of
    the complete ``means`` sequence (the robust sequence baseline).  This
    keeps the window phase-symmetric, with no half-window lag for an even
    ``window``, and avoids one-sided boundary estimates.

    ``gain[i] = local_level / means[i]``.  The median of an even number of
    values is the arithmetic mean of the two middle values.  Gains are *not*
    renormalised: energy drift is reported by :func:`validate` instead of
    being silently hidden.  Return a new list of floats.
    """
    count = len(means)
    half = window // 2
    baseline = _median(means)
    gains: list[float] = []

    for index in range(count):
        full_window = (
            window != 1
            and index - half >= 0
            and index + half < count
        )
        if full_window:
            others = [
                means[j]
                for j in range(index - half, index + half + 1)
                if j != index
            ]
            local_level = _median(others)
        else:
            local_level = baseline
        gains.append(local_level / means[index])

    return gains


def apply_gain(frame: list[list[float]], gain: float) -> list[list[float]]:
    """Scale every pixel by ``gain`` in linear light and clamp to ``[0, 1]``.

    Each result cell is ``min(1.0, max(0.0, value * gain))``; ``gain`` is a
    non-negative real number.  Scaling happens before clipping and applies to
    every pixel, including pixels already at ``1.0``: clamping is the final
    step, so an over-exposed region stays monotone instead of developing a
    hard boundary between scaled and untouched highlights.

    Where the scaled value remains inside ``[0, 1]`` the operation is exactly
    reversible with ``1.0 / gain`` (absolute tolerance ``1e-9``).  Values are
    linear light, not sRGB, so the brightness change is proportional to
    ``gain``.  Return a matrix of the same shape.
    """
    return [
        [min(1.0, max(0.0, value * gain)) for value in row]
        for row in frame
    ]


def match_hist(frame: list[list[float]], ref_hist: list[int]) -> list[list[float]]:
    """Histogram-match ``frame`` to the reference histogram ``ref_hist``.

    ``ref_hist`` is a non-empty sequence of non-negative integer counts whose
    length ``b`` is the number of uniform bins over ``[0, 1]``; its total may
    be any positive value (``frame_hist`` returns totals of 1000).  Source
    counts use the same clamping and binning rules as :func:`frame_hist`.
    Let ``s[c]`` be the source cumulative probability through bin ``c`` and
    ``r[t]`` the reference cumulative probability through bin ``t``.  Every
    source pixel in bin ``c`` is mapped to the centre ``(t + 0.5) / b`` of the
    first reference bin ``t`` with ``r[t] >= s[c]``.

    The map is non-decreasing in intensity (order preserving, with no tonal
    inversion) and depends only on intensity proportions -- never on the
    number of source pixels -- so matching the same scene at two resolutions
    gives the same result.  Return a matrix of the same shape.
    """
    bins = len(ref_hist)
    counts = [0] * bins
    for row in frame:
        for value in row:
            counts[_bin_index(value, bins)] += 1

    source_total = sum(counts)
    ref_total = sum(ref_hist)

    source_cumulative = []
    running = 0
    for count in counts:
        running += count
        source_cumulative.append(running)

    ref_cumulative = []
    running = 0
    for count in ref_hist:
        running += count
        ref_cumulative.append(running)

    # Integer cross-multiplication keeps the mapping independent of the
    # source resolution: r[t] >= s[c]  <=>  ref_cum[t]*N >= src_cum[c]*R.
    mapping = [0.0] * bins
    target_bin = 0
    for source_bin in range(bins):
        target = source_cumulative[source_bin] * ref_total
        while (
            target_bin < bins - 1
            and ref_cumulative[target_bin] * source_total < target
        ):
            target_bin += 1
        mapping[source_bin] = (target_bin + 0.5) / bins

    return [
        [mapping[_bin_index(value, bins)] for value in row]
        for row in frame
    ]


def detect_cuts(means: list[float], threshold: float) -> list[int]:
    """Return the indices at which a new shot starts.

    A cut at ``i`` (``1 <= i < len(means)``) exists when the relative change
    ``abs(means[i] - means[i - 1]) / means[i - 1]`` is strictly greater than
    ``threshold``.  ``means`` is non-empty and positive; ``threshold`` is a
    non-negative fraction.  Relative changes are used so that a cut in a dark
    scene (small absolute difference) is detected as reliably as one in a
    bright scene.

    Return ascending indices; ``[]`` when there is no cut.  Inputs are not
    modified.
    """
    cuts = []
    for index in range(1, len(means)):
        if _relative_change(means[index], means[index - 1]) > threshold:
            cuts.append(index)
    return cuts


def correct_sequence(
    frames: list[list[list[float]]], config: dict[str, object]
) -> list[list[list[float]]]:
    """Correct a whole sequence, resetting the gain estimate at every cut.

    ``config`` carries:

    * ``window`` (int >= 1): symmetric median window passed to
      :func:`global_gain`;
    * ``cut_threshold`` (float >= 0): relative threshold passed to
      :func:`detect_cuts`.

    Compute the per-frame means, split ``frames`` at every detected shot
    start, then for each shot call :func:`global_gain` on that shot's means
    and :func:`apply_gain` to each of its frames with the matching gain.
    Gain estimates are never shared across a cut, so two shots of different
    brightness keep their own levels instead of being pulled towards the same
    value.

    Return the corrected frames in input order and input shape.  Inputs are
    not modified.
    """
    window = int(config["window"])
    cut_threshold = float(config["cut_threshold"])

    means = [frame_mean(frame) for frame in frames]
    boundaries = [0] + detect_cuts(means, cut_threshold) + [len(frames)]

    corrected: list[list[list[float]]] = []
    for shot in range(len(boundaries) - 1):
        start = boundaries[shot]
        stop = boundaries[shot + 1]
        gains = global_gain(means[start:stop], window)
        for offset, frame in enumerate(frames[start:stop]):
            corrected.append(apply_gain(frame, gains[offset]))
    return corrected


def validate(frames: list[list[list[float]]], gains: list[object]) -> list[str]:
    """Return human-readable problems with a gain sequence; ``[]`` is valid.

    ``gains`` has one entry per frame: a float gain, or ``None`` marking the
    first frame of a new shot (a cut marker).  Report:

    * a length mismatch (checked first);
    * a gain that is not a finite real number, or is outside
      ``[GAIN_MIN, GAIN_MAX]`` = ``[0.5, 2.0]``;
    * a numeric-gain arithmetic mean more than ``GAIN_MEAN_TOL`` (0.01) away
      from ``1.0`` -- the energy-conservation check that must report a
      globally dark or bright correction;
    * a cut marker at frame 0, or at a frame whose relative mean change is
      not greater than ``DEFAULT_CUT_THRESHOLD`` (0.25); and a frame whose
      relative mean change exceeds that threshold but carries no marker.

    Each message names its check with one of the words ``length``, ``range``,
    ``mean`` or ``cut``.  This function reports problems instead of raising.
    Inputs are not modified.
    """
    problems: list[str] = []

    if len(gains) != len(frames):
        problems.append(
            f"length: expected {len(frames)} gains, got {len(gains)}"
        )
        return problems

    means = [frame_mean(frame) for frame in frames]
    numeric_gains: list[float] = []

    for index, gain in enumerate(gains):
        if gain is None:
            continue
        if (
            isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(gain)
        ):
            problems.append(f"range: gain at frame {index} is not a finite number")
            continue
        value = float(gain)
        numeric_gains.append(value)
        if value < GAIN_MIN or value > GAIN_MAX:
            problems.append(
                f"range: gain {value!r} at frame {index} outside "
                f"[{GAIN_MIN}, {GAIN_MAX}]"
            )

    if numeric_gains:
        average = sum(numeric_gains) / len(numeric_gains)
        if abs(average - 1.0) > GAIN_MEAN_TOL:
            problems.append(
                f"mean: gain sequence mean {average!r} deviates from 1.0 "
                f"by more than {GAIN_MEAN_TOL}"
            )

    for index, gain in enumerate(gains):
        change = (
            _relative_change(means[index], means[index - 1])
            if index > 0
            else None
        )
        if gain is None:
            if index == 0:
                problems.append("cut: unexpected marker at frame 0")
            elif change is not None and change <= DEFAULT_CUT_THRESHOLD:
                problems.append(
                    f"cut: marker at frame {index} does not follow a shot "
                    f"change (relative change {change!r})"
                )
        elif change is not None and change > DEFAULT_CUT_THRESHOLD:
            problems.append(
                f"cut: missing marker at frame {index} "
                f"(relative change {change!r})"
            )

    return problems
