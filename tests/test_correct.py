"""video-flicker-fix 的红色基线测试。

10 个测试分别覆盖题面列出的 10 个多帧统计一致性缺陷，另有 5 个组合/
边界测试。初始状态下所有测试都调用尚未实现的 flicker.correct 桩，因此
应恰好 15 failed，且没有 collection/import/syntax error。
"""

import pytest

from flicker.correct import (
    apply_gain,
    correct_sequence,
    detect_cuts,
    frame_hist,
    frame_mean,
    global_gain,
    match_hist,
    validate,
)

TOL = 1e-9


def _assert_close(actual, expected, tolerance=TOL):
    assert actual == pytest.approx(expected, abs=tolerance)


def _flatten(frame):
    return [value for row in frame for value in row]


def _brightness_sequence(means):
    return [[[value]] for value in means]


def _corrected_means(frames, config):
    return [frame_mean(frame) for frame in correct_sequence(frames, config)]


def test_global_gain_uses_robust_sequence_baseline():
    """缺陷 1：用当前帧均值或固定 0.5 做基准，整体亮度被归一化，画面忽明忽暗。"""
    means = [0.30, 0.30, 0.30, 0.30, 0.36, 0.30, 0.30, 0.30, 0.30]

    gains = global_gain(means, window=4)

    assert len(gains) == len(means)
    corrected = [mean * gain for mean, gain in zip(means, gains)]
    for value in corrected:
        _assert_close(value, 0.30)


def test_global_gain_window_excludes_current_frame():
    """缺陷 2：窗口包含当前帧，增益与闪烁同相，闪烁被放大而不是抑制。"""
    means = [
        0.5 * (1.0 + 0.10 * (1 if index % 2 == 0 else -1))
        for index in range(8)
    ]

    gains = global_gain(means, window=4)

    corrected = [mean * gain for mean, gain in zip(means, gains)]
    assert max(corrected) - min(corrected) <= 0.005
    for value in corrected:
        _assert_close(value, 0.5, tolerance=0.005)


def test_match_hist_mapping_is_monotone():
    """缺陷 3：按 bin 索引直接赋值，映射不单调，出现色阶反转。"""
    source = [
        [0.05, 0.15, 0.25, 0.35, 0.45],
        [0.55, 0.65, 0.75, 0.85, 0.95],
    ]
    reference = [[0.90, 0.10, 0.10, 0.10, 0.70, 0.70, 0.70, 0.70]]

    matched = match_hist(source, frame_hist(reference, bins=8))

    values = _flatten(matched)
    for index in range(len(values) - 1):
        assert values[index] <= values[index + 1] + TOL, (
            f"tonal inversion at {index}: {values}"
        )
    for value in values:
        assert 0.0 <= value <= 1.0


def test_apply_gain_clipping_is_reversible_below_saturation():
    """缺陷 4：对已钳制的高光继续乘，过曝区边界出现硬边，可逆性断言失败。"""
    frame = [[0.20, 0.30, 0.45, 0.90]]

    scaled = apply_gain(frame, 1.5)
    _assert_close(scaled[0][0], 0.30)
    _assert_close(scaled[0][1], 0.45)
    _assert_close(scaled[0][2], 0.675)
    _assert_close(scaled[0][3], 1.0)
    for index in range(len(scaled[0]) - 1):
        assert scaled[0][index] <= scaled[0][index + 1] + TOL

    restored = apply_gain(scaled, 1.0 / 1.5)
    _assert_close(restored[0][0], 0.20)
    _assert_close(restored[0][1], 0.30)
    _assert_close(restored[0][2], 0.45)
    # Saturated highlight cannot be recovered, but it must be scaled back down.
    _assert_close(restored[0][3], 1.0 / 1.5)


def test_detect_cuts_uses_relative_change_in_dark_scenes():
    """缺陷 5：相邻均值差用绝对阈值，暗场景切分全部漏检。"""
    dark = [0.02] * 5 + [0.05] * 5
    bright = [0.80] * 5 + [0.81] * 5
    two_cuts = [0.02] * 3 + [0.05] * 3 + [0.02] * 3

    assert detect_cuts(dark, threshold=0.2) == [5]
    assert detect_cuts(bright, threshold=0.2) == []
    assert detect_cuts(two_cuts, threshold=0.2) == [3, 6]


def test_correct_sequence_resets_gain_at_shot_boundaries():
    """缺陷 6：切分处不重置，跨镜头增益把两个镜头往同一亮度拉。"""
    means = []
    for base in (0.2, 0.6):
        for index in range(6):
            means.append(
                base * (1.0 + 0.05 * (1 if index % 2 == 0 else -1))
            )
    frames = _brightness_sequence(means)

    corrected = _corrected_means(
        frames, {"window": 4, "cut_threshold": 0.2}
    )

    assert len(corrected) == len(frames)
    for value in corrected[:6]:
        _assert_close(value, 0.2)
    for value in corrected[6:]:
        _assert_close(value, 0.6)


def test_global_gain_even_window_has_no_phase_lag():
    """缺陷 7：偶数窗口取 window//2 导致增益滞后半窗，相位对不齐。"""
    means = [0.50, 0.54, 0.58, 0.62, 0.66, 0.62, 0.58, 0.54, 0.50]

    gains = global_gain(means, window=4)
    corrected = [mean * gain for mean, gain in zip(means, gains)]

    for index in range(len(means)):
        _assert_close(corrected[index], corrected[len(means) - 1 - index])
    centroid = sum(index * value for index, value in enumerate(corrected)) / sum(
        corrected
    )
    _assert_close(centroid, 4.0)


def test_frame_hist_is_resolution_independent():
    """缺陷 8：直方图不归一化，匹配映射依赖像素数，换分辨率结果就变。"""
    small = [[0.10, 0.35], [0.60, 0.85]]
    big = [
        [small[row // 2][column // 2] for column in range(4)]
        for row in range(4)
    ]
    reference = [
        [0.0625, 0.3125, 0.3125, 0.5625],
        [0.8125, 0.8125, 0.8125, 0.9375],
    ]

    small_hist = frame_hist(small, bins=4)
    big_hist = frame_hist(big, bins=4)
    assert small_hist == [250, 250, 250, 250]
    assert big_hist == small_hist
    assert sum(small_hist) == 1000

    ref_hist = frame_hist(reference, bins=8)
    small_match = match_hist(small, ref_hist)
    big_match = match_hist(big, ref_hist)
    for row in range(4):
        for column in range(4):
            _assert_close(
                big_match[row][column],
                small_match[row // 2][column // 2],
            )


def test_apply_gain_works_in_linear_light():
    """缺陷 9：在 sRGB 域乘增益，亮度校正量与增益不成正比，单调性断言失败。"""
    frame = [[0.05, 0.10, 0.20, 0.40]]

    down = apply_gain(frame, 0.5)
    _assert_close(_flatten(down), [0.025, 0.05, 0.10, 0.20])

    up = apply_gain(frame, 2.0)
    _assert_close(_flatten(up), [0.10, 0.20, 0.40, 0.80])
    _assert_close(sum(_flatten(up)) / 4.0, 2.0 * frame_mean(frame))

    restored = apply_gain(apply_gain(frame, 0.5), 2.0)
    _assert_close(_flatten(restored), _flatten(frame))


def test_validate_reports_gain_energy_drift():
    """缺陷 10：不检查增益序列均值为 1，整体偏暗的序列不被报告。"""
    frames = [[[0.30, 0.30]] for _ in range(5)]

    assert validate(frames, [1.0] * 5) == []

    underexposed = validate(frames, [0.80] * 5)
    assert underexposed
    assert any("mean" in problem.lower() for problem in underexposed)

    length_problems = validate(frames, [1.0] * 4)
    assert any("length" in problem.lower() for problem in length_problems)

    range_problems = validate(frames, [1.0, 1.0, 3.0, 1.0, 1.0])
    assert any("range" in problem.lower() for problem in range_problems)

    cut_frames = [[[0.20]] for _ in range(3)] + [[[0.60]] for _ in range(3)]
    unmarked = validate(cut_frames, [1.0] * 6)
    assert any("cut" in problem.lower() for problem in unmarked)
    marked = validate(cut_frames, [1.0, 1.0, 1.0, None, 1.0, 1.0])
    assert marked == []


def test_constant_brightness_sequence_is_unchanged():
    """组合：恒定亮度的序列，校正后亮度不变（±1e-9）。"""
    frames = [[[0.42]] for _ in range(8)]

    corrected = _corrected_means(
        frames, {"window": 4, "cut_threshold": 0.2}
    )

    for value in corrected:
        _assert_close(value, 0.42)


def test_five_percent_flicker_is_suppressed_below_one_percent():
    """组合：人为 ±5% 闪烁被压到 ±1% 以内。"""
    means = [
        0.5 * (1.0 + 0.05 * (1 if index % 2 == 0 else -1))
        for index in range(8)
    ]
    frames = _brightness_sequence(means)

    corrected = _corrected_means(
        frames, {"window": 4, "cut_threshold": 0.2}
    )

    assert max(corrected) - min(corrected) <= 0.005
    for value in corrected:
        assert abs(value - 0.5) <= 0.005


def test_two_shots_keep_their_relative_brightness():
    """组合：两个不同亮度的镜头校正后仍保持各自相对亮度（不被拉平）。"""
    means = [0.2] * 5 + [0.6] * 5
    frames = _brightness_sequence(means)

    corrected = _corrected_means(
        frames, {"window": 4, "cut_threshold": 0.2}
    )

    first = sum(corrected[:5]) / 5.0
    second = sum(corrected[5:]) / 5.0
    _assert_close(first, 0.2)
    _assert_close(second, 0.6)
    _assert_close(second / first, 3.0)


def test_histogram_match_l1_distance_below_0_05():
    """组合：直方图匹配后目标分布与参考的 L1 距离 < 0.05。"""
    source = [
        [0.0625, 0.1875, 0.3125, 0.4375],
        [0.5625, 0.6875, 0.8125, 0.9375],
    ]
    reference = [
        [0.0625, 0.3125, 0.3125, 0.5625],
        [0.8125, 0.8125, 0.8125, 0.9375],
    ]

    ref_hist = frame_hist(reference, bins=8)
    matched = match_hist(source, ref_hist)
    matched_hist = frame_hist(matched, bins=8)

    ref_total = sum(ref_hist)
    l1 = sum(
        abs(matched_count / 1000.0 - ref_count / ref_total)
        for matched_count, ref_count in zip(matched_hist, ref_hist)
    )
    assert l1 < 0.05


def test_gain_sequence_mean_is_one():
    """组合：增益序列均值 ≈ 1（±0.01）。"""
    means = [
        0.5 * (1.0 + 0.05 * (1 if index % 2 == 0 else -1))
        for index in range(8)
    ]

    gains = global_gain(means, window=4)

    average = sum(gains) / len(gains)
    _assert_close(average, 1.0, tolerance=0.01)
