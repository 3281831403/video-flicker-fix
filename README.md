# video-flicker-fix

## 题面

老胶片去闪烁后，画面反而更闪、两个镜头还被拉成同一亮度：① `global_gain` 用当前帧均值做基准（应该用序列的稳健基准），导致整体亮度被归一化到 0.5，画面忽明忽暗；② 滑动窗口用包含当前帧的均值，增益与输入负相关，闪烁被放大而不是抑制；③ `match_hist` 的映射不单调（按 bin 索引直接赋值），出现色阶反转；④ `apply_gain` 对已钳制的高光继续乘，导致过曝区边界出现硬边（可逆性断言失败）；⑤ `detect_cuts` 用相邻均值差但阈值是绝对值，暗场景切分全部漏检；⑥ `correct_sequence` 在切分处不重置，跨镜头的增益把两个镜头往同一亮度拉；⑦ `global_gain` 的窗口长度为偶数时取 `window//2` 而不是对称窗口，增益整体滞后半窗（相位误差断言）；⑧ `frame_hist` 不归一化，直方图匹配的映射依赖帧像素数（不同分辨率结果不同）；⑨ `apply_gain` 的增益应用在 sRGB 域而不是线性域，亮度校正量与增益不成正比（单调性断言）；⑩ `validate` 不检查“增益序列的均值应为 1”（能量守恒），整体偏暗的序列不被报告。flicker/correct.py 是核心，tests/test_correct.py 15个红。要修的是：`frame_mean`、`frame_hist`、`global_gain`、`apply_gain`、`match_hist`、`detect_cuts`、`correct_sequence`、`validate` 八个函数的实现，全部契约见 docstring。

陷阱在多帧统计：基准要用稳健量、窗口要对称、镜头边界必须重置。

## 运行方式

```powershell
cd D:\gsb-20261119\video-flicker-fix
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败都来自 `flicker/correct.py` 中 8 个桩的 `NotImplementedError`。修复核心模块后测试应转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_correct.py` 是评测基准，**不可修改**；只允许修改 `flicker/correct.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数，不调用 `sleep`。帧为 `[0, 1]` 线性光二维浮点矩阵，所有浮点断言按 `1e-9` 绝对容差（docstring 另行说明的除外）。
