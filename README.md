# ANC

主动降噪的算法笔记和仿真。笔记在 `doc/`，C 实现在各自的目录里，仿真脚本在 `sim/`。脚本会自己用 CMake 编译对应的库，再出图。

依赖：CMake 3.20、C 编译器、Python 3、NumPy、Matplotlib。在仓库根目录执行。

## 仿真

维纳滤波，整段数据一次求解。笔记见 `doc/wiener.md`。图写到 `output/sim/01_signals.png`，库在 `output/wiener/`。

```bash
python3 sim/wiener_sim.py
```

多通道 LMS，每次送入一个采样点。笔记见 `doc/lms.md`。图写到 `output/sim/02_lms.png`，库在 `output/lms/`。

```bash
python3 sim/lms_sim.py
```

多通道 FxLMS，每次送入一个采样点。`FxlmsCore` 先对参考做预滤波，再用当前误差和滤波后参考更新，最后算扬声器输出。笔记见 `doc/fxlms.md`。图写到 `output/sim/03_fxlms.png`，库在 `output/fxlms/`。

```bash
python3 sim/fxlms_sim.py
```

多通道窄带 FxLMS。阶次在构造时给定，基频每个采样点送入，库内合成各阶次的余弦和正弦，再用当前误差更新。笔记见 `doc/nbfxlms.md`。图写到 `output/sim/05_nbfxlms.png`，库在 `output/nbfxlms/`。

```bash
python3 sim/nbfxlms_sim.py
```

多通道频域 FxLMS。每次送入一块参考和一块误差，在频域做次级路径滤波、相关更新和扬声器输出。笔记见 `doc/fdfxlms.md`。图写到 `output/sim/06_fdfxlms.png`，库在 `output/fdfxlms/`。

```bash
python3 sim/fdfxlms_sim.py
```
