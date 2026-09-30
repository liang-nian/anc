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
