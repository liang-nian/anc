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

多通道分块频域 FxLMS。次级路径和控制滤波器都按块切开，每一块仍做重叠保留，过去的频谱留在延迟线上。笔记见 `doc/pffxlms.md`。图写到 `output/sim/07_pffxlms.png`，库在 `output/pffxlms/`。

```bash
python3 sim/pffxlms_sim.py
```

单通道卡尔曼滤波主动降噪。把最优 FIR 系数和次级路径状态放进同一个协方差卡尔曼，每次送入一个采样点。笔记见 `doc/kfanc.md`。图写到 `output/sim/08_kfanc.png`，库在 `output/kfanc/`。

```bash
python3 sim/kfanc_sim.py
```

多通道 RLS，没有次级路径。每个误差通道各有一套协方差，每次送入一个采样点。笔记见 `doc/rls.md`。图写到 `output/sim/09_rls.png`，库在 `output/rls/`。

```bash
python3 sim/rls_sim.py
```

多通道滤波 RLS。参考经次级路径之后做协方差递推，扬声器仍乘原始参考。笔记见 `doc/frls.md`。图写到 `output/sim/10_frls.png`，库在 `output/frls/`。

```bash
python3 sim/frls_sim.py
```

多通道快速阵列 RLS。外环和滤波 RLS 相同，协方差换成每路参考秩 2 的生成元。笔记见 `doc/farls.md`。图写到 `output/sim/11_farls.png`，库在 `output/farls/`。

```bash
python3 sim/farls_sim.py
```

多通道快速阵列卡尔曼。每只扬声器的状态是控制系数和次级延迟线，协方差用同一套低秩旋转往前推。笔记见 `doc/fakf.md`。图写到 `output/sim/12_fakf.png`，库在 `output/fakf/`。

```bash
python3 sim/fakf_sim.py
```

同一条路径上的对比。次级路径前 8 点为 0，末两点是舱室响应，这是频域一块的最长延时。控制长度 8，块长 8，种子 7，16000 点。在线算法用 `doc/fakf.md` 第 4.2 节的参数。维纳滤波整段一次求解，只进性能，不进第 4.3 节的耗时。耗时只包 `Core`，去掉开头 64 点；频域一次处理 8 点，耗时除以 8。下降曲线写到 `output/sim/13_cmp_perf.png`，耗时写到 `output/sim/13_cmp_time.png`。笔记里的记录图在 `doc/image/`。

```bash
python3 sim/cmp_sim.py
```
