# 频域 FxLMS

在很多场景时域算法已经可以达到较为理想的效果，但是时域算法无法实现对特定频段的精确控制(这里不包括窄带算法，窄带算法可以实现逐频点的控制)，当我们存在宽带降噪需求，同时又需要针对特定频段进行控制优化时，就需要频域 Fx-LMS 算法。本章的模型如下：误差传感器和参考传感器采集到 1 块信号，输入算法后计算得到 1 块扬声器输出，经过次级路径传播后到达误差传感器处执行降噪。

ps:公式推导只是按照自己的理解进行简化版的推导，也许并不严谨。

## 1. 数学模型

时域核心算法实际上由 3 个部分组成，分别是参考信号预滤波，滤波器更新和输出计算。这三个公式都属于相关或者卷积计算，均可以使用 OLS 转到频域进行，这就是频域算法的理论基础。

假设当前模型为单通道，比较特殊的是我们假设控制滤波器和次级路径长度均为 $M$，每次送入 $M$ 个新的参考和 $M$ 个误差。

当前送入的参考块、误差块记为 $\mathbf{x}_{\mathrm{blk}}$、$\mathbf{e}_{\mathrm{blk}}$，上一块留下的参考、滤波后参考记为 $\mathbf{x}_{\mathrm{old}}$、$\mathbf{x}'_{\mathrm{old}}$，补 0 的一块记为 $\mathbf{0}$，这些块长度都是 $M$。频域帧长度 $N=2M$，前一块占 $0\le t<M$，后一块占 $M\le t<N$。

参考经次级路径得到滤波后参考

$$
x'(n)=\mathbf{s}(n)*x(n)
$$

这是卷积公式，转到频域使用 OLS 法求解。先拼接得到 OLS 解卷积所需的两块数据，参考帧的拼接是前一块放 $\mathbf{x}_{\mathrm{old}}$，后一块放 $\mathbf{x}_{\mathrm{blk}}$。次级路径放在前一块开头，后一块整块是补 0 块

$$
\begin{aligned}
x_{\mathrm{frame}}[t]
&=
\begin{cases}
x_{\mathrm{old}}[t] & 0\le t<M\\
x_{\mathrm{blk}}[t-M] & M\le t<N
\end{cases}\\
s_{\mathrm{frame}}[t]
&=
\begin{cases}
s[t] & 0\le t<M\\
0 & M\le t<N
\end{cases}
\end{aligned}
$$

然后对两边做 DFT，频域相乘，得到频域结果。

$$
X_s(k)=S(k)\,X_{\mathrm{frame}}(k)
$$

最后需要逆变换回到时域，抛弃前半混叠块丢掉，后一块留下，记为 $\mathbf{x}'_{\mathrm{blk}}$

$$
\begin{aligned}
x'_{\mathrm{circ}}&=\mathrm{IDFT}(X_s)\\
x'_{\mathrm{blk}}[i]&=x'_{\mathrm{circ}}[M+i],\qquad i=0,\ldots,M-1
\end{aligned}
$$

滤波后的参考信号输入到 LMS 算法进行更新，更新项的计算是相关公式，同样基于 OLS 计算频域相关。同理先拼接所需的 2 块信号，前半放上一块留下的 $\mathbf{x}'_{\mathrm{old}}$，后半放刚计算出的 $\mathbf{x}'_{\mathrm{blk}}$，做 DFT 得到 $X'(k)$

$$
x'_{\mathrm{frame}}[t]=
\begin{cases}
x'_{\mathrm{old}}[t] & 0\le t<M\\
x'_{\mathrm{blk}}[t-M] & M\le t<N
\end{cases}
$$

误差的补充块和卷积计算时滤波器的补充块不同，由于相关和卷积存在共轭关系，误差帧的补 0 块放在前一块，后一块放 $\mathbf{e}_{\mathrm{blk}}$

$$
e_{\mathrm{frame}}[t]=
\begin{cases}
0 & 0\le t<M\\
e_{\mathrm{blk}}[t-M] & M\le t<N
\end{cases}
$$

同理频域相乘后逆变换回到时域。前半留下，就是 $g[m]$，后半置 0

$$
\begin{aligned}
G(k)&=E(k)\,X^{\prime *}(k)\\
g_{\mathrm{circ}}&=\mathrm{IDFT}(G)\\
g_{\mathrm{frame}}[t]
&=
\begin{cases}
g_{\mathrm{circ}}[t] & 0\le t<M\\
0 & M\le t<N
\end{cases}\\
\end{aligned}
$$

此时我们得到了更新项的时域表达，在 OLS 计算卷积的公式中，我们已经知道滤波器后半补0，所以实际上在频域计算中的自适应滤波器时域信号表示是

$$
h_{\mathrm{frame}}[t]=
\begin{cases}
w_t & 0\le t<M\\
0 & M\le t<N
\end{cases}\\
$$

因此在时域上会发生正确的更新，并且在补充块上的加减均为 0 不会产生混叠项，当然我们这里还是会在频域进行更新处理，滤波器本身也是频域的，更新公式为

$$
H(k) =  H(k)+\mu\,G_c(k)
$$

$G_c(k)$ 是 $g_{\mathrm{frame}}$ 的 DFT，$H(k)$ 是 $h_{\mathrm{frame}}$ 的 DFT，这里步长可以按照频点来给，这是实现频段优化的最基础方法。

最后一步是计算输出，扬声器信号是控制滤波器和参考信号的卷积

$$
y(n)=\mathbf{w}(n)*x(n)
$$

与参考信号滤波计算一致，前半 $\mathbf{x}_{\mathrm{old}}$，后一块 $\mathbf{x}_{\mathrm{blk}}$，所以直接复用参考信号滤波计算的过程量即可，系数用刚更新的 $H(k)$，频域相乘后逆变换回到时域，保留后半，就是这一块扬声器信号 $\mathbf{y}_{\mathrm{blk}}$

$$
\begin{aligned}
Y(k)&=H(k)\,X_{\mathrm{frame}}(k)\\
y_{\mathrm{circ}}&=\mathrm{IDFT}(Y)\\
y_{\mathrm{blk}}[i]&=y_{\mathrm{circ}}[M+i],\qquad i=0,\ldots,M-1
\end{aligned}
$$

多通道时参考有 $P$ 路，扬声器有 $L$ 路，误差有 $Q$ 路，$H_{lp}(k)$ 的第一个下标是扬声器，第二个是参考。

先做参考信号预滤波，第 $p$ 路参考的拼接是前半 $\mathbf{x}_{p,\mathrm{old}}$，后半 $\mathbf{x}_{p,\mathrm{blk}}$。次级路径前半为 $\mathbf{s}_{ql}$ ，后半补 0

$$
\begin{aligned}
x_{p,\mathrm{frame}}[t]
&=
\begin{cases}
x_{p,\mathrm{old}}[t] & 0\le t<M\\
x_{p,\mathrm{blk}}[t-M] & M\le t<N
\end{cases}\\
s_{ql,\mathrm{frame}}[t]
&=
\begin{cases}
s_{ql}[t] & 0\le t<M\\
0 & M\le t<N
\end{cases}
\end{aligned}
$$

频域相乘，逆变换回到时域，前半丢掉，后半留下，得到 $\mathbf{x}'_{qlp,\mathrm{blk}}$。再拼到滤波后参考帧上，前半是上一帧计算结果 $\mathbf{x}'_{qlp,\mathrm{old}}$，后半是刚计算出的这一块，做 DFT 得到 $X'_{qlp}(k)$，更新用的就是这份频谱

$$
\begin{aligned}
X_{s,qlp}(k)&=S_{ql}(k)\,X_{p,\mathrm{frame}}(k)\\
x'_{qlp,\mathrm{circ}}&=\mathrm{IDFT}(X_{s,qlp})\\
x'_{qlp,\mathrm{blk}}[i]&=x'_{qlp,\mathrm{circ}}[M+i]\\
x'_{qlp,\mathrm{frame}}[t]
&=
\begin{cases}
x'_{qlp,\mathrm{old}}[t] & 0\le t<M\\
x'_{qlp,\mathrm{blk}}[t-M] & M\le t<N
\end{cases}
\end{aligned}
$$

再做更新，第 $q$ 路误差的补 0 块放在前半，后半放 $\mathbf{e}_{q,\mathrm{blk}}$

$$
e_{q,\mathrm{frame}}[t]=
\begin{cases}
0 & 0\le t<M\\
e_{q,\mathrm{blk}}[t-M] & M\le t<N
\end{cases}
$$

同一条控制滤波器把各误差通道的相关加在一起，逆变换回到时域。前半留下，后半换成补 0 块，再做 DFT，乘 $\mu$ 加到 $H_{lp}(k)$。$H_{lp}(k)$ 对应的时域帧仍是前半 $\mathbf{w}_{lp}$，后半 $\mathbf{0}$

$$
\begin{aligned}
G_{lp}(k)
&=\sum_{q=0}^{Q-1}E_q(k)\,X_{qlp}^{\prime *}(k)\\
g_{lp,\mathrm{circ}}&=\mathrm{IDFT}(G_{lp})\\
g_{lp,\mathrm{frame}}[t]
&=
\begin{cases}
g_{lp,\mathrm{circ}}[t] & 0\le t<M\\
0 & M\le t<N
\end{cases}\\
H_{lp}(k)&\leftarrow H_{lp}(k)+\mu\,G_{c,lp}(k)
\end{aligned}
$$

最后做输出，第 $p$ 路参考复用预滤波那一帧，前半为上一帧旧参考，后半为当前参考。用刚更新的 $H_{lp}(k)$ 把各参考加到扬声器 $l$ 上，逆变换回到时域，前半丢掉，后半留下

$$
\begin{aligned}
Y_l(k)&=\sum_{p=0}^{P-1}H_{lp}(k)\,X_{p,\mathrm{frame}}(k)\\
y_{l,\mathrm{circ}}&=\mathrm{IDFT}(Y_l)\\
y_{l,\mathrm{blk}}[i]&=y_{l,\mathrm{circ}}[M+i],\qquad i=0,\ldots,M-1
\end{aligned}
$$

$\mathbf{y}_{l,\mathrm{blk}}$ 就是扬声器 $l$ 的当前块输出。

## 2. 实例

取一路参考、一路扬声器、一路误差，块长 $M=2$，DFT 长度 $N=4$，次级路径 $M_s=2$，$\mu=1/2$，先从时域计算参考信号预滤波

$$
\begin{aligned}
\mathbf{s}&=[1,\;1/2]^{\mathsf T}\\
x(n)&: 1,\;1\\
d(n)&: 1,\;1
\end{aligned}
\qquad n=0,1
$$

系数从 $\mathbf{w}=[0,\,0]^{\mathsf T}$ 开始，整块都用这组系数。还没有到来的历史补 0。

$n=0$，参考抽头是 $[1,\,0]^{\mathsf T}$，扬声器历史是 0：

$$
\begin{aligned}
y(0)&=0\\
y'(0)&=1\cdot 0+\frac{1}{2}\cdot 0=0\\
e(0)&=1-0=1\\
x'(0)&=1\cdot 1+\frac{1}{2}\cdot 0=1
\end{aligned}
$$

$n=1$，系数仍是 0，参考抽头是 $[1,\,1]^{\mathsf T}$：

$$
\begin{aligned}
y(1)&=0\\
y'(1)&=0\\
e(1)&=1\\
x'(1)&=1\cdot 1+\frac{1}{2}\cdot 1=\frac{3}{2}
\end{aligned}
$$

上面的 $x'(0)=1$、$x'(1)=3/2$ 是次级路径对参考做时域卷积的结果。同一块参考再用频域 OLS 算一遍。历史补 0，参考帧前一块是 $[0,\,0]$，后一块是这一块参考 $[1,\,1]$。次级路径放在前一块，后一块补 0

$$
\begin{aligned}
x_{\mathrm{frame}}&=[0,\;0,\;1,\;1]\\
s_{\mathrm{frame}}&=[1,\;1/2,\;0,\;0]
\end{aligned}
$$

两边的 DFT 是

$$
\begin{aligned}
X_{\mathrm{frame}}&=\left[2,\;-1+j,\;0,\;-1-j\right]\\
S&=\left[\frac{3}{2},\;1-\frac{1}{2}j,\;\frac{1}{2},\;1+\frac{1}{2}j\right]
\end{aligned}
$$

频域相乘，再逆变换回到时域

$$
\begin{aligned}
X_s&=\left[3,\;-\frac{1}{2}+\frac{3}{2}j,\;0,\;-\frac{1}{2}-\frac{3}{2}j\right]\\
x'_{\mathrm{circ}}&=\left[\frac{1}{2},\;0,\;1,\;\frac{3}{2}\right]
\end{aligned}
$$

前一块 $\left[\frac{1}{2},\,0\right]$ 丢掉。后一块留下

$$
\mathbf{x}'_{\mathrm{blk}}=\left[1,\;\frac{3}{2}\right]
$$

这和时域卷积的 $x'(0)=1$、$x'(1)=3/2$ 是同一组数。

滤波后参考再拼一帧，前一块补 0，后一块放刚留下的 $\mathbf{x}'_{\mathrm{blk}}$。误差帧前一块补 0，后一块放这一块误差 $[1,\,1]$

$$
\begin{aligned}
x'_{\mathrm{frame}}&=[0,\;0,\;1,\;3/2]\\
e_{\mathrm{frame}}&=[0,\;0,\;1,\;1]
\end{aligned}
$$

两边的 DFT、相乘后的频谱、逆变换回到时域的结果是

$$
\begin{aligned}
X'&=\left[\frac{5}{2},\;-1+\frac{3}{2}j,\;-\frac{1}{2},\;-1-\frac{3}{2}j\right]\\
E&=\left[2,\;-1+j,\;0,\;-1-j\right]\\
G&=\left[5,\;\frac{5}{2}+\frac{1}{2}j,\;0,\;\frac{5}{2}-\frac{1}{2}j\right]\\
g_{\mathrm{circ}}&=\left[\frac{5}{2},\;1,\;0,\;\frac{3}{2}\right]
\end{aligned}
$$

前一块 $\left[\frac{5}{2},\,1\right]$ 留下。它和直接做相关一致：$e(0)x'(0)+e(1)x'(1)=1+\frac{3}{2}=\frac{5}{2}$，$e(0)x'(-1)+e(1)x'(0)=1$。后一块置 0。$\mu=1/2$ 加到系数上

$$
\mathbf{w}
=\begin{bmatrix}0\\0\end{bmatrix}
+\frac{1}{2}\begin{bmatrix}5/2\\1\end{bmatrix}
=\begin{bmatrix}5/4\\1/2\end{bmatrix}
$$

## 3. 仿真

码仓：https://github.com/liang-nian/anc.git

### 3.1 函数对应的公式

`FdfxlmsCreate` 按参考路数 $P$、扬声器路数 $L$、误差路数 $Q$、块长 $M$、次级路径阶数 $M_s$ 和步长 $\mu$ 建状态，并把次级路径模型抄进来。块长必须是 2 的幂，因为我这里写的只支持 2 幂次傅立叶。控制滤波器、相关和扬声器输出的 DFT 长度是 $N=2M$。次级路径滤波的 DFT 长度 $N_s$ 是不小于 $M+M_s-1$ 的 2 的幂；这个长度短于 $N$ 时就取 $N$。次级路径补 0 后做 DFT，系数频谱、参考帧和滤波后参考的 $N$ 点历史都从 0 开始。

送进库的参考按下标 $pM+i$ 排，误差按下标 $qM+i$ 排，扬声器按下标 $lM+i$ 排。$i=0$ 是这一块里最旧的一点。次级路径模型按下标 $(qL+l)M_s+j$ 排，$j=0$ 是当前抽头。`FdfxlmsCopyWeights` 抄出的抽头按下标 $(lP+p)M+m$ 排，$m=0$ 是当前抽头。

`FdfxlmsCore` 一次处理一块，顺序是预滤波、更新、输出。

`FdfxlmsPrefilter` 把这一块参考接到长度为 $N_s$ 的历史后面，做 DFT，乘上 $S_{ql}(k)$，逆变换后留下最后 $M$ 点，再接到长度为 $N$ 的滤波后参考历史上。

`FdfxlmsUpdate` 用这一块误差和刚接好的滤波后参考做第 1 节的相关。各误差通道的频谱先相加，逆变换后把 $m\ge M$ 的点置 0，再变回频域，乘 $\mu$ 加到 $H_{lp}(k)$

$$
H_{lp}(k)\leftarrow H_{lp}(k)+\mu\,G_{c,lp}(k)
$$

`FdfxlmsOutput` 取参考历史的最后 $N$ 点做 DFT，乘上更新后的 $H_{lp}(k)$，逆变换留下最后 $M$ 点，写入扬声器这一块。

仿真脚本在库外先按已经播出的扬声器历史测出这一块误差，再调用 `FdfxlmsCore`。次级路径前 $M$ 点为 0，这一块误差由这些已经播出的样本决定。扬声器历史为 0 时，误差就是目标信号本身。`FdfxlmsCopyWeights` 把 $H_{lp}(k)$ 逆变换，抄出前 $M$ 个实部。`FdfxlmsDestroy` 释放状态。

### 3.2 仿真设计

采样率 $f_s=2000\,\mathrm{Hz}$，$N_{\mathrm{sim}}=16000$，随机数种子为 7。参考 $P=2$ 路，扬声器 $L=2$ 路，误差 $Q=2$ 路。块长 $M=8$，控制滤波器的 DFT 长度是 16，次级路径的 DFT 长度是 32。次级路径阶数 $M_s=10$，步长 $\mu=0.003$。

两路参考和 FxLMS 仿真用的是同一组阶次，幅度和相位相同，各自再加一段白噪声，标准差 0.35。纯净目标信号两路各自独立，标准差 0.25。

四条次级路径的前 8 点为 0，末两点是原来的响应。滤波后参考用的模型系数就是这组真实路径的系数

$$
\begin{aligned}
\mathbf{s}_{00}&=[0,\,0,\,0,\,0,\,0,\,0,\,0,\,0,\,0.90,\,0.20]^{\mathsf T}\\
\mathbf{s}_{01}&=[0,\,0,\,0,\,0,\,0,\,0,\,0,\,0,\,0.15,\,0.05]^{\mathsf T}\\
\mathbf{s}_{10}&=[0,\,0,\,0,\,0,\,0,\,0,\,0,\,0,\,0.12,\,-0.04]^{\mathsf T}\\
\mathbf{s}_{11}&=[0,\,0,\,0,\,0,\,0,\,0,\,0,\,0,\,0.85,\,0.15]^{\mathsf T}
\end{aligned}
$$

初级路径仍是这组 8 点系数和次级路径的卷积，学习目标就是这组系数

$$
\begin{aligned}
\mathbf{h}_{qp}&=\sum_{l=0}^{1}\mathbf{s}_{ql}*\mathbf{b}_{lp}\\
\mathbf{b}_{00}&=[0.90,\,-0.55,\,0.35,\,-0.20,\,0.12,\,-0.06,\,0.03,\,-0.01]^{\mathsf T}\\
\mathbf{b}_{01}&=[0.40,\,0.22,\,-0.12,\,0.06,\,-0.03,\,0.01,\,0,\,0]^{\mathsf T}\\
\mathbf{b}_{10}&=[0.25,\,-0.18,\,0.10,\,-0.05,\,0.02,\,0,\,0,\,0]^{\mathsf T}\\
\mathbf{b}_{11}&=[0.80,\,-0.30,\,0.16,\,-0.08,\,0.04,\,-0.02,\,0.01,\,0]^{\mathsf T}
\end{aligned}
$$

后一半 DFT 置 0 之后，更新等价于 8 点线性相关，所以这 8 个抽头可以对上 $\mathbf{b}_{lp}$，系数从 0 开始。

### 3.3 仿真结果

![06 多通道频域 FxLMS](../output/sim/06_fdfxlms.png)

波形取记录末尾 $150\,\mathrm{ms}$，功率曲线是整段 $8\,\mathrm{s}$ 上、$0.2\,\mathrm{s}$ 滑动平均后的误差功率，功率谱取末尾 $4\,\mathrm{s}$，误差 NMSE 用末尾 $2\,\mathrm{s}$，系数取整段跑完之后的值，和 $\mathbf{b}_{lp}$ 比较。

**「两路参考信号」**：黑线是通道 0，大约在 $-2.4$ 到 $3$；蓝线是通道 1，大约在 $-1.7$ 到 $2.3$。阶次的起伏还在，上面叠着各自的白噪声。

**「两路纯净目标信号」**：两路都是标准差 0.25 的白噪声，这一段大约在 $-0.9$ 到 $0.7$。

**「两路目标信号」**：通道 0 大约在 $-2.7$ 到 $3.3$，通道 1 大约在 $-1.7$ 到 $2.2$。里面是两路参考经各自主路径之后的成分，再加纯净目标信号。

**「两路误差功率往纯净目标信号靠拢」**：红线从大约 $-4.9\,\mathrm{dB}$ 下降，黄线从大约 $-7.5\,\mathrm{dB}$ 下降，靠近 $-12\,\mathrm{dB}$ 的青绿虚线。虚线是两路纯净目标信号的平均功率。开头系数还是 0，这一块还没更新，所以误差功率高于这条虚线。

**「收敛后，通道 0 的误差贴住纯净目标信号」**：紫线仍是通道 0 的目标信号。红线是 $e_0$，青绿虚线是 $u_0$，两条叠在大约 $-0.9$ 到 $0.7$。末尾 $2\,\mathrm{s}$ 里，$e_0$ 相对 $u_0$ 的 NMSE 是 $3.61\times 10^{-2}$。通道 1 同一段的 NMSE 是 $2.72\times 10^{-2}$。

**「通道 0 的阶次被消掉之后，剩下的谱就是纯净目标信号」**：紫线在 40、80、120、160 Hz 有四个峰，大约 43、39、35、32 dB。峰之间的宽带底大约 15 dB。红线和青绿虚线贴在约 11 dB 的一条平线上，四个峰消失。

**「四条控制滤波器收敛到 b」**：圆点是 $\mathbf{b}_{lp}$，叉号是跑完 $8\,\mathrm{s}$ 之后的控制系数，同一条滤波器用同一个颜色。四组从左到右是 $(l,p)=(0,0)$、$(0,1)$、$(1,0)$、$(1,1)$。叉号落在圆点上。相对 $\mathbf{b}_{lp}$ 的系数 NMSE 是 $4.87\times 10^{-3}$。

**「收敛系数减去 b」**：同一组颜色，画的是收敛系数减去 $\mathbf{b}_{lp}$。差最大的一点大约 $0.047$，出现在扬声器 1、参考 1 的抽头 $m=1$。

## 4. 补充

### 4.1 OLS 卷积和相关

学习卷积的时候有一条非常著名的结论，时域卷积等于频域乘积，但实际上这个公式是基于线性卷积进行分析的。计算机只能在离散域上面操作，当频域被离散时，时域会出现周期化，从而导致时域信号循环混叠，导致圆周卷积现象，所以离散域上面频域卷积会采用 OLA 或者 OLS 方法获得线性卷积结果，其本质是一样的，都是在将周期化的时域做每周期延拓从而避免混叠，只是在延拓段的处理方式不一样，通常会采用 OLS 法，在内存和算力消耗上面更有优势。

滤波器长度 $M$，DFT 长度 $N=2M$。给定系数 $\mathbf{w}$ 和输入 $x(n)$，每次用 $M$ 个新样本算出对应的 $M$ 个卷积结果。输入帧由上一块留下的 $M$ 个旧样本接上这 $M$ 个新样本，$t=0$ 是最旧的一点

$$
x_{\mathrm{blk}}[t]=x(n-N+1+t),\qquad t=0,\ldots,N-1
$$

系数补 $M$ 个 0，凑成 $N$ 点

$$
h[m]=
\begin{cases}
w_m & 0\le m<M\\
0 & M\le m<N
\end{cases}
$$

两边做 DFT，频域相乘，再做逆变换回到时域

$$
\begin{aligned}
Y(k)&=H(k)\,X_{\mathrm{blk}}(k)\\
y_{\mathrm{circ}}&=\mathrm{IDFT}(Y)
\end{aligned}
$$

$y_{\mathrm{circ}}$ 长 $N$。前 $M$ 点是这一帧里的混叠段，截掉。后 $M$ 点留下，按时间对齐到这 $M$ 个新样本

$$
y(n-M+1+i)=y_{\mathrm{circ}}[M+i],\qquad i=0,\ldots,M-1
$$

下一块再算时，把这一帧的后 $M$ 点留下来当作旧样本，后面接 $M$ 个新输入。

相关给定的是这一块的误差 $e$ 和滤波后参考 $x'$，已知相关和卷积存在时域反转频域共轭关系，这导致 OLS 计算方法存在轻微的差异。滤波后参考的帧和上面的输入帧一样，旧的 $M$ 点接新的 $M$ 点

$$
x'_{\mathrm{blk}}[t]=x'(n-N+1+t),\qquad t=0,\ldots,N-1
$$

误差补到 $N$ 点时，前 $M$ 点补 0，后 $M$ 点放这一块的误差

$$
e_{\mathrm{blk}}[t]=
\begin{cases}
0 & 0\le t<M\\
e(n-N+1+t) & M\le t<N
\end{cases}
$$

两边做 DFT乘积，再做逆变换回到时域

$$
\begin{aligned}
G(k)&=E(k)\,X_{\mathrm{blk}}^{\prime *}(k)\\
g_{\mathrm{circ}}&=\mathrm{IDFT}(G)
\end{aligned}
$$

$g_{\mathrm{circ}}$ 长 $N$。后 $M$ 点是绕回来的滞后，截掉。前 $M$ 点留下

$$
g(m)=g_{\mathrm{circ}}(m),\qquad m=0,\ldots,M-1
$$

$g(m)$ 就是这一块上 $\sum_{i=0}^{M-1}e(n-M+1+i)\,x'(n-M+1+i-m)$。下一块的滤波后参考同样留下后 $M$ 点当作旧样本。

在实例里面可以看到具体的时域频域计算对比。

### 4.2 次级路径和自适应滤波器长度

在上述推导中设定次级路径和滤波器长度等长，这是工程实践中比较常用的做法，主要是为了提高数据的可复用性，但是如果不等长该如何处理呢？

正如 4.1 节所表述的那样，即使形式存在差异，但是补充块的本质是绕开混叠区，那么理所当然的，补充块长度虽然有最短要求，却没有最长要求，因此，如果次级路径和滤波器长度不等长，直接以更长的为基准进行计算即可。也可以像本章代码一样，采用不同的补充块形式，只要满足基础的点数要求即可。

### 4.3 三条核心公式的独立性

其实从频域的推导也可以看出来，参考信号滤波，滤波器更新和输出计算是具备独立性的，他们彼此之间并不具备强依赖，所以很多优化措施可以十分灵活的施加于任意一个环节，从而在工程实现上带来非常多的变体。需要保障的重点其实只有一个，那就是相关计算时保障信号相位差的正确性。

### 4.4 时延缺陷

频域算法最大的缺陷在于时延，为了避免时域混叠，必须有足够的点数用于频域卷积和相关计算，所以每次更新必须等待一段时间积累数据，导致输出和更新的时延远大于时域算法。在仿真里面体现不出太大的差异，但是到了实际物理系统里面，时延会带来极大的性能劣化。
