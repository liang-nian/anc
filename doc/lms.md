# LMS

维纳滤波在统计量准确的前提下可以直接求得最优解，但是实际的 ANC 系统几乎不会选它当核心算法。耳机和车载都把时延卡得很死，核心算法不能有太大的算力消耗。而维纳滤波计算自相关元素就已经涉及到大量的内积，更何况后面还有矩阵运算。

那么，能不能像维纳滤波那一章的仿真一样，不再做在线流式更新，直接拿全部数据做一次离线计算？目标系统若是理想的线性时不变系统，那么确实是可以的，但实际应用中并非如此。自适应滤波器的建模对象是噪声传递路径，可是车上的噪声传递路径包含吸振器，悬架等元件，其特性就是会随着激励发生变化的。所以 ANC 的核心算法必须能在线自适应更新。

我们必须找到一个满足以上要求的核心算法。换个角度思考，维纳滤波是令梯度为零，直接得到这个凸问题的最优解，如果我们不再直接求这个最优解，而是用梯度下降，实时地逼近它呢？由此得到的就是 LMS 算法。

ps:公式推导只是按照自己的理解进行简化版的推导，也许并不严谨。

## 1. 数学模型

维纳滤波里，均方误差对系数的梯度是

$$
\frac{\partial J}{\partial\mathbf{w}}=-2\mathbf{p}+2\mathbf{R}\mathbf{w}
$$

这里沿梯度往下走。步长写成 $\mu/2$，梯度里的 2 就消掉，更新式里只留下 $\mu$：

$$
\begin{aligned}
\mathbf{w}(n+1)
&=\mathbf{w}(n)-\frac{\mu}{2}\frac{\partial J}{\partial\mathbf{w}}\\
&=\mathbf{w}(n)+\mu\bigl(\mathbf{p}-\mathbf{R}\mathbf{w}(n)\bigr)
\end{aligned}
$$

$\mathbf{p}$ 和 $\mathbf{R}$ 仍然是期望，这么看似乎并没有减少计算量。但是大佬们给出了一个推导结论，在滤波器系数固定的情况下，滤波器系数可以穿透期望公式，从而使得瞬时梯度成为真实梯度的无偏估计，所以我们可以将梯度简化为实时值的计算

$$
\hat{\mathbf{p}}(n)=d(n)\mathbf{x}(n),\qquad
\hat{\mathbf{R}}(n)=\mathbf{x}(n)\mathbf{x}^{\mathsf T}(n)
$$

需要注意 $\mathbf{x}(n)$ 是矢量，$d(n)$ 是标量。这里把估计代进下降方向，两项相减就剩误差和参考的乘积：

$$
\begin{aligned}
\hat{\mathbf{p}}(n)-\hat{\mathbf{R}}(n)\mathbf{w}(n)
&=\bigl(d(n)-\mathbf{w}^{\mathsf T}(n)\mathbf{x}(n)\bigr)\mathbf{x}(n)\\
&=e(n)\mathbf{x}(n)
\end{aligned}
$$

于是得到 LMS 算法更新公式

$$
\mathbf{w}(n+1)=\mathbf{w}(n)+\mu\, e(n)\mathbf{x}(n)
$$

从信号处理角度换一种更加直观粗略的解释方法，维纳滤波基于序列相关进行计算，而 LMS 则只取一点，似乎和相关无关了，但随着时间的延展，梯度更新项本质上是在累加，所以实际上也仍然是在做相关计算。从 $\mathbf{w}(0)$ 累到第 $n$ 点：

$$
\begin{aligned}
\mathbf{w}(n)
&=\mathbf{w}(0)+\mu\sum_{t=0}^{n-1}e(t)\mathbf{x}(t)\\
\sum_{t=0}^{n-1}e(t)\mathbf{x}(t)
&=\sum_{t=0}^{n-1}d(t)\mathbf{x}(t)
-\sum_{t=0}^{n-1}\mathbf{x}(t)\mathbf{x}^{\mathsf T}(t)\mathbf{w}(t)
\end{aligned}
$$

右边第一项是 $d$ 和 $\mathbf{x}$ 的互相关累加，第二项是 $\mathbf{x}$ 的自相关累加再乘上当时的系数。系数若在这段时间里保持不变，$\mathbf{w}(t)$ 可以提出来，第二项就是 $\bigl(\sum_{t=0}^{n-1}\mathbf{x}(t)\mathbf{x}^{\mathsf T}(t)\bigr)\mathbf{w}$，这样就可以照应维纳滤波的公式。

多通道时参考有 $P$ 路，误差有 $Q$ 路。每一对 $(q,p)$ 各有一条 $M$ 阶 FIR。第 $p$ 路参考的抽头是

$$
\mathbf{x}_p(n)
=\begin{bmatrix}x_p(n)&x_p(n-1)&\cdots&x_p(n-M+1)\end{bmatrix}^{\mathsf T}
$$

第 $q$ 路把各参考滤完再相加，误差只更新汇入自己的那一组系数：

$$
\begin{aligned}
y_q(n)
&=\sum_{p=0}^{P-1}\mathbf{w}_{qp}^{\mathsf T}(n)\mathbf{x}_p(n)\\
e_q(n)
&=d_q(n)-y_q(n)\\
\mathbf{w}_{qp}(n+1)
&=\mathbf{w}_{qp}(n)+\mu\, e_q(n)\mathbf{x}_p(n)
\end{aligned}
$$

## 2. 实例

取 $P=2$ 路参考、$Q=1$ 路误差、阶数 $M=2$、步长 $\mu=1/2$，2 点输入信号序列值为

$$
\begin{aligned}
x_0(n)&: 1,\;1\\
x_1(n)&: 1,\;-1\\
d(n)&: 1,\;1
\end{aligned}
\qquad n=0,1
$$

两条系数向量是 $\mathbf{w}_{00}$ 和 $\mathbf{w}_{01}$，第一个下标是误差通道，第二个下标是参考通道。

$n=0$，四项乘积对应两条参考的当前抽头和历史抽头，缺失值直接补0：

$$
\begin{aligned}
y(0)
&=0\cdot 1+0\cdot 0+0\cdot 1+0\cdot 0=0\\
e(0)
&=1-0=1\\
\mathbf{w}_{00}(1)
&=\begin{bmatrix}0\\0\end{bmatrix}+\frac{1}{2}\cdot 1\cdot\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}1/2\\0\end{bmatrix}\\
\mathbf{w}_{01}(1)
&=\begin{bmatrix}0\\0\end{bmatrix}+\frac{1}{2}\cdot 1\cdot\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}1/2\\0\end{bmatrix}
\end{aligned}
$$

$n=1$ 时，两路参考抽头是 $\mathbf{x}_0(1)=[1,\,1]^{\mathsf T}$，$\mathbf{x}_1(1)=[-1,\,1]^{\mathsf T}$：

$$
\begin{aligned}
y(1)
&=\frac{1}{2}\cdot 1+0\cdot 1+\frac{1}{2}\cdot(-1)+0\cdot 1=0\\
e(1)
&=1-0=1\\
\mathbf{w}_{00}(2)
&=\begin{bmatrix}1/2\\0\end{bmatrix}+\frac{1}{2}\cdot 1\cdot\begin{bmatrix}1\\1\end{bmatrix}
=\begin{bmatrix}1\\1/2\end{bmatrix}\\
\mathbf{w}_{01}(2)
&=\begin{bmatrix}1/2\\0\end{bmatrix}+\frac{1}{2}\cdot 1\cdot\begin{bmatrix}-1\\1\end{bmatrix}
=\begin{bmatrix}0\\1/2\end{bmatrix}
\end{aligned}
$$

## 3. 仿真

### 3.1 函数对应的公式

`LmsCreate` 按 $P$、$Q$、$M$ 和步长 $\mu$ 建状态。系数和每路参考的延迟线都置 0。系数按下标 $(qP+p)M+k$ 排，$k=0$ 是当前抽头。

`LmsStep` 每次吃一个采样点：各参考一个数，各目标一个数，给出各路误差。先把新样本推进延迟线，用更新前的系数算

$$
y_q=\sum_{p=0}^{P-1}\sum_{k=0}^{M-1}w_{qp,k}\, x_p(n-k),\qquad
e_q=d_q-y_q
$$

再更新

$$
w_{qp,k}\leftarrow w_{qp,k}+\mu\, e_q\, x_p(n-k)
$$

各路 $e_q$ 全部算完之后才改系数。脚本按时间流式调用，每一点调用一次 `LmsStep`。

`LmsCopyWeights` 把当前系数抄出来，用来和预设主路径对照。`LmsDestroy` 释放状态。

### 3.2 仿真设计

采样率 $f_s=2000\,\mathrm{Hz}$，$N=16000$，随机数种子为 7。参考 $P=2$ 路，误差 $Q=2$ 路，每条 FIR 阶数 $M=8$，步长 $\mu=0.002$。

两路参考使用的是同一组阶次，幅度和相位不同，各自再加一段白噪声，标准差 0.35：

$$
\begin{aligned}
s_0(n)=&\sin(2\pi\cdot 40\, n/f_s)+0.7\sin(2\pi\cdot 80\, n/f_s)\\
&+0.45\sin(2\pi\cdot 120\, n/f_s)+0.25\sin(2\pi\cdot 160\, n/f_s)\\
s_1(n)=&0.75\sin(2\pi\cdot 40\, n/f_s+0.6)+0.55\sin(2\pi\cdot 80\, n/f_s+0.6)\\
&+0.30\sin(2\pi\cdot 120\, n/f_s+0.6)+0.20\sin(2\pi\cdot 160\, n/f_s+0.6)\\
x_p(n)=&s_p(n)+0.35\,g_p(n)
\end{aligned}
$$

$g_p(n)$ 是彼此独立的单位方差高斯白噪声。相位 $0.6$ 的单位是弧度。

四条主路径是预先写好的 8 点 FIR：

$$
\begin{aligned}
\mathbf{h}_{00}&=[0.90,\,-0.55,\,0.35,\,-0.20,\,0.12,\,-0.06,\,0.03,\,-0.01]^{\mathsf T}\\
\mathbf{h}_{01}&=[0.40,\,0.22,\,-0.12,\,0.06,\,-0.03,\,0.01,\,0,\,0]^{\mathsf T}\\
\mathbf{h}_{10}&=[0.25,\,-0.18,\,0.10,\,-0.05,\,0.02,\,0,\,0,\,0]^{\mathsf T}\\
\mathbf{h}_{11}&=[0.80,\,-0.30,\,0.16,\,-0.08,\,0.04,\,-0.02,\,0.01,\,0]^{\mathsf T}
\end{aligned}
$$

纯净目标信号 $u_q(n)$ 是目标点上没有从参考传播过来的成分，两路各自独立，标准差 0.25，与参考无关：

$$
d_q(n)=\sum_{p=0}^{1}\mathbf{h}_{qp}(n)*x_p(n)+u_q(n)
$$

系数从 0 开始，边滤波边更新。收敛之后

$$
e_q(n)\approx u_q(n)
$$

步长保持 $0.002$，系数在主路径附近还有抖动，所以误差比 $u_q$ 多出一截。

### 3.3 仿真结果

![02 多通道 LMS](../output/sim/02_lms.png)

波形取记录末尾 $150\,\mathrm{ms}$。功率曲线是整段 $8\,\mathrm{s}$ 上、$0.2\,\mathrm{s}$ 滑动平均后的误差功率。功率谱取末尾 $4\,\mathrm{s}$。NMSE 用末尾 $2\,\mathrm{s}$。

**「两路参考信号」**：黑线是通道 0，大约在 $-2.4$ 到 $3$；蓝线是通道 1，大约在 $-1.7$ 到 $2.4$。阶次的起伏还在，上面叠着各自的白噪声。

**「两路纯净目标信号」**：两路都是标准差 0.25 的白噪声，这一段大约在 $-0.9$ 到 $0.7$。

**「两路目标信号」**：通道 0 大约在 $-3.4$ 到 $3$，通道 1 大约在 $-1.8$ 到 $1.7$。里面是两路参考经各自主路径之后的成分，再加纯净目标信号。

**「两路误差功率往纯净目标信号靠拢」**：红线从大约 $-5\,\mathrm{dB}$ 下降，黄线从大约 $-7\,\mathrm{dB}$ 下降。大约两秒后贴上 $-12\,\mathrm{dB}$ 的青绿虚线，之后在虚线附近小幅抖动。虚线是两路纯净目标信号的平均功率。开头系数还是 0，主路径还没建起来，所以误差功率高于这条虚线。

**「收敛后，通道 0 的误差贴住纯净目标信号」**：紫线仍是通道 0 的目标信号。红线是 $e_0$，青绿虚线是 $u_0$，两条叠在大约 $-0.9$ 到 $0.7$。末尾 $2\,\mathrm{s}$ 里，$e_0$ 相对 $u_0$ 的 NMSE 是 $1.79\times 10^{-2}$。通道 1 同一段的 NMSE 是 $1.53\times 10^{-2}$。四条 FIR 相对预设主路径的 NMSE 是 $1.10\times 10^{-3}$。

**「通道 0 的阶次被消掉之后，剩下的谱就是纯净目标信号」**：紫线在 40、80、120、160 Hz 有四个峰，大约 41、38、34、30 dB，最高约 41 dB。峰之间的宽带底大约 14 dB，里面有参考白噪声经主路径之后的分量。红线和青绿虚线贴在约 11 dB 的一条平线上，四个峰消失。

## 4. 总结

LMS 算法也有着许多有价值的特性，最为经典的就是特征值特性，通过分析梯度下降法的误差收敛条件可以得到步长的约束公式

$$
0<\mu<\frac{2}{\lambda_{\max}}
$$

因此，自相关矩阵的特征值决定了步长的合法范围，以及收敛的速度，显然，最大特征值和最小特征值越接近，越有利于算法的收敛，这个特性也为实际量产中部分优化手段提供了理论基础。

LMS 算法已经满足了基本的计算需求，但是仍然缺失了一部分数学建模，目前的模型假设来源于维纳滤波，在维纳滤波里面，我们直接将参考信号滤波后的结果视作降噪信号，但实际上在物理系统中，这个声音需要扬声器来播放，而扬声器传播到降噪点才获得降噪信号，所以必须将这段传递路径考虑进去重新进行建模。