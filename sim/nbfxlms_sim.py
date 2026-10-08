#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from ctypes import CDLL, POINTER, c_double, c_int, c_size_t, c_void_p
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "nbfxlms"
BUILD_DIR = ROOT / "output" / "nbfxlms"
FIG_DIR = ROOT / "output" / "sim"

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
    }
)

C_REF0 = "#222222"
C_REF1 = "#3d5a80"
C_CLEAN0 = "#2a9d8f"
C_CLEAN1 = "#1d3557"
C_TARGET0 = "#5b4b8a"
C_TARGET1 = "#9b5de5"
C_AFTER = "#c0392b"
C_MARK = "#e09f3e"

N_HARM = 4
N_SPK = 2
N_ERR = 2
SEC_ORDER = 3
STEP = 0.01
ORIGIN_STD = 0.25
F0 = 40.0
ORDERS = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
FREQS = F0 * ORDERS
SECONDARY = np.zeros((N_ERR, N_SPK, SEC_ORDER), dtype=np.float64)
SECONDARY[0, 0] = [0.0, 0.90, 0.20]
SECONDARY[0, 1] = [0.0, 0.15, 0.05]
SECONDARY[1, 0] = [0.0, 0.12, -0.04]
SECONDARY[1, 1] = [0.0, 0.85, 0.15]
BEFORE = np.zeros((N_SPK, N_HARM, 2), dtype=np.float64)
BEFORE[0, 0] = [0.70, -0.25]
BEFORE[0, 1] = [0.45, 0.20]
BEFORE[0, 2] = [0.25, -0.10]
BEFORE[0, 3] = [0.12, 0.05]
BEFORE[1, 0] = [0.30, 0.40]
BEFORE[1, 1] = [-0.15, 0.28]
BEFORE[1, 2] = [0.10, 0.16]
BEFORE[1, 3] = [-0.06, 0.08]


class NbfxlmsLib:
    def __init__(self) -> None:
        subprocess.check_call(
            ["cmake", "-S", str(NB_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"]
        )
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR)])
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = CDLL(str(BUILD_DIR / f"libnbfxlms{ext}"))
        lib.NbfxlmsCreate.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_double,
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.NbfxlmsCreate.restype = c_void_p
        lib.NbfxlmsDestroy.argtypes = [c_void_p]
        lib.NbfxlmsCore.argtypes = [
            c_void_p,
            c_double,
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.NbfxlmsCore.restype = c_int
        lib.NbfxlmsCopyWeights.argtypes = [c_void_p, POINTER(c_double)]
        lib.NbfxlmsCopyWeights.restype = c_int
        self.lib = lib

    def create(self, order: np.ndarray, secondary: np.ndarray) -> c_void_p:
        sec = np.ascontiguousarray(secondary, dtype=np.float64)
        od = np.ascontiguousarray(order, dtype=np.float64)
        state = self.lib.NbfxlmsCreate(
            c_size_t(N_HARM),
            c_size_t(N_SPK),
            c_size_t(N_ERR),
            c_size_t(SEC_ORDER),
            c_double(STEP),
            sec.ctypes.data_as(POINTER(c_double)),
            od.ctypes.data_as(POINTER(c_double)),
        )
        if not state:
            raise RuntimeError("NbfxlmsCreate failed")
        return state

    def destroy(self, state: c_void_p) -> None:
        self.lib.NbfxlmsDestroy(state)

    def core(self, state: c_void_p, omega0: float, e: np.ndarray, y: np.ndarray) -> None:
        rc = self.lib.NbfxlmsCore(
            state,
            c_double(omega0),
            e.ctypes.data_as(POINTER(c_double)),
            y.ctypes.data_as(POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"NbfxlmsCore failed: {rc}")

    def weights(self, state: c_void_p) -> np.ndarray:
        w = np.zeros(N_SPK * N_HARM * 2, dtype=np.float64)
        rc = self.lib.NbfxlmsCopyWeights(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError(f"NbfxlmsCopyWeights failed: {rc}")
        return w.reshape(N_SPK, N_HARM, 2)


def shift_push(line: np.ndarray, value: float) -> None:
    for k in range(line.shape[0] - 1, 0, -1):
        line[k] = line[k - 1]
    line[0] = value


def mic_error(y_delay: np.ndarray, d: np.ndarray) -> np.ndarray:
    """误差由已经播出的扬声器历史和目标信号相减得到。次级路径第 0 拍为 0。"""
    e = np.zeros(N_ERR, dtype=np.float64)
    for q in range(N_ERR):
        acc = 0.0
        for l in range(N_SPK):
            for j in range(SEC_ORDER):
                acc += SECONDARY[q, l, j] * y_delay[l, j]
        e[q] = d[q] - acc
    return e


def open_speaker_slot(y_delay: np.ndarray) -> None:
    for l in range(N_SPK):
        shift_push(y_delay[l], 0.0)


def write_speaker(y: np.ndarray, y_delay: np.ndarray) -> None:
    for l in range(N_SPK):
        y_delay[l, 0] = float(y[l])


def reference_bank(n: int, order: np.ndarray, omega0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """与 C 同一套旋转。阶次固定，每一拍用这一拍的基频角频率。"""
    xc = np.ones(order.size, dtype=np.float64)
    xs = np.zeros(order.size, dtype=np.float64)
    xc_hist = np.zeros((order.size, n), dtype=np.float64)
    xs_hist = np.zeros((order.size, n), dtype=np.float64)
    for t in range(n):
        xc_hist[:, t] = xc
        xs_hist[:, t] = xs
        wi = order * omega0[t]
        rot_c = np.cos(wi)
        rot_s = np.sin(wi)
        nxc = rot_c * xc - rot_s * xs
        nxs = rot_s * xc + rot_c * xs
        xc = nxc
        xs = nxs
    return xc_hist, xs_hist


def tonal_from_weights(xc: np.ndarray, xs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = xc.shape[1]
    y = np.zeros((N_SPK, n), dtype=np.float64)
    for l in range(N_SPK):
        for i in range(N_HARM):
            y[l] += weights[l, i, 0] * xc[i] + weights[l, i, 1] * xs[i]
    tonal = np.zeros((N_ERR, n), dtype=np.float64)
    for q in range(N_ERR):
        for l in range(N_SPK):
            tonal[q] += np.convolve(y[l], SECONDARY[q, l], mode="full")[:n]
    return tonal


def numpy_nbfxlms(
    d: np.ndarray, order: np.ndarray, omega0: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """与 C 同一套逐点更新，次级路径放在循环里，只用于核对。"""
    n = d.shape[1]
    xc = np.ones(N_HARM, dtype=np.float64)
    xs = np.zeros(N_HARM, dtype=np.float64)
    x_state_c = np.zeros((N_HARM, SEC_ORDER), dtype=np.float64)
    x_state_s = np.zeros((N_HARM, SEC_ORDER), dtype=np.float64)
    w = np.zeros((N_SPK, N_HARM, 2), dtype=np.float64)
    y_delay = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    e = np.zeros((N_ERR, n), dtype=np.float64)
    y = np.zeros(N_SPK, dtype=np.float64)
    for t in range(n):
        open_speaker_slot(y_delay)
        e_now = mic_error(y_delay, d[:, t])
        x_now_c = xc.copy()
        x_now_s = xs.copy()
        for i in range(N_HARM):
            shift_push(x_state_c[i], float(x_now_c[i]))
            shift_push(x_state_s[i], float(x_now_s[i]))
        wi = order * omega0[t]
        rot_c = np.cos(wi)
        rot_s = np.sin(wi)
        nxc = rot_c * xc - rot_s * xs
        nxs = rot_s * xc + rot_c * xs
        xc = nxc
        xs = nxs
        fx_c = np.zeros((N_ERR, N_SPK, N_HARM), dtype=np.float64)
        fx_s = np.zeros((N_ERR, N_SPK, N_HARM), dtype=np.float64)
        for q in range(N_ERR):
            for l in range(N_SPK):
                for i in range(N_HARM):
                    fc = 0.0
                    fs = 0.0
                    for j in range(SEC_ORDER):
                        fc += SECONDARY[q, l, j] * x_state_c[i, j]
                        fs += SECONDARY[q, l, j] * x_state_s[i, j]
                    fx_c[q, l, i] = fc
                    fx_s[q, l, i] = fs
        for l in range(N_SPK):
            for i in range(N_HARM):
                acc_c = 0.0
                acc_s = 0.0
                for q in range(N_ERR):
                    acc_c += e_now[q] * fx_c[q, l, i]
                    acc_s += e_now[q] * fx_s[q, l, i]
                w[l, i, 0] += step * acc_c
                w[l, i, 1] += step * acc_s
        for l in range(N_SPK):
            acc = 0.0
            for i in range(N_HARM):
                acc += w[l, i, 0] * x_now_c[i] + w[l, i, 1] * x_now_s[i]
            y[l] = acc
        write_speaker(y, y_delay)
        e[:, t] = e_now
    return e, w


def welch_psd(x: np.ndarray, fs: float, n_fft: int = 512, hop: int = 256):
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    nframes = 0
    for start in range(0, x.size - n_fft + 1, hop):
        spec = np.fft.rfft(x[start : start + n_fft] * window)
        acc += np.abs(spec) ** 2
        nframes += 1
    if nframes == 0:
        raise RuntimeError("signal shorter than one Welch frame")
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return freq, acc / nframes


def save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def synthesize(n: int, fs: float, rng: np.random.Generator):
    omega0 = np.full(n, 2.0 * np.pi * F0 / fs, dtype=np.float64)
    xc, xs = reference_bank(n, ORDERS, omega0)
    origin = ORIGIN_STD * rng.normal(size=(N_ERR, n))
    d = tonal_from_weights(xc, xs, BEFORE) + origin
    return omega0, xc, xs, origin, d


def simulate(lib: NbfxlmsLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    omega0, xc, xs, origin, d = synthesize(n, fs, rng)

    state = lib.create(ORDERS, SECONDARY)
    e = np.zeros((N_ERR, n), dtype=np.float64)
    y_delay = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    sample_y = np.zeros(N_SPK, dtype=np.float64)
    sample_e = np.zeros(N_ERR, dtype=np.float64)
    try:
        for i in range(n):
            open_speaker_slot(y_delay)
            sample_e[:] = mic_error(y_delay, d[:, i])
            lib.core(state, float(omega0[i]), sample_e, sample_y)
            write_speaker(sample_y, y_delay)
            e[:, i] = sample_e
        w_c = lib.weights(state)
    finally:
        lib.destroy(state)

    e_np, w_np = numpy_nbfxlms(d, ORDERS, omega0, STEP)
    gap = float(np.max(np.abs(w_c - w_np)))
    if gap > 1e-8:
        raise RuntimeError(f"C and NumPy weights differ by {gap}")

    tail = slice(-4000, None)
    origin_nmse = []
    for q in range(N_ERR):
        num = float(np.sum((e[q, tail] - origin[q, tail]) ** 2))
        den = float(np.sum(origin[q, tail] ** 2))
        origin_nmse.append(num / den)
    if max(origin_nmse) > 5e-2:
        raise RuntimeError(f"residual does not track the original signal, NMSE {origin_nmse}")
    coeff_nmse = float(np.sum((w_c - BEFORE) ** 2) / np.sum(BEFORE**2))

    fig, axes = plt.subplots(8, 1, figsize=(9.2, 23.2), constrained_layout=True)
    zoom0, zoom1 = n - 300, n
    n_axis = (np.arange(zoom0, zoom1) - zoom0) / fs * 1e3

    def time_pair(ax, a, b, ca, cb, la, lb, title):
        ax.plot(n_axis, a[zoom0:zoom1], color=ca, lw=1.15, label=la)
        ax.plot(n_axis, b[zoom0:zoom1], color=cb, lw=1.15, label=lb)
        ax.set_xlabel("时间 / ms")
        ax.set_ylabel("幅度")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=2)

    time_pair(axes[0], xc[0], xc[1], C_REF0, C_REF1, "40 Hz", "80 Hz", "基频和二次的余弦参考")
    time_pair(
        axes[1], origin[0], origin[1], C_CLEAN0, C_CLEAN1, "通道 0", "通道 1", "两路纯净目标信号"
    )
    time_pair(axes[2], d[0], d[1], C_TARGET0, C_TARGET1, "通道 0", "通道 1", "两路目标信号")

    win = 400
    kernel = np.ones(win) / win
    t_learn = (np.arange(n - win + 1) + win) / fs
    for q, color in enumerate((C_AFTER, C_MARK)):
        power = np.convolve(e[q] ** 2, kernel, mode="valid")
        axes[3].plot(
            t_learn, 10.0 * np.log10(np.maximum(power, 1e-20)), color=color, lw=1.2, label=f"通道 {q}"
        )
    origin_db = 10.0 * np.log10(float(np.mean(origin**2)))
    axes[3].axhline(origin_db, color=C_CLEAN0, lw=1.15, ls="--", label="纯净目标信号")
    axes[3].set_xlabel("时间 / s")
    axes[3].set_ylabel("功率 / dB")
    axes[3].set_title("两路误差功率往纯净目标信号靠拢")
    axes[3].legend(loc="upper right", ncol=3)

    axes[4].plot(n_axis, d[0, zoom0:zoom1], color=C_TARGET0, lw=1.15, label="目标信号")
    axes[4].plot(n_axis, e[0, zoom0:zoom1], color=C_AFTER, lw=1.15, label="抵消后")
    axes[4].plot(
        n_axis, origin[0, zoom0:zoom1], color=C_CLEAN0, lw=1.15, ls="--", label="纯净目标信号"
    )
    axes[4].set_xlabel("时间 / ms")
    axes[4].set_ylabel("幅度")
    axes[4].set_title(f"收敛后，通道 0 的误差贴住纯净目标信号    NMSE = {origin_nmse[0]:.2e}")
    axes[4].legend(loc="upper right", ncol=3)

    spec0 = slice(-8000, None)
    freq, p_before = welch_psd(d[0, spec0], fs)
    _, p_after = welch_psd(e[0, spec0], fs)
    _, p_origin = welch_psd(origin[0, spec0], fs)

    def to_db(p: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(p, 1e-20))

    axes[5].plot(freq, to_db(p_before), color=C_TARGET0, lw=1.2, label="目标信号")
    axes[5].plot(freq, to_db(p_after), color=C_AFTER, lw=1.2, label="抵消后")
    axes[5].plot(freq, to_db(p_origin), color=C_CLEAN0, lw=1.15, ls="--", label="纯净目标信号")
    axes[5].set_xlim(0, 400)
    axes[5].set_xlabel("频率 / Hz")
    axes[5].set_ylabel("功率谱 / dB")
    axes[5].set_title("通道 0 的阶次被消掉之后，剩下的谱就是纯净目标信号")
    axes[5].legend(loc="upper right", ncol=3)

    group = 3
    pair_color = (C_REF0, C_REF1, C_TARGET0, C_AFTER, C_CLEAN0, C_CLEAN1, C_MARK, C_TARGET1)
    names = []
    for l in range(N_SPK):
        for i, hz in enumerate(FREQS):
            idx = l * N_HARM + i
            loc = np.arange(2) + idx * group
            color = pair_color[idx]
            axes[6].plot(loc, BEFORE[l, i], color=color, ls="none", marker="o", ms=5.5)
            axes[6].plot(loc, w_c[l, i], color=color, lw=1.15, marker="x", ms=6)
            err = w_c[l, i] - BEFORE[l, i]
            axes[7].vlines(loc, 0.0, err, color=color, lw=1.15)
            axes[7].plot(loc, err, color=color, ls="none", marker="o", ms=4.5)
            names.append(f"扬声器 {l}\n{int(hz)} Hz")
    centers = [i * group + 0.5 for i in range(N_SPK * N_HARM)]
    for ax in (axes[6], axes[7]):
        ax.set_xticks(centers)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_xlabel("阶次")
        ax.set_ylabel("系数")
    axes[6].set_title(f"各阶次的余弦、正弦系数收敛到 b    系数 NMSE = {coeff_nmse:.2e}")
    axes[7].axhline(0.0, color="#888888", lw=0.8)
    axes[7].set_title("收敛系数减去 b")
    path = save(fig, "05_nbfxlms.png")
    return path, origin_nmse, coeff_nmse, gap


def main() -> None:
    lib = NbfxlmsLib()
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap = simulate(lib, rng)
    for q, nmse in enumerate(origin_nmse):
        print(f"channel {q} residual vs origin NMSE {nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}")
    print(f"C vs NumPy {gap:.3e}")
    print(path)


if __name__ == "__main__":
    main()
