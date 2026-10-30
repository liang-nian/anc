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
FXLMS_DIR = ROOT / "fxlms"
BUILD_DIR = ROOT / "output" / "fxlms"
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

ORDER = 8
SEC_ORDER = 3
N_REF = 2
N_SPK = 2
N_ERR = 2
STEP = 0.004
REF_NOISE_STD = 0.35
ORIGIN_STD = 0.25
TONES = (40.0, 80.0, 120.0, 160.0)
AMP0 = (1.0, 0.7, 0.45, 0.25)
AMP1 = (0.75, 0.55, 0.30, 0.20)
PHASE1 = 0.6
SECONDARY = np.zeros((N_ERR, N_SPK, SEC_ORDER), dtype=np.float64)
SECONDARY[0, 0] = [0.0, 0.90, 0.20]
SECONDARY[0, 1] = [0.0, 0.15, 0.05]
SECONDARY[1, 0] = [0.0, 0.12, -0.04]
SECONDARY[1, 1] = [0.0, 0.85, 0.15]
BEFORE = np.zeros((N_SPK, N_REF, 8), dtype=np.float64)
BEFORE[0, 0] = [0.90, -0.55, 0.35, -0.20, 0.12, -0.06, 0.03, -0.01]
BEFORE[0, 1] = [0.40, 0.22, -0.12, 0.06, -0.03, 0.01, 0.0, 0.0]
BEFORE[1, 0] = [0.25, -0.18, 0.10, -0.05, 0.02, 0.0, 0.0, 0.0]
BEFORE[1, 1] = [0.80, -0.30, 0.16, -0.08, 0.04, -0.02, 0.01, 0.0]
PRIMARY = np.zeros((N_ERR, N_REF, BEFORE.shape[-1] + SEC_ORDER - 1), dtype=np.float64)
for _q in range(N_ERR):
    for _p in range(N_REF):
        for _l in range(N_SPK):
            PRIMARY[_q, _p] += np.convolve(SECONDARY[_q, _l], BEFORE[_l, _p])


class FxlmsLib:
    def __init__(self) -> None:
        subprocess.check_call(
            ["cmake", "-S", str(FXLMS_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"]
        )
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR)])
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = CDLL(str(BUILD_DIR / f"libfxlms{ext}"))
        lib.FxlmsCreate.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_double,
            POINTER(c_double),
        ]
        lib.FxlmsCreate.restype = c_void_p
        lib.FxlmsDestroy.argtypes = [c_void_p]
        lib.FxlmsCore.argtypes = [
            c_void_p,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.FxlmsCore.restype = c_int
        lib.FxlmsCopyWeights.argtypes = [c_void_p, POINTER(c_double)]
        lib.FxlmsCopyWeights.restype = c_int
        self.lib = lib

    def create(
        self,
        n_ref: int,
        n_spk: int,
        n_err: int,
        order: int,
        sec_order: int,
        step: float,
        secondary: np.ndarray,
    ) -> c_void_p:
        sec = np.ascontiguousarray(secondary, dtype=np.float64)
        state = self.lib.FxlmsCreate(
            c_size_t(n_ref),
            c_size_t(n_spk),
            c_size_t(n_err),
            c_size_t(order),
            c_size_t(sec_order),
            c_double(step),
            sec.ctypes.data_as(POINTER(c_double)),
        )
        if not state:
            raise RuntimeError("FxlmsCreate failed")
        return state

    def destroy(self, state: c_void_p) -> None:
        self.lib.FxlmsDestroy(state)

    def core(self, state: c_void_p, x: np.ndarray, e: np.ndarray, y: np.ndarray) -> None:
        rc = self.lib.FxlmsCore(
            state,
            x.ctypes.data_as(POINTER(c_double)),
            e.ctypes.data_as(POINTER(c_double)),
            y.ctypes.data_as(POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"FxlmsCore failed: {rc}")

    def weights(self, state: c_void_p, n_spk: int, n_ref: int, order: int) -> np.ndarray:
        w = np.zeros(n_spk * n_ref * order, dtype=np.float64)
        rc = self.lib.FxlmsCopyWeights(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError(f"FxlmsCopyWeights failed: {rc}")
        return w.reshape(n_spk, n_ref, order)


def primary_mix(x: np.ndarray) -> np.ndarray:
    n_err, n_ref, _ = PRIMARY.shape
    d = np.zeros((n_err, x.shape[1]), dtype=np.float64)
    for q in range(n_err):
        for p in range(n_ref):
            d[q] += np.convolve(x[p], PRIMARY[q, p], mode="full")[: x.shape[1]]
    return d


def shift_push(line: np.ndarray, value: float) -> None:
    for k in range(line.shape[0] - 1, 0, -1):
        line[k] = line[k - 1]
    line[0] = value


def mic_error(y_delay: np.ndarray, d: np.ndarray) -> np.ndarray:
    """误差由已经播出的扬声器历史和目标信号相减得到。次级路径第 0 点为 0。"""
    e = np.zeros(N_ERR, dtype=np.float64)
    for q in range(N_ERR):
        acc = 0.0
        for l in range(N_SPK):
            for j in range(SEC_ORDER):
                acc += SECONDARY[q, l, j] * y_delay[l, j]
        e[q] = d[q] - acc
    return e


def open_speaker_slot(y_delay: np.ndarray) -> None:
    """当前扬声器样本还没算出。第 0 点系数为 0，先占一个空位对齐延迟。"""
    for l in range(N_SPK):
        shift_push(y_delay[l], 0.0)


def write_speaker(y: np.ndarray, y_delay: np.ndarray) -> None:
    for l in range(N_SPK):
        y_delay[l, 0] = float(y[l])


def numpy_fxlms(x: np.ndarray, d: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """与 C 同一套逐点更新，次级路径放在循环里，只用于核对。"""
    n_ref, n = x.shape
    n_err = d.shape[0]
    order = ORDER
    x_len = max(order, SEC_ORDER)
    w = np.zeros((N_SPK, n_ref, order), dtype=np.float64)
    x_state = np.zeros((n_ref, x_len), dtype=np.float64)
    fx_state = np.zeros((n_err, N_SPK, n_ref, order), dtype=np.float64)
    y_delay = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    e = np.zeros((n_err, n), dtype=np.float64)
    y = np.zeros(N_SPK, dtype=np.float64)
    for i in range(n):
        open_speaker_slot(y_delay)
        e_now = mic_error(y_delay, d[:, i])
        for p in range(n_ref):
            shift_push(x_state[p], float(x[p, i]))
        for q in range(n_err):
            for l in range(N_SPK):
                for p in range(n_ref):
                    fx = 0.0
                    for j in range(SEC_ORDER):
                        fx += SECONDARY[q, l, j] * x_state[p, j]
                    shift_push(fx_state[q, l, p], fx)
        for l in range(N_SPK):
            for p in range(n_ref):
                for k in range(order):
                    acc = 0.0
                    for q in range(n_err):
                        acc += e_now[q] * fx_state[q, l, p, k]
                    w[l, p, k] += step * acc
        for l in range(N_SPK):
            acc = 0.0
            for p in range(n_ref):
                for k in range(order):
                    acc += w[l, p, k] * x_state[p, k]
            y[l] = acc
        write_speaker(y, y_delay)
        e[:, i] = e_now
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
    t = np.arange(n) / fs
    x = np.zeros((N_REF, n), dtype=np.float64)
    for amp, freq in zip(AMP0, TONES):
        x[0] += amp * np.sin(2.0 * np.pi * freq * t)
    for amp, freq in zip(AMP1, TONES):
        x[1] += amp * np.sin(2.0 * np.pi * freq * t + PHASE1)
    x[0] += REF_NOISE_STD * rng.normal(size=n)
    x[1] += REF_NOISE_STD * rng.normal(size=n)
    origin = ORIGIN_STD * rng.normal(size=(N_ERR, n))
    d = primary_mix(x) + origin
    return x, origin, d


def simulate(lib: FxlmsLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    x, origin, d = synthesize(n, fs, rng)

    state = lib.create(N_REF, N_SPK, N_ERR, ORDER, SEC_ORDER, STEP, SECONDARY)
    e = np.zeros((N_ERR, n), dtype=np.float64)
    y_delay = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    sample_x = np.zeros(N_REF, dtype=np.float64)
    sample_y = np.zeros(N_SPK, dtype=np.float64)
    sample_e = np.zeros(N_ERR, dtype=np.float64)
    try:
        for i in range(n):
            sample_x[:] = x[:, i]
            open_speaker_slot(y_delay)
            sample_e[:] = mic_error(y_delay, d[:, i])
            lib.core(state, sample_x, sample_e, sample_y)
            write_speaker(sample_y, y_delay)
            e[:, i] = sample_e
        w_c = lib.weights(state, N_SPK, N_REF, ORDER)
    finally:
        lib.destroy(state)

    e_np, w_np = numpy_fxlms(x, d, STEP)
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

    def time_pair(ax, a, b, ca, cb, title):
        ax.plot(n_axis, a[zoom0:zoom1], color=ca, lw=1.15, label="通道 0")
        ax.plot(n_axis, b[zoom0:zoom1], color=cb, lw=1.15, label="通道 1")
        ax.set_xlabel("时间 / ms")
        ax.set_ylabel("幅度")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=2)

    time_pair(axes[0], x[0], x[1], C_REF0, C_REF1, "两路参考信号")
    time_pair(axes[1], origin[0], origin[1], C_CLEAN0, C_CLEAN1, "两路纯净目标信号")
    time_pair(axes[2], d[0], d[1], C_TARGET0, C_TARGET1, "两路目标信号")

    win = 400
    kernel = np.ones(win) / win
    t_learn = (np.arange(n - win + 1) + win) / fs
    for q, color in enumerate((C_AFTER, "#e09f3e")):
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

    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    pair_color = (C_REF0, C_REF1, C_TARGET0, C_AFTER)
    group = ORDER + 2
    for i, ((l, p), color) in enumerate(zip(pairs, pair_color)):
        loc = np.arange(ORDER) + i * group
        axes[6].plot(
            loc, BEFORE[l, p], color=color, ls="none", marker="o", ms=5.5, label=f"扬声器 {l}，参考 {p}"
        )
        axes[6].plot(loc, w_c[l, p], color=color, lw=1.15, marker="x", ms=6)
        err = w_c[l, p] - BEFORE[l, p]
        axes[7].vlines(loc, 0.0, err, color=color, lw=1.15)
        axes[7].plot(loc, err, color=color, ls="none", marker="o", ms=4.5)
    centers = [i * group + (ORDER - 1) / 2 for i in range(4)]
    names = ["扬声器 0\n参考 0", "扬声器 0\n参考 1", "扬声器 1\n参考 0", "扬声器 1\n参考 1"]
    for ax in (axes[6], axes[7]):
        ax.set_xticks(centers)
        ax.set_xticklabels(names)
        ax.set_xlabel("控制滤波器")
        ax.set_ylabel("系数")
    axes[6].set_title(f"四条控制滤波器收敛到 b    系数 NMSE = {coeff_nmse:.2e}")
    axes[6].legend(loc="upper right", ncol=2)
    axes[7].axhline(0.0, color="#888888", lw=0.8)
    axes[7].set_title("收敛系数减去 b")
    path = save(fig, "03_fxlms.png")
    return path, origin_nmse, coeff_nmse, gap


def main() -> None:
    lib = FxlmsLib()
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap = simulate(lib, rng)
    for q, nmse in enumerate(origin_nmse):
        print(f"channel {q} residual vs origin NMSE {nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}")
    print(f"C vs NumPy {gap:.3e}")
    print(path)


if __name__ == "__main__":
    main()
