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
LMS_DIR = ROOT / "lms"
BUILD_DIR = ROOT / "output" / "lms"
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
N_REF = 2
N_ERR = 2
STEP = 0.002
REF_NOISE_STD = 0.35
ORIGIN_STD = 0.25
TONES = (40.0, 80.0, 120.0, 160.0)
AMP0 = (1.0, 0.7, 0.45, 0.25)
AMP1 = (0.75, 0.55, 0.30, 0.20)
PHASE1 = 0.6
PRIMARY = np.zeros((N_ERR, N_REF, ORDER), dtype=np.float64)
PRIMARY[0, 0] = [0.90, -0.55, 0.35, -0.20, 0.12, -0.06, 0.03, -0.01]
PRIMARY[0, 1] = [0.40, 0.22, -0.12, 0.06, -0.03, 0.01, 0.0, 0.0]
PRIMARY[1, 0] = [0.25, -0.18, 0.10, -0.05, 0.02, 0.0, 0.0, 0.0]
PRIMARY[1, 1] = [0.80, -0.30, 0.16, -0.08, 0.04, -0.02, 0.01, 0.0]


class LmsLib:
    def __init__(self) -> None:
        subprocess.check_call(
            ["cmake", "-S", str(LMS_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"]
        )
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR)])
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = CDLL(str(BUILD_DIR / f"liblms{ext}"))
        lib.LmsCreate.argtypes = [c_size_t, c_size_t, c_size_t, c_double]
        lib.LmsCreate.restype = c_void_p
        lib.LmsDestroy.argtypes = [c_void_p]
        lib.LmsStep.argtypes = [
            c_void_p,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.LmsStep.restype = c_int
        lib.LmsCopyWeights.argtypes = [c_void_p, POINTER(c_double)]
        lib.LmsCopyWeights.restype = c_int
        self.lib = lib

    def create(self, n_ref: int, n_err: int, order: int, step: float) -> c_void_p:
        state = self.lib.LmsCreate(
            c_size_t(n_ref), c_size_t(n_err), c_size_t(order), c_double(step)
        )
        if not state:
            raise RuntimeError("LmsCreate failed")
        return state

    def destroy(self, state: c_void_p) -> None:
        self.lib.LmsDestroy(state)

    def step(self, state: c_void_p, x: np.ndarray, d: np.ndarray, e: np.ndarray) -> None:
        rc = self.lib.LmsStep(
            state,
            x.ctypes.data_as(POINTER(c_double)),
            d.ctypes.data_as(POINTER(c_double)),
            e.ctypes.data_as(POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"LmsStep failed: {rc}")

    def weights(self, state: c_void_p, n_err: int, n_ref: int, order: int) -> np.ndarray:
        w = np.zeros(n_err * n_ref * order, dtype=np.float64)
        rc = self.lib.LmsCopyWeights(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError(f"LmsCopyWeights failed: {rc}")
        return w.reshape(n_err, n_ref, order)


def primary_mix(x: np.ndarray) -> np.ndarray:
    n_err, n_ref, _ = PRIMARY.shape
    d = np.zeros((n_err, x.shape[1]), dtype=np.float64)
    for q in range(n_err):
        for p in range(n_ref):
            d[q] += np.convolve(x[p], PRIMARY[q, p], mode="full")[: x.shape[1]]
    return d


def numpy_lms(x: np.ndarray, d: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """与 C 同一套逐点更新，只用于核对。"""
    n_ref, n = x.shape
    n_err = d.shape[0]
    order = ORDER
    w = np.zeros((n_err, n_ref, order), dtype=np.float64)
    delay = np.zeros((n_ref, order), dtype=np.float64)
    e = np.zeros((n_err, n), dtype=np.float64)
    for i in range(n):
        for p in range(n_ref):
            line = delay[p]
            for k in range(order - 1, 0, -1):
                line[k] = line[k - 1]
            line[0] = x[p, i]
        y = np.zeros(n_err, dtype=np.float64)
        for q in range(n_err):
            for p in range(n_ref):
                acc = 0.0
                for k in range(order):
                    acc += w[q, p, k] * delay[p, k]
                y[q] += acc
        e[:, i] = d[:, i] - y
        for q in range(n_err):
            gain = step * e[q, i]
            for p in range(n_ref):
                for k in range(order):
                    w[q, p, k] += gain * delay[p, k]
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


def simulate(lib: LmsLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    x, origin, d = synthesize(n, fs, rng)

    state = lib.create(N_REF, N_ERR, ORDER, STEP)
    e = np.zeros((N_ERR, n), dtype=np.float64)
    sample_x = np.zeros(N_REF, dtype=np.float64)
    sample_d = np.zeros(N_ERR, dtype=np.float64)
    sample_e = np.zeros(N_ERR, dtype=np.float64)
    try:
        for i in range(n):
            sample_x[:] = x[:, i]
            sample_d[:] = d[:, i]
            lib.step(state, sample_x, sample_d, sample_e)
            e[:, i] = sample_e
        w_c = lib.weights(state, N_ERR, N_REF, ORDER)
    finally:
        lib.destroy(state)

    e_np, w_np = numpy_lms(x, d, STEP)
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
    coeff_nmse = float(np.sum((w_c - PRIMARY) ** 2) / np.sum(PRIMARY**2))

    fig, axes = plt.subplots(6, 1, figsize=(9.2, 17.6), constrained_layout=True)
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
        axes[3].plot(t_learn, 10.0 * np.log10(np.maximum(power, 1e-20)), color=color, lw=1.2, label=f"通道 {q}")
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
    path = save(fig, "02_lms.png")
    return path, origin_nmse, coeff_nmse, gap


def main() -> None:
    lib = LmsLib()
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap = simulate(lib, rng)
    for q, nmse in enumerate(origin_nmse):
        print(f"channel {q} residual vs origin NMSE {nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}, C vs NumPy {gap:.3e}")
    print(path)


if __name__ == "__main__":
    main()
