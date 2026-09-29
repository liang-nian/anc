#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from ctypes import CDLL, POINTER, byref, c_double, c_int, c_size_t
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIENER_DIR = ROOT / "wiener"
BUILD_DIR = ROOT / "output" / "wiener"
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

C_REF = "#222222"
C_CLEAN = "#2a9d8f"
C_TARGET = "#5b4b8a"
C_AFTER = "#c0392b"

PRIMARY = np.array(
    [0.90, -0.55, 0.35, -0.20, 0.12, -0.08, 0.05, -0.03, 0.02, -0.01, 0.005, -0.002],
    dtype=np.float64,
)
TONES = ((1.0, 40.0), (0.7, 80.0), (0.45, 120.0), (0.25, 160.0))
REF_NOISE_STD = 0.35
ORIGIN_STD = 0.25


class WienerLib:
    def __init__(self) -> None:
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(WIENER_DIR),
                "-B",
                str(BUILD_DIR),
                "-DCMAKE_BUILD_TYPE=Release",
            ]
        )
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR)])
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = CDLL(str(BUILD_DIR / f"libwiener{ext}"))
        lib.WienerFirDesign.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            c_size_t,
            c_size_t,
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.WienerFirDesign.restype = c_int
        lib.WienerFirApply.argtypes = [
            POINTER(c_double),
            c_size_t,
            POINTER(c_double),
            c_size_t,
            POINTER(c_double),
        ]
        lib.WienerFirApply.restype = c_int
        self.lib = lib

    @staticmethod
    def _as(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=np.float64)

    def fir_design(self, x: np.ndarray, d: np.ndarray, order: int):
        x = self._as(x)
        d = self._as(d)
        w = np.zeros(order, dtype=np.float64)
        mse = c_double()
        rc = self.lib.WienerFirDesign(
            x.ctypes.data_as(POINTER(c_double)),
            d.ctypes.data_as(POINTER(c_double)),
            c_size_t(x.size),
            c_size_t(order),
            w.ctypes.data_as(POINTER(c_double)),
            byref(mse),
        )
        if rc != 0:
            raise RuntimeError(f"WienerFirDesign failed: {rc}")
        return w, float(mse.value)

    def fir_apply(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x = self._as(x)
        w = self._as(w)
        y = np.zeros(x.size, dtype=np.float64)
        rc = self.lib.WienerFirApply(
            x.ctypes.data_as(POINTER(c_double)),
            c_size_t(x.size),
            w.ctypes.data_as(POINTER(c_double)),
            c_size_t(w.size),
            y.ctypes.data_as(POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"WienerFirApply failed: {rc}")
        return y


def numpy_fir(x: np.ndarray, d: np.ndarray, order: int) -> np.ndarray:
    """与 C 同一套样本正规方程，只用于核对系数。"""
    start = order - 1
    count = float(x.size - start)
    gram = np.empty((order, order), dtype=np.float64)
    p = np.empty(order, dtype=np.float64)
    for i in range(order):
        xi = x[start - i : x.size - i]
        p[i] = np.dot(d[start:], xi) / count
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = np.dot(xi, x[start - j : x.size - j]) / count
    return np.linalg.solve(gram, p)


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


def order_tones(n: int, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    return sum(amp * np.sin(2.0 * np.pi * freq * t) for amp, freq in TONES)


def simulate(lib: WienerLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    order = PRIMARY.size

    x = order_tones(n, fs) + REF_NOISE_STD * rng.normal(size=n)
    arrived = np.convolve(x, PRIMARY, mode="full")[:n]
    origin = ORIGIN_STD * rng.normal(size=n)
    d = arrived + origin

    w_c, train_mse = lib.fir_design(x, d, order)
    w_np = numpy_fir(x, d, order)
    coeff_gap = float(np.max(np.abs(w_c - w_np)))
    if coeff_gap > 1e-8:
        raise RuntimeError(f"C and NumPy coefficients differ by {coeff_gap}")

    y = lib.fir_apply(x, w_c)
    residual = d - y
    start = order - 1
    gap = residual[start:] - origin[start:]
    origin_nmse = float(np.sum(gap**2) / np.sum(origin[start:] ** 2))
    coeff_nmse = float(np.sum((w_c - PRIMARY) ** 2) / np.sum(PRIMARY**2))
    if origin_nmse > 1e-3:
        raise RuntimeError(f"residual does not match the original signal, NMSE {origin_nmse}")

    fig, axes = plt.subplots(5, 1, figsize=(9.2, 14.8), constrained_layout=True)
    zoom0, zoom1 = 800, 1100
    n_axis = (np.arange(zoom0, zoom1) - zoom0) / fs * 1e3

    def time_panel(ax, y, color, title, ylabel="幅度"):
        ax.plot(n_axis, y[zoom0:zoom1], color=color, lw=1.15)
        ax.set_xlabel("时间 / ms")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    time_panel(axes[0], x, C_REF, "参考信号")
    time_panel(axes[1], origin, C_CLEAN, "纯净目标信号")
    time_panel(axes[2], d, C_TARGET, "目标信号")

    axes[3].plot(n_axis, d[zoom0:zoom1], color=C_TARGET, lw=1.15, label="目标信号")
    axes[3].plot(n_axis, residual[zoom0:zoom1], color=C_AFTER, lw=1.15, label="抵消后")
    axes[3].plot(n_axis, origin[zoom0:zoom1], color=C_CLEAN, lw=1.15, ls="--", label="纯净目标信号")
    axes[3].set_xlabel("时间 / ms")
    axes[3].set_ylabel("幅度")
    axes[3].set_title(f"抵消后的误差贴住纯净目标信号    NMSE = {origin_nmse:.2e}")
    axes[3].legend(loc="upper right", ncol=3)

    freq, p_before = welch_psd(d[start:], fs)
    _, p_after = welch_psd(residual[start:], fs)
    _, p_origin = welch_psd(origin[start:], fs)

    def to_db(p: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(p, 1e-20))

    axes[4].plot(freq, to_db(p_before), color=C_TARGET, lw=1.2, label="目标信号")
    axes[4].plot(freq, to_db(p_after), color=C_AFTER, lw=1.2, label="抵消后")
    axes[4].plot(freq, to_db(p_origin), color=C_CLEAN, lw=1.15, ls="--", label="纯净目标信号")
    axes[4].set_xlim(0, 400)
    axes[4].set_xlabel("频率 / Hz")
    axes[4].set_ylabel("功率谱 / dB")
    axes[4].set_title("阶次被消掉之后，剩下的谱就是纯净目标信号")
    axes[4].legend(loc="upper right", ncol=3)
    path = save(fig, "01_signals.png")
    return path, origin_nmse, coeff_nmse, coeff_gap, train_mse


def main() -> None:
    lib = WienerLib()
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap, train_mse = simulate(lib, rng)
    print(f"residual vs origin NMSE {origin_nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}, C vs NumPy {gap:.3e}")
    print(f"train MSE {train_mse:.6f}, origin power {ORIGIN_STD ** 2:.6f}")
    print(path)


if __name__ == "__main__":
    main()
