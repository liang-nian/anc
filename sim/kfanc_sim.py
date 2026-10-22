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
KFANC_DIR = ROOT / "kfanc"
BUILD_DIR = ROOT / "output" / "kfanc"
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

ORDER = 8
SEC_ORDER = 3
LAMBDA = 1.0
MEAS_VAR = 0.0625
WEIGHT_VAR = 1.0
REF_NOISE_STD = 0.35
ORIGIN_STD = 0.25
TONES = (40.0, 80.0, 120.0, 160.0)
AMPS = (1.0, 0.7, 0.45, 0.25)
SECONDARY = np.array([0.0, 0.90, 0.20], dtype=np.float64)
BEFORE = np.array([0.90, -0.55, 0.35, -0.20, 0.12, -0.06, 0.03, -0.01], dtype=np.float64)
PRIMARY = np.convolve(SECONDARY, BEFORE)


class KfancLib:
    def __init__(self) -> None:
        subprocess.check_call(
            ["cmake", "-S", str(KFANC_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"]
        )
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR)])
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = CDLL(str(BUILD_DIR / f"libkfanc{ext}"))
        lib.KfancCreate.argtypes = [
            c_size_t,
            c_size_t,
            c_double,
            c_double,
            c_double,
            POINTER(c_double),
        ]
        lib.KfancCreate.restype = c_void_p
        lib.KfancDestroy.argtypes = [c_void_p]
        lib.KfancDestroy.restype = None
        lib.KfancCore.argtypes = [c_void_p, c_double, c_double, POINTER(c_double)]
        lib.KfancCore.restype = c_int
        lib.KfancCopyWeights.argtypes = [c_void_p, POINTER(c_double)]
        lib.KfancCopyWeights.restype = c_int
        self.lib = lib

    def create(
        self,
        order: int,
        sec_order: int,
        lam: float,
        meas_var: float,
        weight_var: float,
        secondary: np.ndarray,
    ) -> c_void_p:
        sec = np.ascontiguousarray(secondary, dtype=np.float64)
        state = self.lib.KfancCreate(
            order,
            sec_order,
            lam,
            meas_var,
            weight_var,
            sec.ctypes.data_as(POINTER(c_double)),
        )
        if not state:
            raise RuntimeError("KfancCreate failed")
        return state

    def core(self, state: c_void_p, x: float, e: float) -> float:
        y = np.zeros(1, dtype=np.float64)
        rc = self.lib.KfancCore(state, x, e, y.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError("KfancCore failed")
        return float(y[0])

    def weights(self, state: c_void_p, order: int) -> np.ndarray:
        w = np.zeros(order, dtype=np.float64)
        rc = self.lib.KfancCopyWeights(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError("KfancCopyWeights failed")
        return w

    def destroy(self, state: c_void_p) -> None:
        self.lib.KfancDestroy(state)


def numpy_kfanc_order(
    x: np.ndarray,
    d: np.ndarray,
    secondary: np.ndarray,
    lam: float,
    meas_var: float,
    weight_var: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    sec = np.asarray(secondary, dtype=np.float64)
    ns = sec.size - 1
    nstate = order + ns
    scale = lam ** -0.5
    w = np.zeros(order, dtype=np.float64)
    psi = np.zeros(ns, dtype=np.float64)
    r = np.zeros(order, dtype=np.float64)
    play = np.zeros(sec.size, dtype=np.float64)
    p = np.zeros((nstate, nstate), dtype=np.float64)
    for i in range(order):
        p[i, i] = weight_var
    e = np.zeros(n, dtype=np.float64)
    for k in range(n):
        r[1:] = r[:-1]
        r[0] = x[k]
        y = float(np.dot(w, r))
        play[1:] = play[:-1]
        play[0] = 0.0
        e[k] = d[k] - float(np.dot(sec, play))
        play[0] = y
        eps = e[k] - float(np.dot(sec[1:], psi))
        ck = np.zeros(nstate, dtype=np.float64)
        if sec[0] != 0.0:
            ck[:order] = sec[0] * r
        if ns:
            ck[order:] = sec[1:]
        ak = np.zeros((nstate, nstate), dtype=np.float64)
        for i in range(order):
            ak[i, i] = scale
        if ns:
            ak[order, :order] = r
            for i in range(1, ns):
                ak[order + i, order + i - 1] = 1.0
        pc = np.dot(p, ck)
        re = meas_var + float(np.dot(ck, pc))
        inv = 1.0 / re
        pfilt = p - np.outer(pc, pc) * inv
        work = np.dot(ak, pfilt)
        p = np.dot(work, ak.T)
        p = 0.5 * (p + p.T)
        gain = np.dot(ak, pc) * inv
        w = scale * w + gain[:order] * eps
        if ns:
            nxt = np.zeros(ns, dtype=np.float64)
            nxt[0] = gain[order] * eps
            for i in range(1, ns):
                nxt[i] = psi[i - 1] + gain[order + i] * eps
            psi = nxt
    return e, w


def check_instance(lib: KfancLib) -> None:
    """笔记第二节：两步手算，w 停在 [1/2, 0]。"""
    sec = np.array([0.0, 1.0], dtype=np.float64)
    state = lib.create(2, 2, 1.0, 1.0, 1.0, sec)
    try:
        y0 = lib.core(state, 1.0, 1.0)
        w0 = lib.weights(state, 2)
        y1 = lib.core(state, 0.0, 1.0)
        w1 = lib.weights(state, 2)
    finally:
        lib.destroy(state)
    if abs(y0) > 1e-15 or abs(y1) > 1e-15:
        raise RuntimeError(f"instance outputs {y0}, {y1}")
    if np.max(np.abs(w0)) > 1e-15:
        raise RuntimeError(f"instance step 0 weights {w0}")
    expect = np.array([0.5, 0.0])
    if np.max(np.abs(w1 - expect)) > 1e-12:
        raise RuntimeError(f"instance step 1 weights {w1}")


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
    x = np.zeros(n, dtype=np.float64)
    for amp, freq in zip(AMPS, TONES):
        x += amp * np.sin(2.0 * np.pi * freq * t)
    x += REF_NOISE_STD * rng.normal(size=n)
    origin = ORIGIN_STD * rng.normal(size=n)
    d = np.convolve(PRIMARY, x)[:n] + origin
    return x, origin, d


def simulate(lib: KfancLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    x, origin, d = synthesize(n, fs, rng)
    state = lib.create(ORDER, SEC_ORDER, LAMBDA, MEAS_VAR, WEIGHT_VAR, SECONDARY)
    e = np.zeros(n, dtype=np.float64)
    play = np.zeros(SEC_ORDER, dtype=np.float64)
    try:
        for i in range(n):
            play[1:] = play[:-1]
            play[0] = 0.0
            err = float(d[i] - np.dot(SECONDARY, play))
            y = lib.core(state, float(x[i]), err)
            play[0] = y
            e[i] = err
        w_c = lib.weights(state, ORDER)
    finally:
        lib.destroy(state)

    e_np, w_np = numpy_kfanc_order(x, d, SECONDARY, LAMBDA, MEAS_VAR, WEIGHT_VAR, ORDER)
    gap = float(np.max(np.abs(w_c - w_np)))
    if gap > 1e-8:
        raise RuntimeError(f"C and NumPy weights differ by {gap}")
    if np.max(np.abs(e - e_np)) > 1e-8:
        raise RuntimeError("C and NumPy residuals differ")

    tail = slice(-4000, None)
    num = float(np.sum((e[tail] - origin[tail]) ** 2))
    den = float(np.sum(origin[tail] ** 2))
    origin_nmse = num / den
    if origin_nmse > 5e-2:
        raise RuntimeError(f"residual does not track the original signal, NMSE {origin_nmse}")
    coeff_nmse = float(np.sum((w_c - BEFORE) ** 2) / np.sum(BEFORE**2))

    fig, axes = plt.subplots(8, 1, figsize=(9.2, 23.2), constrained_layout=True)
    zoom0, zoom1 = n - 300, n
    n_axis = (np.arange(zoom0, zoom1) - zoom0) / fs * 1e3

    def time_one(ax, sig, color, title):
        ax.plot(n_axis, sig[zoom0:zoom1], color=color, lw=1.15)
        ax.set_xlabel("时间 / ms")
        ax.set_ylabel("幅度")
        ax.set_title(title)

    time_one(axes[0], x, C_REF, "参考信号")
    time_one(axes[1], origin, C_CLEAN, "纯净目标信号")
    time_one(axes[2], d, C_TARGET, "目标信号")

    win = 400
    kernel = np.ones(win) / win
    t_learn = (np.arange(n - win + 1) + win) / fs
    power = np.convolve(e**2, kernel, mode="valid")
    axes[3].plot(t_learn, 10.0 * np.log10(np.maximum(power, 1e-20)), color=C_AFTER, lw=1.2, label="误差")
    origin_db = 10.0 * np.log10(float(np.mean(origin**2)))
    axes[3].axhline(origin_db, color=C_CLEAN, lw=1.15, ls="--", label="纯净目标信号")
    axes[3].set_xlabel("时间 / s")
    axes[3].set_ylabel("功率 / dB")
    axes[3].set_title("误差功率往纯净目标信号靠拢")
    axes[3].legend(loc="upper right")

    axes[4].plot(n_axis, d[zoom0:zoom1], color=C_TARGET, lw=1.15, label="目标信号")
    axes[4].plot(n_axis, e[zoom0:zoom1], color=C_AFTER, lw=1.15, label="抵消后")
    axes[4].plot(n_axis, origin[zoom0:zoom1], color=C_CLEAN, lw=1.15, ls="--", label="纯净目标信号")
    axes[4].set_xlabel("时间 / ms")
    axes[4].set_ylabel("幅度")
    axes[4].set_title(f"收敛后，误差贴住纯净目标信号    NMSE = {origin_nmse:.2e}")
    axes[4].legend(loc="upper right", ncol=3)

    spec0 = slice(-8000, None)
    freq, p_before = welch_psd(d[spec0], fs)
    _, p_after = welch_psd(e[spec0], fs)
    _, p_origin = welch_psd(origin[spec0], fs)

    def to_db(p: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(p, 1e-20))

    axes[5].plot(freq, to_db(p_before), color=C_TARGET, lw=1.2, label="目标信号")
    axes[5].plot(freq, to_db(p_after), color=C_AFTER, lw=1.2, label="抵消后")
    axes[5].plot(freq, to_db(p_origin), color=C_CLEAN, lw=1.15, ls="--", label="纯净目标信号")
    axes[5].set_xlim(0, 400)
    axes[5].set_xlabel("频率 / Hz")
    axes[5].set_ylabel("功率谱 / dB")
    axes[5].set_title("阶次被消掉之后，剩下的谱就是纯净目标信号")
    axes[5].legend(loc="upper right", ncol=3)

    loc = np.arange(ORDER)
    axes[6].plot(loc, BEFORE, color=C_TARGET, ls="none", marker="o", ms=5.5, label="b")
    axes[6].plot(loc, w_c, color=C_AFTER, lw=1.15, marker="x", ms=6, label="估计")
    err = w_c - BEFORE
    axes[7].vlines(loc, 0.0, err, color=C_AFTER, lw=1.15)
    axes[7].plot(loc, err, color=C_AFTER, ls="none", marker="o", ms=4.5)
    for ax in (axes[6], axes[7]):
        ax.set_xlabel("抽头")
        ax.set_ylabel("系数")
    axes[6].set_title(f"控制滤波器收敛到 b    系数 NMSE = {coeff_nmse:.2e}")
    axes[6].legend(loc="upper right")
    axes[7].axhline(0.0, color="#888888", lw=0.8)
    axes[7].set_title("收敛系数减去 b")
    path = save(fig, "08_kfanc.png")
    return path, origin_nmse, coeff_nmse, gap


def main() -> None:
    lib = KfancLib()
    check_instance(lib)
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap = simulate(lib, rng)
    print(f"residual vs origin NMSE {origin_nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}")
    print(f"C vs NumPy {gap:.3e}")
    print(path)


if __name__ == "__main__":
    main()
