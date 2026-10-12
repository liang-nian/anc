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
FD_DIR = ROOT / "fdfxlms"
BUILD_DIR = ROOT / "output" / "fdfxlms"
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

BLOCK = 8
N_REF = 2
N_SPK = 2
N_ERR = 2
STEP = 0.003
REF_NOISE_STD = 0.35
ORIGIN_STD = 0.25
TONES = (40.0, 80.0, 120.0, 160.0)
AMP0 = (1.0, 0.7, 0.45, 0.25)
AMP1 = (0.75, 0.55, 0.30, 0.20)
PHASE1 = 0.6
SEC_TAIL = np.zeros((N_ERR, N_SPK, 2), dtype=np.float64)
SEC_TAIL[0, 0] = [0.90, 0.20]
SEC_TAIL[0, 1] = [0.15, 0.05]
SEC_TAIL[1, 0] = [0.12, -0.04]
SEC_TAIL[1, 1] = [0.85, 0.15]
SEC_ORDER = BLOCK + SEC_TAIL.shape[-1]
SECONDARY = np.zeros((N_ERR, N_SPK, SEC_ORDER), dtype=np.float64)
SECONDARY[:, :, BLOCK:] = SEC_TAIL
BEFORE = np.zeros((N_SPK, N_REF, BLOCK), dtype=np.float64)
BEFORE[0, 0] = [0.90, -0.55, 0.35, -0.20, 0.12, -0.06, 0.03, -0.01]
BEFORE[0, 1] = [0.40, 0.22, -0.12, 0.06, -0.03, 0.01, 0.0, 0.0]
BEFORE[1, 0] = [0.25, -0.18, 0.10, -0.05, 0.02, 0.0, 0.0, 0.0]
BEFORE[1, 1] = [0.80, -0.30, 0.16, -0.08, 0.04, -0.02, 0.01, 0.0]
PRIMARY = np.zeros((N_ERR, N_REF, BEFORE.shape[-1] + SEC_ORDER - 1), dtype=np.float64)
for _q in range(N_ERR):
    for _p in range(N_REF):
        for _l in range(N_SPK):
            PRIMARY[_q, _p] += np.convolve(SECONDARY[_q, _l], BEFORE[_l, _p])


class FdfxlmsLib:
    def __init__(self) -> None:
        subprocess.check_call(
            ["cmake", "-S", str(FD_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"]
        )
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR)])
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = CDLL(str(BUILD_DIR / f"libfdfxlms{ext}"))
        lib.FdfxlmsCreate.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_double,
            POINTER(c_double),
        ]
        lib.FdfxlmsCreate.restype = c_void_p
        lib.FdfxlmsDestroy.argtypes = [c_void_p]
        lib.FdfxlmsCore.argtypes = [
            c_void_p,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.FdfxlmsCore.restype = c_int
        lib.FdfxlmsCopyWeights.argtypes = [c_void_p, POINTER(c_double)]
        lib.FdfxlmsCopyWeights.restype = c_int
        self.lib = lib

    def create(self, secondary: np.ndarray) -> c_void_p:
        sec = np.ascontiguousarray(secondary, dtype=np.float64)
        state = self.lib.FdfxlmsCreate(
            c_size_t(N_REF),
            c_size_t(N_SPK),
            c_size_t(N_ERR),
            c_size_t(BLOCK),
            c_size_t(SEC_ORDER),
            c_double(STEP),
            sec.ctypes.data_as(POINTER(c_double)),
        )
        if not state:
            raise RuntimeError("FdfxlmsCreate failed")
        return state

    def destroy(self, state: c_void_p) -> None:
        self.lib.FdfxlmsDestroy(state)

    def core(self, state: c_void_p, x: np.ndarray, e: np.ndarray, y: np.ndarray) -> None:
        rc = self.lib.FdfxlmsCore(
            state,
            x.ctypes.data_as(POINTER(c_double)),
            e.ctypes.data_as(POINTER(c_double)),
            y.ctypes.data_as(POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"FdfxlmsCore failed: {rc}")

    def weights(self, state: c_void_p) -> np.ndarray:
        w = np.zeros(N_SPK * N_REF * BLOCK, dtype=np.float64)
        rc = self.lib.FdfxlmsCopyWeights(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError(f"FdfxlmsCopyWeights failed: {rc}")
        return w.reshape(N_SPK, N_REF, BLOCK)


def primary_mix(x: np.ndarray) -> np.ndarray:
    n_err, n_ref, _ = PRIMARY.shape
    d = np.zeros((n_err, x.shape[1]), dtype=np.float64)
    for q in range(n_err):
        for p in range(n_ref):
            d[q] += np.convolve(x[p], PRIMARY[q, p], mode="full")[: x.shape[1]]
    return d


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def fx_length(block: int, sec_order: int) -> int:
    nfft = block * 2
    sec_nfft = next_pow2(block + sec_order - 1)
    return sec_nfft if sec_nfft > nfft else nfft


def block_error(y_played: np.ndarray, d_blk: np.ndarray, t0: int) -> np.ndarray:
    """这一块误差由已经播出的扬声器历史算出。次级路径前一块为 0。"""
    e_blk = np.zeros((N_ERR, BLOCK), dtype=np.float64)
    for i in range(BLOCK):
        tt = t0 + i
        for q in range(N_ERR):
            acc = 0.0
            for l in range(N_SPK):
                for j in range(SEC_ORDER):
                    src = tt - j
                    if src >= 0:
                        acc += SECONDARY[q, l, j] * y_played[l, src]
            e_blk[q, i] = d_blk[q, i] - acc
    return e_blk


def numpy_fdfxlms(x: np.ndarray, d: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """与 C 同一套按块的频域预滤波、更新和输出，只用于核对。"""
    n_ref, n = x.shape
    block = BLOCK
    nfft = block * 2
    sec_nfft = fx_length(block, SEC_ORDER)
    secondary_spec = np.zeros((N_ERR, N_SPK, sec_nfft), dtype=np.complex128)
    for q in range(N_ERR):
        for l in range(N_SPK):
            pad = np.zeros(sec_nfft, dtype=np.float64)
            pad[:SEC_ORDER] = SECONDARY[q, l]
            secondary_spec[q, l] = np.fft.fft(pad)
    weights = np.zeros((N_SPK, n_ref, nfft), dtype=np.complex128)
    x_hist = np.zeros((n_ref, sec_nfft), dtype=np.float64)
    xp_hist = np.zeros((N_ERR, N_SPK, n_ref, nfft), dtype=np.float64)
    y_played = np.zeros((N_SPK, n), dtype=np.float64)
    e = np.zeros((N_ERR, n), dtype=np.float64)
    for b in range(n // block):
        sl = slice(b * block, (b + 1) * block)
        e_blk = block_error(y_played, d[:, sl], b * block)
        for p in range(n_ref):
            x_hist[p, :-block] = x_hist[p, block:]
            x_hist[p, -block:] = x[p, sl]
        for p in range(n_ref):
            spec_x = np.fft.fft(x_hist[p])
            for q in range(N_ERR):
                for l in range(N_SPK):
                    filtered = np.fft.ifft(secondary_spec[q, l] * spec_x).real
                    xp_hist[q, l, p, :-block] = xp_hist[q, l, p, block:]
                    xp_hist[q, l, p, -block:] = filtered[-block:]
        spec_e = []
        for q in range(N_ERR):
            frame = np.zeros(nfft, dtype=np.float64)
            frame[block:] = e_blk[q]
            spec_e.append(np.fft.fft(frame))
        for l in range(N_SPK):
            for p in range(n_ref):
                grad_spec = np.zeros(nfft, dtype=np.complex128)
                for q in range(N_ERR):
                    spec_xp = np.fft.fft(xp_hist[q, l, p])
                    grad_spec += spec_e[q] * np.conj(spec_xp)
                grad = np.fft.ifft(grad_spec).real
                grad[block:] = 0.0
                weights[l, p] += step * np.fft.fft(grad)
        ref_spec = [np.fft.fft(x_hist[p, -nfft:]) for p in range(n_ref)]
        for l in range(N_SPK):
            spec_y = np.zeros(nfft, dtype=np.complex128)
            for p in range(n_ref):
                spec_y += weights[l, p] * ref_spec[p]
            y_played[l, sl] = np.fft.ifft(spec_y).real[-block:]
        e[:, sl] = e_blk
    w = np.zeros((N_SPK, n_ref, block), dtype=np.float64)
    for l in range(N_SPK):
        for p in range(n_ref):
            w[l, p] = np.fft.ifft(weights[l, p]).real[:block]
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


def simulate(lib: FdfxlmsLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    x, origin, d = synthesize(n, fs, rng)

    state = lib.create(SECONDARY)
    e = np.zeros((N_ERR, n), dtype=np.float64)
    y_played = np.zeros((N_SPK, n), dtype=np.float64)
    x_blk = np.zeros(N_REF * BLOCK, dtype=np.float64)
    e_blk = np.zeros(N_ERR * BLOCK, dtype=np.float64)
    y_blk = np.zeros(N_SPK * BLOCK, dtype=np.float64)
    try:
        for b in range(n // BLOCK):
            sl = slice(b * BLOCK, (b + 1) * BLOCK)
            measured = block_error(y_played, d[:, sl], b * BLOCK)
            for p in range(N_REF):
                x_blk[p * BLOCK : (p + 1) * BLOCK] = x[p, sl]
            for q in range(N_ERR):
                e_blk[q * BLOCK : (q + 1) * BLOCK] = measured[q]
            lib.core(state, x_blk, e_blk, y_blk)
            for l in range(N_SPK):
                y_played[l, sl] = y_blk[l * BLOCK : (l + 1) * BLOCK]
            for q in range(N_ERR):
                e[q, sl] = measured[q]
        w_c = lib.weights(state)
    finally:
        lib.destroy(state)

    e_np, w_np = numpy_fdfxlms(x, d, STEP)
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

    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    pair_color = (C_REF0, C_REF1, C_TARGET0, C_AFTER)
    group = BLOCK + 2
    for i, ((l, p), color) in enumerate(zip(pairs, pair_color)):
        loc = np.arange(BLOCK) + i * group
        axes[6].plot(loc, BEFORE[l, p], color=color, ls="none", marker="o", ms=5.5)
        axes[6].plot(loc, w_c[l, p], color=color, lw=1.15, marker="x", ms=6)
        err = w_c[l, p] - BEFORE[l, p]
        axes[7].vlines(loc, 0.0, err, color=color, lw=1.15)
        axes[7].plot(loc, err, color=color, ls="none", marker="o", ms=4.5)
    centers = [i * group + (BLOCK - 1) / 2 for i in range(4)]
    names = ["扬声器 0\n参考 0", "扬声器 0\n参考 1", "扬声器 1\n参考 0", "扬声器 1\n参考 1"]
    for ax in (axes[6], axes[7]):
        ax.set_xticks(centers)
        ax.set_xticklabels(names)
        ax.set_xlabel("控制滤波器")
        ax.set_ylabel("系数")
    axes[6].set_title(f"四条控制滤波器收敛到 b    系数 NMSE = {coeff_nmse:.2e}")
    axes[7].axhline(0.0, color="#888888", lw=0.8)
    axes[7].set_title("收敛系数减去 b")
    path = save(fig, "06_fdfxlms.png")
    return path, origin_nmse, coeff_nmse, gap


def main() -> None:
    lib = FdfxlmsLib()
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap = simulate(lib, rng)
    for q, nmse in enumerate(origin_nmse):
        print(f"channel {q} residual vs origin NMSE {nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}")
    print(f"C vs NumPy {gap:.3e}")
    print(path)


if __name__ == "__main__":
    main()
