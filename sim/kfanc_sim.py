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
LAMBDA = 1.0
MEAS_VAR = 0.0625
WEIGHT_VAR = 1.0
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
BEFORE = np.zeros((N_SPK, N_REF, ORDER), dtype=np.float64)
BEFORE[0, 0] = [0.90, -0.55, 0.35, -0.20, 0.12, -0.06, 0.03, -0.01]
BEFORE[0, 1] = [0.40, 0.22, -0.12, 0.06, -0.03, 0.01, 0.0, 0.0]
BEFORE[1, 0] = [0.25, -0.18, 0.10, -0.05, 0.02, 0.0, 0.0, 0.0]
BEFORE[1, 1] = [0.80, -0.30, 0.16, -0.08, 0.04, -0.02, 0.01, 0.0]
PRIMARY = np.zeros((N_ERR, N_REF, ORDER + SEC_ORDER - 1), dtype=np.float64)
for _q in range(N_ERR):
    for _p in range(N_REF):
        for _l in range(N_SPK):
            PRIMARY[_q, _p] += np.convolve(SECONDARY[_q, _l], BEFORE[_l, _p])


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
            c_size_t,
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
        lib.KfancCore.argtypes = [
            c_void_p,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.KfancCore.restype = c_int
        lib.KfancCopyWeights.argtypes = [c_void_p, POINTER(c_double)]
        lib.KfancCopyWeights.restype = c_int
        self.lib = lib

    def create(
        self,
        n_ref: int,
        n_spk: int,
        n_err: int,
        order: int,
        sec_order: int,
        lam: float,
        meas_var: float,
        weight_var: float,
        secondary: np.ndarray,
    ) -> c_void_p:
        sec = np.ascontiguousarray(secondary, dtype=np.float64)
        state = self.lib.KfancCreate(
            n_ref,
            n_spk,
            n_err,
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

    def core(self, state: c_void_p, x: np.ndarray, e: np.ndarray, y: np.ndarray) -> None:
        rc = self.lib.KfancCore(
            state,
            x.ctypes.data_as(POINTER(c_double)),
            e.ctypes.data_as(POINTER(c_double)),
            y.ctypes.data_as(POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError("KfancCore failed")

    def weights(self, state: c_void_p, n_spk: int, n_ref: int, order: int) -> np.ndarray:
        w = np.zeros(n_spk * n_ref * order, dtype=np.float64)
        rc = self.lib.KfancCopyWeights(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError("KfancCopyWeights failed")
        return w.reshape(n_spk, n_ref, order)

    def destroy(self, state: c_void_p) -> None:
        self.lib.KfancDestroy(state)


def mic_error(play: np.ndarray, d_now: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    n_err, n_spk, sec_order = secondary.shape
    ns = sec_order - 1
    err = np.zeros(n_err, dtype=np.float64)
    for q in range(n_err):
        anti = 0.0
        for spk in range(n_spk):
            anti += float(np.dot(secondary[q, spk, 1:], play[spk, :ns]))
        err[q] = d_now[q] - anti
    return err


def numpy_kfanc_order(
    x: np.ndarray,
    d: np.ndarray,
    secondary: np.ndarray,
    lam: float,
    meas_var: float,
    weight_var: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    sec = np.asarray(secondary, dtype=np.float64)
    if sec.ndim == 1:
        sec = sec.reshape(1, 1, -1)
    n_err, n_spk, sec_order = sec.shape
    n_ref = x.shape[0]
    n = x.shape[1]
    ns = sec_order - 1
    nw = n_ref * order
    nb = nw + ns
    nstate = n_spk * nb
    scale = lam ** -0.5
    w = np.zeros((n_spk, nw), dtype=np.float64)
    psi = np.zeros((n_spk, max(ns, 1)), dtype=np.float64)
    ref = np.zeros((n_ref, order), dtype=np.float64)
    play = np.zeros((n_spk, sec_order), dtype=np.float64)
    p = np.zeros((nstate, nstate), dtype=np.float64)
    for spk in range(n_spk):
        for i in range(nw):
            p[spk * nb + i, spk * nb + i] = weight_var
    e = np.zeros((n_err, n), dtype=np.float64)
    eye = np.eye(n_err, dtype=np.float64)
    for k in range(n):
        for pref in range(n_ref):
            ref[pref, 1:] = ref[pref, :-1]
            ref[pref, 0] = x[pref, k]
        err = mic_error(play, d[:, k], sec)
        e[:, k] = err
        y = np.zeros(n_spk, dtype=np.float64)
        for spk in range(n_spk):
            acc = 0.0
            for pref in range(n_ref):
                acc += float(np.dot(w[spk, pref * order : (pref + 1) * order], ref[pref]))
            y[spk] = acc
        eps = err.copy()
        ck = np.zeros((nstate, n_err), dtype=np.float64)
        ak = np.zeros((nstate, nstate), dtype=np.float64)
        for spk in range(n_spk):
            base = spk * nb
            for i in range(nw):
                ak[base + i, base + i] = scale
            if ns:
                for pref in range(n_ref):
                    ak[base + nw, base + pref * order : base + (pref + 1) * order] = ref[pref]
                for j in range(1, ns):
                    ak[base + nw + j, base + nw + j - 1] = 1.0
                eps = eps - np.array(
                    [
                        float(np.dot(sec[q, spk, 1:], psi[spk, :ns]))
                        for q in range(n_err)
                    ]
                )
            for q in range(n_err):
                if sec[q, spk, 0] != 0.0:
                    for pref in range(n_ref):
                        sl = slice(base + pref * order, base + (pref + 1) * order)
                        ck[sl, q] = sec[q, spk, 0] * ref[pref]
                if ns:
                    ck[base + nw : base + nw + ns, q] = sec[q, spk, 1:]
        pc = np.dot(p, ck)
        gram = meas_var * eye + np.dot(ck.T, pc)
        invg = np.linalg.inv(gram)
        pfilt = p - np.dot(np.dot(pc, invg), pc.T)
        work = np.dot(ak, pfilt)
        p = np.dot(work, ak.T)
        p = 0.5 * (p + p.T)
        gain = np.dot(np.dot(ak, pc), invg)
        for spk in range(n_spk):
            base = spk * nb
            w[spk] = scale * w[spk] + np.dot(gain[base : base + nw], eps)
            if ns:
                nxt = np.zeros(ns, dtype=np.float64)
                nxt[0] = float(np.dot(gain[base + nw], eps))
                for j in range(1, ns):
                    nxt[j] = psi[spk, j - 1] + float(np.dot(gain[base + nw + j], eps))
                psi[spk, :ns] = nxt
        for spk in range(n_spk):
            play[spk, 1:] = play[spk, :-1]
            play[spk, 0] = y[spk]
    return e, w.reshape(n_spk, n_ref, order)


def check_instance(lib: KfancLib) -> None:
    """单通道两步手算，w 停在 [1/2, 0]。交叉路径上另一只麦克风要带动这只扬声器。"""
    sec = np.array([0.0, 1.0], dtype=np.float64)
    state = lib.create(1, 1, 1, 2, 2, 1.0, 1.0, 1.0, sec)
    x = np.zeros(1, dtype=np.float64)
    e = np.zeros(1, dtype=np.float64)
    y = np.zeros(1, dtype=np.float64)
    try:
        x[:] = 1.0
        e[:] = 1.0
        lib.core(state, x, e, y)
        y0 = float(y[0])
        w0 = lib.weights(state, 1, 1, 2)
        x[:] = 0.0
        lib.core(state, x, e, y)
        y1 = float(y[0])
        w1 = lib.weights(state, 1, 1, 2)
    finally:
        lib.destroy(state)
    if abs(y0) > 1e-15 or abs(y1) > 1e-15:
        raise RuntimeError(f"instance outputs {y0}, {y1}")
    if np.max(np.abs(w0)) > 1e-15:
        raise RuntimeError(f"instance step 0 weights {w0}")
    expect = np.array([0.5, 0.0])
    if np.max(np.abs(w1.ravel() - expect)) > 1e-12:
        raise RuntimeError(f"instance step 1 weights {w1}")

    cross = np.zeros((2, 2, 2), dtype=np.float64)
    cross[0, 0] = [0.0, 1.0]
    cross[1, 0] = [0.0, 1.0]
    cross[1, 1] = [0.0, 1.0]
    state = lib.create(1, 2, 2, 1, 2, 1.0, 1.0, 1.0, cross)
    x1 = np.ones(1, dtype=np.float64)
    e1 = np.zeros(2, dtype=np.float64)
    y1v = np.zeros(2, dtype=np.float64)
    try:
        lib.core(state, x1, e1, y1v)
        e1[:] = (0.0, 1.0)
        lib.core(state, x1, e1, y1v)
        w_cross = lib.weights(state, 2, 1, 1)
    finally:
        lib.destroy(state)
    if abs(float(w_cross[0, 0, 0])) < 1e-9:
        raise RuntimeError(f"speaker 0 ignored mic 1, w={w_cross}")


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
    d = np.zeros((N_ERR, n), dtype=np.float64)
    for q in range(N_ERR):
        for p in range(N_REF):
            d[q] += np.convolve(x[p], PRIMARY[q, p], mode="full")[:n]
    d += origin
    return x, origin, d


def simulate(lib: KfancLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    x, origin, d = synthesize(n, fs, rng)
    state = lib.create(
        N_REF, N_SPK, N_ERR, ORDER, SEC_ORDER, LAMBDA, MEAS_VAR, WEIGHT_VAR, SECONDARY
    )
    e = np.zeros((N_ERR, n), dtype=np.float64)
    play = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    sample_x = np.zeros(N_REF, dtype=np.float64)
    sample_e = np.zeros(N_ERR, dtype=np.float64)
    sample_y = np.zeros(N_SPK, dtype=np.float64)
    try:
        for i in range(n):
            sample_x[:] = x[:, i]
            sample_e[:] = mic_error(play, d[:, i], SECONDARY)
            lib.core(state, sample_x, sample_e, sample_y)
            play[:, 1:] = play[:, :-1]
            play[:, 0] = sample_y
            e[:, i] = sample_e
        w_c = lib.weights(state, N_SPK, N_REF, ORDER)
    finally:
        lib.destroy(state)

    e_np, w_np = numpy_kfanc_order(x, d, SECONDARY, LAMBDA, MEAS_VAR, WEIGHT_VAR, ORDER)
    gap = float(np.max(np.abs(w_c - w_np)))
    if gap > 1e-8:
        raise RuntimeError(f"C and NumPy weights differ by {gap}")
    if np.max(np.abs(e - e_np)) > 1e-8:
        raise RuntimeError("C and NumPy residuals differ")

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
    axes[4].plot(n_axis, origin[0, zoom0:zoom1], color=C_CLEAN0, lw=1.15, ls="--", label="纯净目标信号")
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
    err = w_c - BEFORE
    for i, ((spk, pref), color) in enumerate(zip(pairs, pair_color)):
        loc = np.arange(ORDER) + i * group
        axes[6].plot(loc, BEFORE[spk, pref], color=color, ls="none", marker="o", ms=4.5)
        axes[6].plot(loc, w_c[spk, pref], color=color, lw=1.05, marker="x", ms=5)
        axes[7].vlines(loc, 0.0, err[spk, pref], color=color, lw=1.05)
        axes[7].plot(loc, err[spk, pref], color=color, ls="none", marker="o", ms=3.5)
    for ax in (axes[6], axes[7]):
        ax.set_xlabel("抽头")
        ax.set_ylabel("系数")
    axes[6].set_title(f"四条控制滤波器收敛到 b    系数 NMSE = {coeff_nmse:.2e}")
    axes[7].axhline(0.0, color="#888888", lw=0.8)
    axes[7].set_title("收敛系数减去 b")
    path = save(fig, "08_kfanc.png")
    return path, origin_nmse, coeff_nmse, gap


def main() -> None:
    lib = KfancLib()
    check_instance(lib)
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap = simulate(lib, rng)
    for q, nmse in enumerate(origin_nmse):
        print(f"channel {q} residual vs origin NMSE {nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}")
    print(f"C vs NumPy {gap:.3e}")
    print(path)


if __name__ == "__main__":
    main()
