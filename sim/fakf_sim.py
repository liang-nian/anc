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
SRC_DIR = ROOT / "fakf"
BUILD_DIR = ROOT / "output" / "fakf"
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
PROC_VAR = 0.0
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


class FakfLib:
    def __init__(self) -> None:
        subprocess.check_call(
            ["cmake", "-S", str(SRC_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"]
        )
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR)])
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib = CDLL(str(BUILD_DIR / f"libfakf{ext}"))
        lib.FakfCreate.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_double,
            c_double,
            c_double,
            c_double,
            POINTER(c_double),
        ]
        lib.FakfCreate.restype = c_void_p
        lib.FakfDestroy.argtypes = [c_void_p]
        lib.FakfCore.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        lib.FakfCore.restype = c_int
        lib.FakfCopyWeights.argtypes = [c_void_p, POINTER(c_double)]
        lib.FakfCopyWeights.restype = c_int
        self.lib = lib

    def create(self, n_ref, n_spk, n_err, order, sec_order, lam, meas, proc, weight, secondary) -> c_void_p:
        sec = np.ascontiguousarray(secondary, dtype=np.float64)
        state = self.lib.FakfCreate(
            c_size_t(n_ref),
            c_size_t(n_spk),
            c_size_t(n_err),
            c_size_t(order),
            c_size_t(sec_order),
            c_double(lam),
            c_double(meas),
            c_double(proc),
            c_double(weight),
            sec.ctypes.data_as(POINTER(c_double)),
        )
        if not state:
            raise RuntimeError("FakfCreate failed")
        return state

    def core(self, state, x: np.ndarray, e: np.ndarray, y: np.ndarray) -> None:
        rc = self.lib.FakfCore(
            state,
            x.ctypes.data_as(POINTER(c_double)),
            e.ctypes.data_as(POINTER(c_double)),
            y.ctypes.data_as(POINTER(c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"FakfCore failed: {rc}")

    def weights(self, state, n_spk, n_ref, order) -> np.ndarray:
        w = np.zeros(n_spk * n_ref * order, dtype=np.float64)
        rc = self.lib.FakfCopyWeights(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError("FakfCopyWeights failed")
        return w.reshape(n_spk, n_ref, order)

    def destroy(self, state) -> None:
        self.lib.FakfDestroy(state)


def check_instance(lib: FakfLib) -> None:
    """两路参考、两只扬声器。次级路径 [0, 1]。第 1 路误差为 0，权值保持 0。"""
    sec = np.zeros((2, 2, 2), dtype=np.float64)
    sec[0, 0] = [0.0, 1.0]
    sec[1, 1] = [0.0, 1.0]
    state = lib.create(2, 2, 2, 1, 2, 1.0, 1.0, 0.0, 1.0, sec)
    x = np.zeros(2, dtype=np.float64)
    e = np.zeros(2, dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)
    try:
        x[:] = (1.0, 0.0)
        e[:] = (1.0, 0.0)
        lib.core(state, x, e, y)
        if np.max(np.abs(y)) > 1e-12:
            raise RuntimeError(f"instance y0 {y}")
        w0 = lib.weights(state, 2, 2, 1)
        x[:] = (0.0, 1.0)
        lib.core(state, x, e, y)
        w1 = lib.weights(state, 2, 2, 1)
    finally:
        lib.destroy(state)
    if np.max(np.abs(w0)) > 1e-12:
        raise RuntimeError(f"instance step 0 {w0}")
    if np.max(np.abs(y)) > 1e-12:
        raise RuntimeError(f"instance y1 {y}")
    expect = np.array([0.5, 0.0])
    if np.max(np.abs(w1[0, :, 0] - expect)) > 1e-12:
        raise RuntimeError(f"instance step 1 speaker 0 {w1[0]}")
    if np.max(np.abs(w1[1])) > 1e-12:
        raise RuntimeError(f"instance step 1 speaker 1 {w1[1]}")


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


def mic_error(play: np.ndarray, d_now: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    n_err, n_spk, sec_order = secondary.shape
    ns = sec_order - 1
    e = np.zeros(n_err, dtype=np.float64)
    for q in range(n_err):
        anti = 0.0
        for spk in range(n_spk):
            anti += float(np.dot(secondary[q, spk, 1:], play[spk, :ns]))
        e[q] = d_now[q] - anti
    return e


def push_play(play: np.ndarray, y: np.ndarray) -> None:
    play[:, 1:] = play[:, :-1]
    play[:, 0] = y


def _givens(pre, nrows, cy):
    x0 = pre[0, 0]
    y0 = pre[0, cy]
    rad = float(np.hypot(x0, y0))
    if not (rad > 0.0):
        return
    c = x0 / rad
    s = y0 / rad
    pre[0, 0] = rad
    pre[0, cy] = 0.0
    for row in range(1, nrows + 1):
        xi = pre[row, 0]
        yi = pre[row, cy]
        pre[row, 0] = c * xi + s * yi
        pre[row, cy] = -s * xi + c * yi


def _hyp(pre, nrows, cy):
    x0 = pre[0, 0]
    y0 = pre[0, cy]
    if y0 == 0.0:
        return
    disc = x0 * x0 - y0 * y0
    if not (disc > 0.0):
        raise RuntimeError("hyperbolic rotation is not real")
    ch = abs(x0) / np.sqrt(disc)
    sh = (y0 / x0) * ch
    pre[0, 0] = np.copysign(np.sqrt(disc), x0)
    pre[0, cy] = 0.0
    for row in range(1, nrows + 1):
        xi = pre[row, 0]
        yi = pre[row, cy]
        pre[row, 0] = ch * xi - sh * yi
        pre[row, cy] = -sh * xi + ch * yi


def _solve_dare(cs, meas, proc):
    ns = cs.size
    p = np.zeros((ns, ns), dtype=np.float64)
    kpsi = np.zeros(ns, dtype=np.float64)
    for _ in range(80):
        v = np.dot(p, cs)
        svar = meas + float(np.dot(cs, v))
        if not (svar > 0.0):
            raise RuntimeError("Riccati variance is not positive")
        av = np.zeros(ns, dtype=np.float64)
        if ns > 1:
            av[1:] = v[:-1]
        nxt = np.zeros((ns, ns), dtype=np.float64)
        if ns > 1:
            nxt[1:, 1:] = p[:-1, :-1]
        nxt -= np.outer(av, av) / svar
        nxt[0, 0] += proc
        diff = float(np.sum(np.abs(nxt - p)))
        p = nxt
        if diff < 1.0e-12:
            break
    v = np.dot(p, cs)
    svar = meas + float(np.dot(cs, v))
    if not (svar > 0.0):
        raise RuntimeError("Riccati variance is not positive")
    rsqrt = np.sqrt(svar)
    if ns > 1:
        kpsi[1:] = v[:-1] / rsqrt
    return kpsi, rsqrt


def _rot_at(pre, prow, cx, cy, hyp):
    x = pre[prow, cx]
    y = pre[prow, cy]
    if hyp:
        if y == 0.0:
            return
        disc = x * x - y * y
        if not (disc > 0.0):
            raise RuntimeError("hyperbolic rotation is not real")
        ch = abs(x) / np.sqrt(disc)
        sh = (y / x) * ch
        xi = pre[:, cx].copy()
        yi = pre[:, cy].copy()
        pre[:, cx] = ch * xi - sh * yi
        pre[:, cy] = -sh * xi + ch * yi
        return
    rad = float(np.hypot(x, y))
    if not (rad > 0.0):
        return
    c = x / rad
    s = y / rad
    xi = pre[:, cx].copy()
    yi = pre[:, cy].copy()
    pre[:, cx] = c * xi + s * yi
    pre[:, cy] = -s * xi + c * yi


def numpy_fakf(x, d, secondary, lam, meas, proc, weight, order):
    n_ref, n = x.shape
    sec = np.asarray(secondary, dtype=np.float64)
    n_err, n_spk, sec_order = sec.shape
    ns = sec_order - 1
    n1 = order + 1
    nlines = n_spk * n_ref
    alpha = 2 * nlines
    n1tot = nlines * n1
    nrows = n1tot + n_spk * ns
    cols = n_err + alpha
    nprows = n_err + nrows
    scale = lam ** -0.5
    nw = n_ref * order
    w = np.zeros((n_spk, nw), dtype=np.float64)
    psi = np.zeros((n_spk, max(ns, 1)), dtype=np.float64)
    gen = np.zeros((nrows, alpha), dtype=np.float64)
    kb = np.zeros((nrows, n_err), dtype=np.float64)
    rsqrt = np.eye(n_err, dtype=np.float64) * np.sqrt(meas)
    sd = np.sqrt(weight)
    tail = sd * lam ** (0.5 * order)
    for t in range(nlines):
        gen[t * n1, 2 * t] = sd
        gen[t * n1 + order, 2 * t + 1] = tail
    if proc > 0.0 and ns > 0:
        for spk in range(n_spk):
            kpsi, r0 = _solve_dare(sec[0, spk, 1:], meas, proc)
            if n_err == 1 and n_spk == 1 and r0 > 0.0:
                rsqrt[0, 0] = r0
            kb[n1tot + spk * ns : n1tot + (spk + 1) * ns, 0] = kpsi
    ref = np.zeros((n_ref, n1), dtype=np.float64)
    play = np.zeros((n_spk, sec_order), dtype=np.float64)
    e = np.zeros((n_err, n), dtype=np.float64)
    for i in range(n):
        for pref in range(n_ref):
            ref[pref, 1:] = ref[pref, :-1]
            ref[pref, 0] = x[pref, i]
        err = mic_error(play, d[:, i], sec)
        e[:, i] = err
        eps = err.copy()
        if ns:
            for spk in range(n_spk):
                for q in range(n_err):
                    eps[q] -= float(np.dot(sec[q, spk, 1:], psi[spk, :ns]))
        pre = np.zeros((nprows, cols), dtype=np.float64)
        pre[:n_err, :n_err] = rsqrt
        for a in range(alpha):
            for q in range(n_err):
                acc = 0.0
                for spk in range(n_spk):
                    dot = 0.0
                    for pref in range(n_ref):
                        t = spk * n_ref + pref
                        hist = ref[pref]
                        dot += float(np.dot(hist, gen[t * n1 : (t + 1) * n1, a]))
                    acc += sec[q, spk, 0] * dot
                    if ns:
                        block = gen[n1tot + spk * ns : n1tot + (spk + 1) * ns, a]
                        acc += float(np.dot(sec[q, spk, 1:], block))
                pre[q, n_err + a] = acc
        for t in range(nlines):
            for tap in range(1, n1):
                pre[n_err + t * n1 + tap, :n_err] = kb[t * n1 + tap - 1]
            for tap in range(n1):
                pre[n_err + t * n1 + tap, n_err:] = scale * gen[t * n1 + tap]
        if ns:
            for spk in range(n_spk):
                base = n1tot + spk * ns
                pre[n_err + base : n_err + base + ns, :n_err] = kb[base : base + ns]
                for a in range(alpha):
                    dot = 0.0
                    for pref in range(n_ref):
                        t = spk * n_ref + pref
                        dot += float(np.dot(ref[pref], gen[t * n1 : (t + 1) * n1, a]))
                    pre[n_err + base, n_err + a] = dot
                    for j in range(1, ns):
                        pre[n_err + base + j, n_err + a] = gen[base + j - 1, a]
        for a in range(alpha):
            hyp = (a % 2) == 1
            for q in range(n_err):
                _rot_at(pre, q, q, n_err + a, hyp)
        rsqrt = pre[:n_err, :n_err].copy()
        kb = pre[n_err:, :n_err].copy()
        gen = pre[n_err:, n_err:].copy()
        rhs = np.linalg.solve(rsqrt, eps)
        for spk in range(n_spk):
            for pref in range(n_ref):
                t = spk * n_ref + pref
                sl = slice(pref * order, (pref + 1) * order)
                add = np.dot(kb[t * n1 : t * n1 + order], rhs)
                w[spk, sl] = scale * w[spk, sl] + add
            if ns:
                nxt = np.zeros(ns, dtype=np.float64)
                base = n1tot + spk * ns
                nxt[0] = float(np.dot(kb[base], rhs))
                for j in range(1, ns):
                    nxt[j] = psi[spk, j - 1] + float(np.dot(kb[base + j], rhs))
                psi[spk, :ns] = nxt
        y = np.zeros(n_spk, dtype=np.float64)
        for spk in range(n_spk):
            acc = 0.0
            for pref in range(n_ref):
                acc += float(np.dot(w[spk, pref * order : (pref + 1) * order], ref[pref, :order]))
            y[spk] = acc
        push_play(play, y)
    return e, w.reshape(n_spk, n_ref, order)


def simulate(lib: FakfLib, rng: np.random.Generator):
    fs = 2000.0
    n = 16000
    x, origin, d = synthesize(n, fs, rng)
    state = lib.create(
        N_REF, N_SPK, N_ERR, ORDER, SEC_ORDER, LAMBDA, MEAS_VAR, PROC_VAR, WEIGHT_VAR, SECONDARY
    )
    e = np.zeros((N_SPK, n), dtype=np.float64)
    play = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    sample_x = np.zeros(N_REF, dtype=np.float64)
    sample_e = np.zeros(N_SPK, dtype=np.float64)
    sample_y = np.zeros(N_SPK, dtype=np.float64)
    try:
        for i in range(n):
            sample_x[:] = x[:, i]
            sample_e[:] = mic_error(play, d[:, i], SECONDARY)
            lib.core(state, sample_x, sample_e, sample_y)
            push_play(play, sample_y)
            e[:, i] = sample_e
        w_c = lib.weights(state, N_SPK, N_REF, ORDER)
    finally:
        lib.destroy(state)

    e_np, w_np = numpy_fakf(x, d, SECONDARY, LAMBDA, MEAS_VAR, PROC_VAR, WEIGHT_VAR, ORDER)
    gap = float(np.max(np.abs(w_c - w_np)))
    if gap > 1e-8:
        raise RuntimeError(f"C and NumPy weights differ by {gap}")
    if np.max(np.abs(e - e_np)) > 1e-8:
        raise RuntimeError("C and NumPy residuals differ")

    tail = slice(-4000, None)
    origin_nmse = []
    for s in range(N_SPK):
        num = float(np.sum((e[s, tail] - origin[s, tail]) ** 2))
        den = float(np.sum(origin[s, tail] ** 2))
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
    for s, color in enumerate((C_AFTER, "#e09f3e")):
        power = np.convolve(e[s] ** 2, kernel, mode="valid")
        axes[3].plot(
            t_learn, 10.0 * np.log10(np.maximum(power, 1e-20)), color=color, lw=1.2, label=f"通道 {s}"
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
    for i, ((spk, pref), color) in enumerate(zip(pairs, pair_color)):
        loc = np.arange(ORDER) + i * group
        axes[6].plot(
            loc,
            BEFORE[spk, pref],
            color=color,
            ls="none",
            marker="o",
            ms=5.5,
            label=f"扬声器 {spk}，参考 {pref}",
        )
        axes[6].plot(loc, w_c[spk, pref], color=color, lw=1.15, marker="x", ms=6)
        err = w_c[spk, pref] - BEFORE[spk, pref]
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
    path = save(fig, "12_fakf.png")
    return path, origin_nmse, coeff_nmse, gap


def main() -> None:
    lib = FakfLib()
    check_instance(lib)
    rng = np.random.default_rng(7)
    path, origin_nmse, coeff_nmse, gap = simulate(lib, rng)
    for s, nmse in enumerate(origin_nmse):
        print(f"channel {s} residual vs origin NMSE {nmse:.3e}")
    print(f"coeff NMSE {coeff_nmse:.3e}")
    print(f"C vs NumPy {gap:.3e}")
    print(path)


if __name__ == "__main__":
    main()
