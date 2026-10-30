#!/usr/bin/env python3
"""同一条路径上的性能和耗时。参数与 doc/fakf.md 第 4.2 节一致。

维纳滤波整段一次求解，只进性能，不进耗时。
耗时只包 Core，去掉开头 64 点。频域一次 8 点，耗时除以 8。
"""

from __future__ import annotations

import subprocess
import sys
import time
from ctypes import CDLL, POINTER, c_double, c_int, c_size_t, c_void_p
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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

FS = 2000.0
N = 16000
BLOCK = 8
ORDER = 8
N_REF = 2
N_SPK = 2
N_ERR = 2
TAIL = 4000

SEC_TAIL = np.zeros((N_ERR, N_SPK, 2), dtype=np.float64)
SEC_TAIL[0, 0] = [0.90, 0.20]
SEC_TAIL[0, 1] = [0.15, 0.05]
SEC_TAIL[1, 0] = [0.12, -0.04]
SEC_TAIL[1, 1] = [0.85, 0.15]
SEC_ORDER = BLOCK + SEC_TAIL.shape[-1]
SECONDARY = np.zeros((N_ERR, N_SPK, SEC_ORDER), dtype=np.float64)
SECONDARY[:, :, BLOCK:] = SEC_TAIL
BEFORE = np.zeros((N_SPK, N_REF, ORDER), dtype=np.float64)
BEFORE[0, 0] = [0.90, -0.55, 0.35, -0.20, 0.12, -0.06, 0.03, -0.01]
BEFORE[0, 1] = [0.40, 0.22, -0.12, 0.06, -0.03, 0.01, 0.0, 0.0]
BEFORE[1, 0] = [0.25, -0.18, 0.10, -0.05, 0.02, 0.0, 0.0, 0.0]
BEFORE[1, 1] = [0.80, -0.30, 0.16, -0.08, 0.04, -0.02, 0.01, 0.0]
PRIMARY = np.zeros((N_ERR, N_REF, BEFORE.shape[-1] + SEC_ORDER - 1), dtype=np.float64)
for _q in range(N_ERR):
    for _p in range(N_REF):
        for _l in range(N_SPK):
            PRIMARY[_q, _p] += np.convolve(SECONDARY[_q, _l], BEFORE[_l, _p])


def load_lib(folder: str, soname: str, create: str, core: str, copy: str, destroy: str, argtypes):
    src = ROOT / folder
    build = ROOT / "output" / folder
    subprocess.check_call(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"])
    subprocess.check_call(["cmake", "--build", str(build)])
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib = CDLL(str(build / f"lib{soname}{ext}"))
    getattr(lib, create).argtypes = argtypes
    getattr(lib, create).restype = c_void_p
    getattr(lib, core).argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    getattr(lib, core).restype = c_int
    getattr(lib, copy).argtypes = [c_void_p, POINTER(c_double)]
    getattr(lib, copy).restype = c_int
    getattr(lib, destroy).argtypes = [c_void_p]
    return lib


SIZES = [c_size_t] * 5
SEC_PTR = POINTER(c_double)
FX = load_lib("fxlms", "fxlms", "FxlmsCreate", "FxlmsCore", "FxlmsCopyWeights", "FxlmsDestroy", SIZES + [c_double, SEC_PTR])
FD = load_lib("fdfxlms", "fdfxlms", "FdfxlmsCreate", "FdfxlmsCore", "FdfxlmsCopyWeights", "FdfxlmsDestroy", SIZES + [c_double, SEC_PTR])
PF = load_lib("pffxlms", "pffxlms", "PffxlmsCreate", "PffxlmsCore", "PffxlmsCopyWeights", "PffxlmsDestroy", SIZES + [c_size_t, c_double, SEC_PTR])
FR = load_lib("frls", "frls", "FrlsCreate", "FrlsCore", "FrlsCopyWeights", "FrlsDestroy", SIZES + [c_double, c_double, c_double, SEC_PTR])
FA = load_lib("farls", "farls", "FarlsCreate", "FarlsCore", "FarlsCopyWeights", "FarlsDestroy", SIZES + [c_double, c_double, c_double, SEC_PTR])
KF = load_lib("kfanc", "kfanc", "KfancCreate", "KfancCore", "KfancCopyWeights", "KfancDestroy", SIZES + [c_double, c_double, c_double, SEC_PTR])
FK = load_lib("fakf", "fakf", "FakfCreate", "FakfCore", "FakfCopyWeights", "FakfDestroy", SIZES + [c_double, c_double, c_double, c_double, SEC_PTR])


def synthesize(seed: int = 7):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = np.zeros((N_REF, N), dtype=np.float64)
    tones = (40.0, 80.0, 120.0, 160.0)
    for amp, freq in zip((1.0, 0.7, 0.45, 0.25), tones):
        x[0] += amp * np.sin(2.0 * np.pi * freq * t)
    for amp, freq in zip((0.75, 0.55, 0.30, 0.20), tones):
        x[1] += amp * np.sin(2.0 * np.pi * freq * t + 0.6)
    x[0] += 0.35 * rng.normal(size=N)
    x[1] += 0.35 * rng.normal(size=N)
    origin = 0.25 * rng.normal(size=(N_ERR, N))
    d = np.zeros((N_ERR, N), dtype=np.float64)
    for q in range(N_ERR):
        for p in range(N_REF):
            d[q] += np.convolve(x[p], PRIMARY[q, p], mode="full")[:N]
    d += origin
    return x, origin, d


X, ORIGIN, D = synthesize()
SEC = np.ascontiguousarray(SECONDARY, dtype=np.float64)


def mic_error(play: np.ndarray, d_now: np.ndarray) -> np.ndarray:
    ns = SEC_ORDER - 1
    err = np.empty(N_ERR, dtype=np.float64)
    for q in range(N_ERR):
        anti = 0.0
        for l in range(N_SPK):
            anti += float(np.dot(SEC[q, l, 1:], play[l, :ns]))
        err[q] = d_now[q] - anti
    return err


def block_error(y_played: np.ndarray, d_blk: np.ndarray, t0: int) -> np.ndarray:
    e_blk = np.zeros((N_ERR, BLOCK), dtype=np.float64)
    for i in range(BLOCK):
        tt = t0 + i
        for q in range(N_ERR):
            acc = 0.0
            for l in range(N_SPK):
                for j in range(SEC_ORDER):
                    src = tt - j
                    if src >= 0:
                        acc += SEC[q, l, j] * y_played[l, src]
            e_blk[q, i] = d_blk[q, i] - acc
    return e_blk


def metrics(e: np.ndarray, w: np.ndarray) -> dict:
    err = w - BEFORE
    power = np.mean(e**2, axis=0)
    sm = np.convolve(power, np.ones(200) / 200.0, mode="valid")
    floor = float(np.median(sm[-TAIL:]))
    thr = floor * 10**0.1
    hold = 0
    hit = None
    for i, v in enumerate(sm):
        if v <= thr:
            hold += 1
            if hold >= 200:
                hit = i - 199
                break
        else:
            hold = 0
    t_ms = None if hit is None else (hit + 200) / FS * 1000.0
    tail_e = e[:, -TAIL:]
    tail_o = ORIGIN[:, -TAIL:]
    en = np.sum(tail_e**2, axis=0) - np.sum(tail_o**2, axis=0)
    sm_db = 10.0 * np.log10(np.maximum(sm, 1e-20))
    curve = []
    for t in range(0, N, 80):
        k = min(max(t - 199, 0), sm_db.size - 1)
        curve.append(float(sm_db[k]))
    return {
        "coeff_mean": float(np.mean(err)),
        "coeff_var": float(np.var(err)),
        "en_mean": float(np.mean(en)),
        "en_var": float(np.var(en)),
        "t_ms": t_ms,
        "curve": curve,
    }


def run_sample(lib, create_name, args) -> tuple[np.ndarray, np.ndarray]:
    sec = SEC.ctypes.data_as(POINTER(c_double))
    state = getattr(lib, create_name)(*args, sec)
    if not state:
        raise RuntimeError(create_name)
    e = np.zeros((N_ERR, N), dtype=np.float64)
    play = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    sx = np.zeros(N_REF, dtype=np.float64)
    se = np.zeros(N_ERR, dtype=np.float64)
    sy = np.zeros(N_SPK, dtype=np.float64)
    core = getattr(lib, create_name.replace("Create", "Core"))
    copy = getattr(lib, create_name.replace("Create", "CopyWeights"))
    destroy = getattr(lib, create_name.replace("Create", "Destroy"))
    try:
        for i in range(N):
            sx[:] = X[:, i]
            se[:] = mic_error(play, D[:, i])
            rc = core(
                state,
                sx.ctypes.data_as(POINTER(c_double)),
                se.ctypes.data_as(POINTER(c_double)),
                sy.ctypes.data_as(POINTER(c_double)),
            )
            if rc != 0:
                raise RuntimeError(f"{create_name} core {rc}")
            play[:, 1:] = play[:, :-1]
            play[:, 0] = sy
            e[:, i] = se
        w = np.zeros(N_SPK * N_REF * ORDER, dtype=np.float64)
        rc = copy(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError("copy")
    finally:
        destroy(state)
    return e, w.reshape(N_SPK, N_REF, ORDER)


def run_block(lib, create_name, args) -> tuple[np.ndarray, np.ndarray]:
    sec = SEC.ctypes.data_as(POINTER(c_double))
    state = getattr(lib, create_name)(*args, sec)
    if not state:
        raise RuntimeError(create_name)
    e = np.zeros((N_ERR, N), dtype=np.float64)
    y_played = np.zeros((N_SPK, N), dtype=np.float64)
    xb = np.zeros(N_REF * BLOCK, dtype=np.float64)
    eb = np.zeros(N_ERR * BLOCK, dtype=np.float64)
    yb = np.zeros(N_SPK * BLOCK, dtype=np.float64)
    core = getattr(lib, create_name.replace("Create", "Core"))
    copy = getattr(lib, create_name.replace("Create", "CopyWeights"))
    destroy = getattr(lib, create_name.replace("Create", "Destroy"))
    try:
        for b in range(N // BLOCK):
            sl = slice(b * BLOCK, (b + 1) * BLOCK)
            measured = block_error(y_played, D[:, sl], b * BLOCK)
            for p in range(N_REF):
                xb[p * BLOCK : (p + 1) * BLOCK] = X[p, sl]
            for q in range(N_ERR):
                eb[q * BLOCK : (q + 1) * BLOCK] = measured[q]
            rc = core(
                state,
                xb.ctypes.data_as(POINTER(c_double)),
                eb.ctypes.data_as(POINTER(c_double)),
                yb.ctypes.data_as(POINTER(c_double)),
            )
            if rc != 0:
                raise RuntimeError(f"{create_name} core {rc}")
            for l in range(N_SPK):
                y_played[l, sl] = yb[l * BLOCK : (l + 1) * BLOCK]
            e[:, sl] = measured
        w = np.zeros(N_SPK * N_REF * ORDER, dtype=np.float64)
        rc = copy(state, w.ctypes.data_as(POINTER(c_double)))
        if rc != 0:
            raise RuntimeError("copy")
    finally:
        destroy(state)
    return e, w.reshape(N_SPK, N_REF, ORDER)


def wiener() -> tuple[np.ndarray, np.ndarray]:
    """滤波后参考上的整段 Wiener–Hopf。控制长度与 b 相同。"""
    nstate = N_SPK * N_REF * ORDER
    gram = np.zeros((nstate, nstate), dtype=np.float64)
    pvec = np.zeros(nstate, dtype=np.float64)
    fx = np.zeros((N_ERR, N_SPK, N_REF, N), dtype=np.float64)
    for q in range(N_ERR):
        for l in range(N_SPK):
            for p in range(N_REF):
                fx[q, l, p] = np.convolve(X[p], SEC[q, l], mode="full")[:N]
    for q in range(N_ERR):
        cols = []
        for l in range(N_SPK):
            for p in range(N_REF):
                sig = fx[q, l, p]
                for k in range(ORDER):
                    col = np.zeros(N, dtype=np.float64)
                    if k:
                        col[k:] = sig[:-k]
                    else:
                        col[:] = sig
                    cols.append(col)
        phi = np.column_stack(cols)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram += phi.T @ phi
            pvec += phi.T @ D[q]
    w = np.linalg.solve(gram, pvec).reshape(N_SPK, N_REF, ORDER)
    e = np.zeros((N_ERR, N), dtype=np.float64)
    play = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    xref = np.zeros((N_REF, ORDER), dtype=np.float64)
    for i in range(N):
        e[:, i] = mic_error(play, D[:, i])
        for p in range(N_REF):
            xref[p, 1:] = xref[p, :-1]
            xref[p, 0] = X[p, i]
        y = np.zeros(N_SPK, dtype=np.float64)
        for l in range(N_SPK):
            acc = 0.0
            for p in range(N_REF):
                acc += float(np.dot(w[l, p], xref[p]))
            y[l] = acc
        play[:, 1:] = play[:, :-1]
        play[:, 0] = y
    return e, w


def time_us(samples_ns: np.ndarray) -> tuple[float, float]:
    us = samples_ns / 1000.0
    kept = us[us <= 200.0]
    if kept.size == 0:
        kept = us
    return float(np.mean(kept)), float(np.var(kept))


def time_sample(lib, create_name, args) -> tuple[float, float]:
    sec = SEC.ctypes.data_as(POINTER(c_double))
    state = getattr(lib, create_name)(*args, sec)
    play = np.zeros((N_SPK, SEC_ORDER), dtype=np.float64)
    sx = np.zeros(N_REF, dtype=np.float64)
    se = np.zeros(N_ERR, dtype=np.float64)
    sy = np.zeros(N_SPK, dtype=np.float64)
    core = getattr(lib, create_name.replace("Create", "Core"))
    destroy = getattr(lib, create_name.replace("Create", "Destroy"))
    ticks = np.zeros(N, dtype=np.float64)
    try:
        for i in range(N):
            sx[:] = X[:, i]
            se[:] = mic_error(play, D[:, i])
            t0 = time.perf_counter_ns()
            rc = core(
                state,
                sx.ctypes.data_as(POINTER(c_double)),
                se.ctypes.data_as(POINTER(c_double)),
                sy.ctypes.data_as(POINTER(c_double)),
            )
            t1 = time.perf_counter_ns()
            if rc != 0:
                raise RuntimeError(f"core {rc}")
            play[:, 1:] = play[:, :-1]
            play[:, 0] = sy
            ticks[i] = t1 - t0
    finally:
        destroy(state)
    return time_us(ticks[64:])


def time_block(lib, create_name, args) -> tuple[float, float]:
    sec = SEC.ctypes.data_as(POINTER(c_double))
    state = getattr(lib, create_name)(*args, sec)
    y_played = np.zeros((N_SPK, N), dtype=np.float64)
    xb = np.zeros(N_REF * BLOCK, dtype=np.float64)
    eb = np.zeros(N_ERR * BLOCK, dtype=np.float64)
    yb = np.zeros(N_SPK * BLOCK, dtype=np.float64)
    core = getattr(lib, create_name.replace("Create", "Core"))
    destroy = getattr(lib, create_name.replace("Create", "Destroy"))
    nblk = N // BLOCK
    ticks = np.zeros(nblk, dtype=np.float64)
    try:
        for b in range(nblk):
            sl = slice(b * BLOCK, (b + 1) * BLOCK)
            measured = block_error(y_played, D[:, sl], b * BLOCK)
            for p in range(N_REF):
                xb[p * BLOCK : (p + 1) * BLOCK] = X[p, sl]
            for q in range(N_ERR):
                eb[q * BLOCK : (q + 1) * BLOCK] = measured[q]
            t0 = time.perf_counter_ns()
            rc = core(
                state,
                xb.ctypes.data_as(POINTER(c_double)),
                eb.ctypes.data_as(POINTER(c_double)),
                yb.ctypes.data_as(POINTER(c_double)),
            )
            t1 = time.perf_counter_ns()
            if rc != 0:
                raise RuntimeError(f"core {rc}")
            for l in range(N_SPK):
                y_played[l, sl] = yb[l * BLOCK : (l + 1) * BLOCK]
            ticks[b] = (t1 - t0) / BLOCK
    finally:
        destroy(state)
    return time_us(ticks[8:])


def save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    base = [c_size_t(N_REF), c_size_t(N_SPK), c_size_t(N_ERR), c_size_t(ORDER), c_size_t(SEC_ORDER)]
    fd_base = [c_size_t(N_REF), c_size_t(N_SPK), c_size_t(N_ERR), c_size_t(BLOCK), c_size_t(SEC_ORDER)]
    jobs = [
        ("维纳滤波", lambda: wiener(), None),
        ("快速阵列卡尔曼", lambda: run_sample(FK, "FakfCreate", base + [c_double(1.0), c_double(1.0), c_double(0.0), c_double(10.0)]), (FK, "FakfCreate", base + [c_double(1.0), c_double(1.0), c_double(0.0), c_double(10.0)], False)),
        ("卡尔曼滤波", lambda: run_sample(KF, "KfancCreate", base + [c_double(1.0), c_double(1.0), c_double(10.0)]), (KF, "KfancCreate", base + [c_double(1.0), c_double(1.0), c_double(10.0)], False)),
        ("滤波 RLS", lambda: run_sample(FR, "FrlsCreate", base + [c_double(1.0), c_double(0.25), c_double(0.1)]), (FR, "FrlsCreate", base + [c_double(1.0), c_double(0.25), c_double(0.1)], False)),
        ("快速阵列 RLS", lambda: run_sample(FA, "FarlsCreate", base + [c_double(1.0), c_double(0.25), c_double(0.1)]), (FA, "FarlsCreate", base + [c_double(1.0), c_double(0.25), c_double(0.1)], False)),
        ("频域 FxLMS", lambda: run_block(FD, "FdfxlmsCreate", fd_base + [c_double(0.006)]), (FD, "FdfxlmsCreate", fd_base + [c_double(0.006)], True)),
        ("分块频域 FxLMS", lambda: run_block(PF, "PffxlmsCreate", fd_base + [c_size_t(ORDER), c_double(0.006)]), (PF, "PffxlmsCreate", fd_base + [c_size_t(ORDER), c_double(0.006)], True)),
        ("FxLMS", lambda: run_sample(FX, "FxlmsCreate", base + [c_double(0.004)]), (FX, "FxlmsCreate", base + [c_double(0.004)], False)),
    ]

    rows = []
    for name, fn, _timed in jobs:
        e, w = fn()
        row = metrics(e, w)
        row["name"] = name
        rows.append(row)
        speed = "一次求解" if row["t_ms"] is not None and name == "维纳滤波" else f"{row['t_ms'] / 1000.0:.2f} s"
        print(
            f"{name}  {speed}  "
            f"系数 {row['coeff_mean']:.3e} / {row['coeff_var']:.3e}  "
            f"能量 {row['en_mean']:.3e} / {row['en_var']:.3e}"
        )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ncurve = len(rows[0]["curve"])
    taxis = np.arange(ncurve) * 80 / FS
    for row in rows:
        if row["name"] == "维纳滤波":
            ax.plot(taxis, row["curve"], lw=1.6, ls="--", color="#222222", label=row["name"], zorder=5)
        else:
            ax.plot(taxis, row["curve"], lw=1.15, label=row["name"])
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("误差功率 / dB")
    ax.set_title("留出一块延时后，各算法误差功率的下降")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    perf_path = save(fig, "13_cmp_perf.png")

    timed = []
    for name, _fn, spec in jobs:
        if spec is None:
            continue
        lib, create_name, args, blocked = spec
        mean_us, var_us = time_block(lib, create_name, args) if blocked else time_sample(lib, create_name, args)
        timed.append((name, mean_us, var_us))
        print(f"{name}  {mean_us:.2f} μs  方差 {var_us:.2f}")

    timed.sort(key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in timed][::-1]
    means = np.array([item[1] for item in timed][::-1])
    stds = np.array([np.sqrt(item[2]) for item in timed][::-1])
    variances = [item[2] for item in timed][::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, means, color="#3d5a80", height=0.62, zorder=2)
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#c0392b", elinewidth=1.2, capsize=3, zorder=3)
    for i, (m, v) in enumerate(zip(means, variances)):
        ax.text(m + stds[i] + 1.2, i, f"{m:.2f} μs    方差 {v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("每点耗时 / μs")
    ax.set_xlim(0, max(means + stds) * 1.45)
    ax.set_title("这次仿真的 Core 每点耗时（均值 ± 标准差）")
    ax.grid(axis="x", alpha=0.35)
    time_path = save(fig, "13_cmp_time.png")
    print(perf_path)
    print(time_path)


if __name__ == "__main__":
    main()
