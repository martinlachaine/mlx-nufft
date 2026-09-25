"""Per-stage timing profiler for the plan classes behind the public API.

Times every execute() stage of Type1PlanND / Type2PlanND (the classes Plan and
nufft*d1/2 run) and GpuT3Plan (Plan type 3, nufft*d3) over a fixed matrix,
median of N reps with mx.synchronize() around each timed region. Each timed
set is preceded by a short GPU busy burst (steady clocks: after a mostly
idle plan build the first executes of a ~1 ms transform ran 2x slower than
the rest, and the warm-ups alone did not cover that ramp). Nothing here
changes library behavior: the stage runners below re-issue the exact kernel /
FFT sequence of each plan's execute() through the plan's own internal methods
and evaluate + synchronize after every stage. The whole execute() is timed
separately (same warm-up and rep count) and the sum of the stage medians is
reported against it (sum/whole), so the pipelining lost to the per-stage
syncs stays visible. The staged output is also checked against execute()'s.

Matrix (n_trans=1, default options, fixed seed, complex64 inputs):
  types 1 and 2: 1D N=2^20, 2D 512^2, 3D 128^3 and 256^3, M=1e6 points
                 uniform on [-pi, pi)^d
  type 3:        3D generic N=512 (P=1e5 sources, M=N^2 targets, the
                 results/acceptance.json geometry) and 1D P=M=1e6 with
                 X=S=300 (runs as a degenerate slice of the 3D kernel)
  eps in {1e-3, 1e-5}

Stages, in execute order (h2d: mx.array upload of the input, d2h: np.array
download of the result; both are part of execute()):
  type 1: h2d, spread, fft_<ax>, crop_<ax>, ..., crop_x+deconv, d2h
  type 2: h2d, pad_x+deconv, fft_x, pad_<ax>, fft_<ax>, ..., gather, d2h
  type 3: h2d, spread (sort + prephase fused), pad+deconv, fft_z, fft_y,
          fft_x, gather, d2h   (non-slab plans only; slab plans are timed
          whole-execute only)

Usage: profile_stages.py [--reps 7] [--warm 2] [--wake-ms 100]
                         [--cases id[,id..]] [--list] [--tag profile_baseline]
Writes results/<tag>.json and results/<tag>.md. results/ is gitignored: copy
the .md to harness/PROFILE_BASELINE_<version>.md to keep a baseline with the
branch.
"""

import sys
import time
import json
import pathlib
import platform
import statistics
import subprocess
from datetime import datetime, timezone

import numpy as np
import mlx.core as mx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlx_nufft                                             # noqa: E402
from mlx_nufft.nd import Type1PlanND, Type2PlanND            # noqa: E402
from mlx_nufft.gpu_t3 import GpuT3Plan, fft_axis             # noqa: E402
from harness.bench_gpu import machine                        # noqa: E402
from harness.gen import gen_generic, rel_l2                  # noqa: E402

AX = ("x", "y", "z")
SEED = 0
EPS_LIST = (1e-3, 1e-5)
FLAG_TOL = 0.15          # |sum/whole - 1| above this is flagged in the table
WAKE_MS = 100.0          # GPU busy burst before each timed set (0 disables)

# one entry per geometry; each runs at every eps in EPS_LIST
CASES = [
    dict(id="t1_1d", type=1, N=(2 ** 20,), M=1_000_000),
    dict(id="t1_2d", type=1, N=(512, 512), M=1_000_000),
    dict(id="t1_3d_128", type=1, N=(128,) * 3, M=1_000_000),
    dict(id="t1_3d_256", type=1, N=(256,) * 3, M=1_000_000),
    dict(id="t2_1d", type=2, N=(2 ** 20,), M=1_000_000),
    dict(id="t2_2d", type=2, N=(512, 512), M=1_000_000),
    dict(id="t2_3d_128", type=2, N=(128,) * 3, M=1_000_000),
    dict(id="t2_3d_256", type=2, N=(256,) * 3, M=1_000_000),
    dict(id="t3_3d_generic512", type=3, gen="generic", N=512, P=100_000),
    dict(id="t3_1d", type=3, gen="1d", P=1_000_000, M=1_000_000,
         X=300.0, S=300.0),
]


# ---------------------------------------------------------------------------
# stage runners: generators mirroring each plan's execute(), yielding
# (label, array) after building each stage; the profiler evaluates and
# synchronizes at every yield. The last yield carries the numpy result.

def t1_stages(plan, c):
    """Type1PlanND.execute + _fft_modes, one yield per kernel / FFT."""
    nu_tot = int(np.prod(plan.n_up))
    cmx = mx.array(np.asarray(c, dtype=np.complex64))
    yield "h2d", cmx
    if plan._od_ex:
        bf = plan._launch_spread_ex(cmx)
    elif plan._od:
        cpf = mx.view(cmx, dtype=mx.float32)
        bf = plan._spread_od(
            inputs=[cpf, plan.mx_perm] + plan.mx_i1 + plan.mx_fr
                   + [plan._mx_sub_start, plan._mx_sub_count]
                   + plan._mx_sub_o,
            output_shapes=[(nu_tot * 2,)],
            output_dtypes=[mx.float32],
            grid=(plan._OD_TG, plan._od_nsub, 1),
            threadgroup=(plan._OD_TG, 1, 1),
            init_value=0)[0]
    else:
        cpf = mx.view(mx.take(cmx, plan.mx_perm), dtype=mx.float32)
        lanes = plan._lanes
        bf = plan._spread(
            inputs=[cpf] + plan.mx_i1 + plan.mx_fr,
            output_shapes=[(nu_tot * 2,)],
            output_dtypes=[mx.float32],
            grid=(lanes, max(plan.P, 1), 1),
            threadgroup=(lanes, max(1, 1024 // lanes), 1),
            init_value=0)[0]
    yield "spread", bf
    # _fft_modes with no batch axis (nb = 0, B = 1)
    dim, inv = plan.dim, plan.isign > 0
    cyc = (dim - 1,) + tuple(range(dim - 1))
    H = mx.view(bf, dtype=mx.complex64).reshape(*plan.n_up)
    del bf
    H = fft_axis(H, H.ndim - 1, inverse=inv, twiddle_cache=plan._twiddles)
    yield f"fft_{AX[dim - 1]}", H
    for ax in range(dim - 1, 0, -1):
        H = mx.take(mx.transpose(H, cyc), plan._mx_cropidx[ax], axis=0)
        yield f"crop_{AX[ax]}", H
        H = fft_axis(H, H.ndim - 1, inverse=inv, twiddle_cache=plan._twiddles)
        yield f"fft_{AX[ax - 1]}", H
    vf = mx.view(H, dtype=mx.float32).reshape(-1)
    N_tot = int(np.prod(plan.N))
    g, tg = plan._crop_launch(1)
    fk = plan._crop(
        inputs=[vf] + plan.mx_dec,
        output_shapes=[(N_tot * 2,)],
        output_dtypes=[mx.float32],
        grid=g, threadgroup=tg)[0]
    res = mx.view(fk, dtype=mx.complex64).reshape(*plan.N)
    yield "crop_x+deconv", res
    yield "d2h", np.array(res)


def t2_stages(plan, fk):
    """Type2PlanND.execute + _modes_to_grid, one yield per kernel / FFT."""
    dim, N, nu = plan.dim, plan.N, plan.n_up
    inv = plan.isign > 0
    fmx = mx.array(np.asarray(fk, dtype=np.complex64))
    yield "h2d", fmx
    fkf = mx.view(fmx.reshape(-1), dtype=mx.float32)
    d0 = tuple(N[d] for d in range(1, dim)) + (nu[0],)
    if dim == 3:
        g = (nu[0], N[2], N[1])
    elif dim == 2:
        g = (nu[0], N[1], 1)
    else:
        g = (nu[0], 1, 1)
    Hf = plan._pad(
        inputs=[fkf] + plan.mx_dec,
        output_shapes=[(int(np.prod(d0)) * 2,)],
        output_dtypes=[mx.float32],
        grid=g, threadgroup=plan._tg_for(g[0]))[0]
    H = mx.view(Hf, dtype=mx.complex64).reshape(*d0)
    yield "pad_x+deconv", H
    H = fft_axis(H, dim - 1, inverse=inv, twiddle_cache=plan._twiddles)
    yield "fft_x", H
    cyc = tuple(range(1, dim)) + (0,)
    for d in range(1, dim):
        T = mx.transpose(H, cyc)
        parts = [T[..., N[d] // 2:]]
        if nu[d] > N[d]:
            zshape = tuple(T.shape[:-1]) + (nu[d] - N[d],)
            parts.append(mx.zeros(zshape, dtype=mx.complex64))
        if N[d] // 2 > 0:
            parts.append(T[..., :N[d] // 2])
        H = mx.concatenate(parts, axis=dim - 1)
        yield f"pad_{AX[d]}", H
        H = fft_axis(H, dim - 1, inverse=inv, twiddle_cache=plan._twiddles)
        yield f"fft_{AX[d]}", H
    vf = mx.view(H, dtype=mx.float32).reshape(-1)
    if plan._od:
        out = plan._gather_od(
            inputs=[vf, plan.mx_perm] + plan.mx_i1 + plan.mx_fr
                   + [plan._mx_sub_start, plan._mx_sub_count]
                   + plan._mx_sub_o,
            output_shapes=[(plan.P * 2,)],
            output_dtypes=[mx.float32],
            grid=(plan._OD_TG, plan._od_nsub, 1),
            threadgroup=(plan._OD_TG, 1, 1))[0]
    else:
        ins = ([vf] + ([plan.mx_perm] if plan._gather_sorted else [])
               + plan.mx_i1 + plan.mx_fr)
        out = plan._gather(
            inputs=ins,
            output_shapes=[(plan.P * 2,)],
            output_dtypes=[mx.float32],
            grid=(max(plan.P, 1), 1, 1), threadgroup=(256, 1, 1))[0]
    del H, vf
    res = mx.view(out, dtype=mx.complex64)
    yield "gather", res
    if plan.sorted and not plan._od and not plan._gather_sorted:
        if not hasattr(plan, "_mx_inv"):
            inv_perm = np.empty_like(plan.perm)
            inv_perm[plan.perm] = np.arange(plan.P)
            plan._mx_inv = mx.array(inv_perm.astype(np.uint32))
        res = mx.take(res, plan._mx_inv)
        yield "unsort", res
    yield "d2h", np.array(res)


def t3_stages(plan, c):
    """GpuT3Plan.execute (non-slab), one yield per stage and FFT axis."""
    assert not plan.slab_mode
    nu1, nu2, nu3 = plan.n_up

    def _trim():
        if plan.low_mem:
            mx.clear_cache()

    cmx = mx.array(np.asarray(c, dtype=np.complex64))
    yield "h2d", cmx
    cpf = mx.view(cmx, dtype=mx.float32)
    bf = plan._spread_stage(cpf)
    del cpf
    yield "spread", bf
    _trim()
    H = plan._pad(
        inputs=[bf, plan.mx_dec[0], plan.mx_dec[1], plan.mx_dec[2]],
        output_shapes=[(nu1 * nu2 * nu3 * 2,)],
        output_dtypes=[mx.float32],
        grid=(nu3, nu2, nu1), threadgroup=(nu3 if nu3 <= 32 else 32,
                                           1024 // min(nu3, 32), 1))[0]
    del bf
    yield "pad+deconv", H
    _trim()
    H = mx.view(H, dtype=mx.complex64).reshape(nu1, nu2, nu3)
    if plan._vkfft_fft3:
        H = plan._fft3_vkfft(H)
        yield "fft_xyz", H
    else:
        for ax in (2, 1, 0):
            Hn = fft_axis(H, ax, inverse=plan.isign > 0,
                          twiddle_cache=plan._twiddles)
            del H
            H = Hn
            del Hn
            yield f"fft_{AX[ax]}", H
            _trim()
    vf = mx.view(H, dtype=mx.float32).reshape(-1)
    out = plan._gather_stage(vf)
    del H, vf
    res = mx.view(out, dtype=mx.complex64)
    yield "gather", res
    yield "d2h", np.array(res)


STAGE_FNS = {1: t1_stages, 2: t2_stages, 3: t3_stages}


# ---------------------------------------------------------------------------
# problem construction

def build(spec, eps):
    """Plan + input for one matrix entry; returns (plan, inp, info)."""
    rng = np.random.default_rng(SEED)
    if spec["type"] in (1, 2):
        N, M = tuple(spec["N"]), spec["M"]
        x = tuple(rng.uniform(-np.pi, np.pi, M) for _ in range(len(N)))
        if spec["type"] == 1:
            inp = (rng.standard_normal(M) + 1j * rng.standard_normal(M)
                   ).astype(np.complex64)
            cls = Type1PlanND
        else:
            inp = (rng.standard_normal(N) + 1j * rng.standard_normal(N)
                   ).astype(np.complex64)
            cls = Type2PlanND
        t0 = time.perf_counter()
        plan = cls(x, N, eps=eps)          # class defaults: the api.py path
        mx.synchronize()
        t_plan = time.perf_counter() - t0
        if spec["type"] == 1:
            path = ("od_ex" if plan._od_ex else "od" if plan._od else "gm")
        else:
            path = ("od" if plan._od else
                    "sorted_gather" if plan._gather_sorted else "gm")
        info = dict(dim=plan.dim, N=list(N), M=M, P=None, w=plan.w,
                    sigma=plan.sigma, n_up=[int(v) for v in plan.n_up],
                    isign=plan.isign, path=path, t_plan_s=t_plan,
                    pts_per_cell=M / float(np.prod(plan.n_up)))
        return plan, inp, info
    if spec["gen"] == "generic":
        prob = gen_generic(N=spec["N"], P=spec["P"])
        x, s = prob["x"], prob["s"]
        c = np.asarray(prob["c"], dtype=np.complex64)
        dim = 3
    else:
        P, M = spec["P"], spec["M"]
        x1 = rng.uniform(-spec["X"], spec["X"], P)
        s1 = rng.uniform(-spec["S"], spec["S"], M)
        x = (x1, np.zeros(P), np.zeros(P))
        s = (s1, np.zeros(M), np.zeros(M))
        c = (rng.standard_normal(P) + 1j * rng.standard_normal(P)
             ).astype(np.complex64)
        dim = 1
    t0 = time.perf_counter()
    plan = GpuT3Plan(x, s, eps=eps, isign=+1, prec="crit64")
    mx.synchronize()
    t_plan = time.perf_counter() - t0
    path = (("od" if plan._ods is not None else "gm") + "_spread/"
            + ("sorted" if plan._tgt_sorted else "natural") + "_gather"
            + ("/vkfft" if plan._vkfft_fft3 else ""))
    info = dict(dim=dim, N=None, M=int(plan.M), P=int(plan.P), w=plan.w,
                sigma=plan.sigma, nf=[int(v) for v in plan.nf],
                n_up=[int(v) for v in plan.n_up], isign=plan.isign,
                path=path, slab=bool(plan.slab_mode),
                low_mem=bool(plan.low_mem), t_plan_s=t_plan,
                pts_per_cell=plan.P / float(np.prod(plan.n_up)))
    return plan, c, info


# ---------------------------------------------------------------------------
# timing

def _outlier(ts):
    return max(ts) > 2.0 * statistics.median(ts)


def gpu_wake(ms):
    """Keep the GPU busy for ~ms (matmul loop) so the timed reps that follow
    run at steady clocks; a few hundred ms of GPU idle (e.g. the host-side
    kernel_ft of a 1D plan build) is enough to drop them."""
    if ms <= 0:
        return
    a = mx.ones((2048, 2048), dtype=mx.float32)
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < ms:
        mx.eval(a @ a)
    mx.synchronize()


def sync_floor(n=50):
    """Median cost of one stage boundary (eval of a trivial op + synchronize):
    the fixed overhead each per-stage measurement adds over the pipelined
    whole execute."""
    a = mx.zeros((1,))
    ts = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(a + 1.0)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def time_whole(plan, inp, reps, warm, wake_ms):
    gpu_wake(wake_ms)
    for _ in range(warm):
        plan.execute(inp)

    def loop():
        ts = []
        for _ in range(reps):
            mx.synchronize()
            t0 = time.perf_counter()
            out = plan.execute(inp)
            mx.synchronize()
            ts.append(time.perf_counter() - t0)
        return ts, out

    ts, out = loop()
    reran = _outlier(ts)
    if reran:                       # one re-run if a rep is > 2x the median
        ts, out = loop()
    return ts, out, reran


def _staged_once(stage_fn, plan, inp):
    times = {}
    last = None
    mx.synchronize()
    t0 = time.perf_counter()
    for label, arr in stage_fn(plan, inp):
        if isinstance(arr, mx.array):
            mx.eval(arr)
        mx.synchronize()
        t = time.perf_counter()
        times[label] = t - t0
        t0 = t
        last = arr
    return times, last


def time_stages(stage_fn, plan, inp, reps, warm, wake_ms):
    gpu_wake(wake_ms)
    for _ in range(warm):
        _staged_once(stage_fn, plan, inp)

    def loop():
        runs, out = [], None
        for _ in range(reps):
            times, out = _staged_once(stage_fn, plan, inp)
            runs.append(times)
        return runs, out

    runs, out = loop()
    reran = _outlier([sum(r.values()) for r in runs])
    if reran:
        runs, out = loop()
    return runs, out, reran


def run_case(spec, eps, reps, warm, wake_ms, floor_s):
    mx.reset_peak_memory()
    plan, inp, info = build(spec, eps)
    ts, out_whole, reran_w = time_whole(plan, inp, reps, warm, wake_ms)
    whole_med = statistics.median(ts)
    row = dict(id=spec["id"], type=spec["type"], eps=eps, plan=info,
               whole_ms=dict(median=whole_med * 1e3, min=min(ts) * 1e3,
                             reps=[t * 1e3 for t in ts]),
               reran_whole=reran_w)
    if spec["type"] == 3 and plan.slab_mode:
        row.update(stages=[], sum_stages_median_ms=None, sum_over_whole=None,
                   staged_rel_l2=None, reran_stages=False,
                   note="slab-mode plan: whole-execute timing only")
    else:
        runs, out_staged, reran_s = time_stages(STAGE_FNS[spec["type"]],
                                                plan, inp, reps, warm, wake_ms)
        labels = list(runs[0])
        stages = []
        for lab in labels:
            v = [r[lab] for r in runs]
            med = statistics.median(v)
            stages.append(dict(label=lab, median_ms=med * 1e3,
                               min_ms=min(v) * 1e3,
                               pct_of_whole=100.0 * med / whole_med))
        ssum = sum(s["median_ms"] for s in stages)
        row.update(
            stages=stages, sum_stages_median_ms=ssum,
            staged_total_median_ms=statistics.median(
                [sum(r.values()) for r in runs]) * 1e3,
            sum_over_whole=ssum / (whole_med * 1e3),
            # excess of the staged sum over the whole execute, next to what
            # the stage boundaries alone cost at the measured sync floor
            excess_ms=ssum - whole_med * 1e3,
            boundaries_floor_ms=len(stages) * floor_s * 1e3,
            staged_rel_l2=rel_l2(out_staged, out_whole),
            reran_stages=reran_s)
    row["peak_gib"] = mx.get_peak_memory() / 2 ** 30
    del plan, inp
    mx.clear_cache()
    return row


# ---------------------------------------------------------------------------
# reporting

def git_info():
    def g(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    return dict(sha=g("rev-parse", "HEAD"),
                branch=g("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(g("status", "--porcelain", "--untracked-files=no")))


def _fmt_count(n):
    m, e = f"{n:.1e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def case_label(row):
    p = row["plan"]
    if row["type"] in (1, 2):
        N = p["N"]
        if len(N) == 1:
            geom = f"N=2^{int(np.log2(N[0]))}" if N[0] & (N[0] - 1) == 0 \
                else f"N={N[0]}"
        else:
            geom = f"{N[0]}^{len(N)}" if len(set(N)) == 1 \
                else "x".join(str(n) for n in N)
        return f"t{row['type']} {p['dim']}D {geom} M={_fmt_count(p['M'])}"
    if p["dim"] == 3:
        return (f"t3 3D generic N={int(np.sqrt(p['M']))} "
                f"P={_fmt_count(p['P'])} M={_fmt_count(p['M'])}")
    return f"t3 1D X=S=300 P={_fmt_count(p['P'])} M={_fmt_count(p['M'])}"


def _fmt_ms(v):
    return f"{v:.2f}" if v < 100 else f"{v:.1f}"


def to_markdown(doc):
    m = doc["meta"]
    out = [f"# Per-stage execute profile: mlx-nufft {m['mlx_nufft_version']}",
           "",
           f"- machine: {m['machine']}",
           f"- mlx {m['mlx_version']}, python {m['python']}",
           f"- git {m['git']['sha'][:12]} ({m['git']['branch']}"
           f"{', dirty' if m['git']['dirty'] else ''})",
           f"- run: {m['timestamp']}",
           f"- protocol: per measurement, the GPU is kept busy for "
           f"{m['wake_ms']:.0f} ms (matmul loop) so the clocks are at steady "
           f"state, then {m['warm']} warm-up executes, then {m['reps']} "
           "timed executes; median and min reported; "
           "mx.synchronize() before and after every timed region. "
           "Stage times come from a second set of runs that re-issue "
           "execute()'s kernel sequence with an eval + synchronize after "
           "each stage; sum/whole is the sum of stage medians over the "
           "whole-execute median (the per-stage syncs cost pipelining, so "
           "sum/whole above 1 is expected; rows outside "
           f"1 +/- {FLAG_TOL:.2f} are flagged). Stage percentages are of "
           "the whole-execute median. A rep more than 2x its median "
           "triggered one re-run of that measurement (noted per row). "
           f"Measured stage-boundary floor: {m['sync_floor_s'] * 1e3:.3f} ms "
           "per eval + synchronize; for flagged rows the Reading section "
           "puts the excess next to the cost of the stage boundaries at "
           "that floor.",
           "- inputs: n_trans=1, complex64, fixed seed, default plan "
           "options (upsampfac 2.0 for types 1/2 and 1.25 for type 3, "
           "spread_method auto, points_backend auto, MLX FFT, crit64). "
           "h2d/d2h are the input upload and result download that "
           "execute() performs.",
           ""]
    # group rows sharing a stage list (one table per type family)
    groups = {}
    for row in doc["cases"]:
        key = (row["type"], tuple(s["label"] for s in row["stages"]))
        groups.setdefault(key, []).append(row)
    for (typ, labels), rows in groups.items():
        dims = sorted({r["plan"]["dim"] for r in rows})
        out.append(f"## Type {typ}, {'/'.join(f'{d}D' for d in dims)}")
        out.append("")
        if not labels:
            out.append("| case | eps | whole med (ms) | whole min (ms) | "
                       "note |")
            out.append("|---|---|---|---|---|")
            for r in rows:
                out.append(f"| {case_label(r)} | {r['eps']:.0e} | "
                           f"{_fmt_ms(r['whole_ms']['median'])} | "
                           f"{_fmt_ms(r['whole_ms']['min'])} | "
                           f"{r.get('note', '')} |")
            out.append("")
            continue
        head = (["case", "eps", "whole med (ms)"] + list(labels)
                + ["whole min (ms)", "sum/whole"])
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for r in rows:
            cells = [case_label(r), f"{r['eps']:.0e}",
                     _fmt_ms(r["whole_ms"]["median"])]
            for s in r["stages"]:
                cells.append(f"{_fmt_ms(s['median_ms'])} "
                             f"({s['pct_of_whole']:.0f}%)")
            ratio = r["sum_over_whole"]
            flag = " FLAG" if abs(ratio - 1.0) > FLAG_TOL else ""
            rerun = (" (re-run)" if r["reran_whole"] or r["reran_stages"]
                     else "")
            cells += [_fmt_ms(r["whole_ms"]["min"]),
                      f"{ratio:.2f}{flag}{rerun}"]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    out.append("## Plan variants, build time and memory")
    out.append("")
    out.append("| case | eps | w | n_up | pts/cell | path | plan build (s) "
               "| peak GiB | staged vs whole rel-L2 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in doc["cases"]:
        p = r["plan"]
        rl = r["staged_rel_l2"]
        out.append(f"| {case_label(r)} | {r['eps']:.0e} | {p['w']} | "
                   f"{'x'.join(str(v) for v in p['n_up'])} | "
                   f"{p['pts_per_cell']:.3g} | {p['path']} | "
                   f"{p['t_plan_s']:.3f} | {r['peak_gib']:.2f} | "
                   f"{'n/a' if rl is None else f'{rl:.1e}'} |")
    out.append("")
    out.append("## Reading")
    out.append("")
    flags = []
    for r in doc["cases"]:
        if not r["stages"]:
            continue
        top = max(r["stages"], key=lambda s: s["median_ms"])
        out.append(f"- {case_label(r)} eps={r['eps']:.0e}: dominant stage "
                   f"{top['label']} at {top['pct_of_whole']:.0f}% "
                   f"({_fmt_ms(top['median_ms'])} of "
                   f"{_fmt_ms(r['whole_ms']['median'])} ms)")
        if abs(r["sum_over_whole"] - 1.0) > FLAG_TOL:
            flags.append(f"{case_label(r)} eps={r['eps']:.0e} "
                         f"(sum/whole {r['sum_over_whole']:.2f}: excess "
                         f"{r['excess_ms']:.2f} ms vs {len(r['stages'])} "
                         f"boundaries x floor = "
                         f"{r['boundaries_floor_ms']:.2f} ms)")
    out.append("")
    out.append("Flagged (sum of stages vs whole execute differs by more than "
               f"{100 * FLAG_TOL:.0f}%): "
               + ("; ".join(flags) if flags else "none"))
    out.append("")
    return "\n".join(out)


def main(argv):
    reps, warm, tag, only = 7, 2, "profile_baseline", None
    wake_ms = WAKE_MS
    args = list(argv)
    if "--list" in args:
        for c in CASES:
            print(c["id"])
        return
    if "--reps" in args:
        reps = int(args[args.index("--reps") + 1])
    if "--warm" in args:
        warm = int(args[args.index("--warm") + 1])
    if "--wake-ms" in args:
        wake_ms = float(args[args.index("--wake-ms") + 1])
    if "--tag" in args:
        tag = args[args.index("--tag") + 1]
    if "--cases" in args:
        only = args[args.index("--cases") + 1].split(",")
    specs = [c for c in CASES if only is None or c["id"] in only]
    if only is not None and len(specs) != len(only):
        raise SystemExit(f"unknown case id in {only}; see --list")

    meta = dict(machine=machine(), mlx_version=mx.__version__,
                mlx_nufft_version=mlx_nufft.__version__,
                python=platform.python_version(), git=git_info(),
                timestamp=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                reps=reps, warm=warm, wake_ms=wake_ms, seed=SEED,
                eps=list(EPS_LIST), sync_floor_s=sync_floor())
    print(f"machine: {meta['machine']}\nmlx {meta['mlx_version']}  "
          f"mlx_nufft {meta['mlx_nufft_version']}  git {meta['git']['sha'][:12]}"
          f"  reps={reps} warm={warm}  sync floor "
          f"{meta['sync_floor_s'] * 1e3:.3f} ms", flush=True)

    rows = []
    t_start = time.perf_counter()
    for spec in specs:
        for eps in EPS_LIST:
            t0 = time.perf_counter()
            row = run_case(spec, eps, reps, warm, wake_ms,
                           meta["sync_floor_s"])
            rows.append(row)
            st = "  ".join(f"{s['label']} {_fmt_ms(s['median_ms'])}"
                           f"({s['pct_of_whole']:.0f}%)" for s in row["stages"])
            ratio = row["sum_over_whole"]
            print(f"[{spec['id']} eps={eps:.0e}] whole med "
                  f"{_fmt_ms(row['whole_ms']['median'])} ms min "
                  f"{_fmt_ms(row['whole_ms']['min'])} ms  sum/whole "
                  f"{'n/a' if ratio is None else f'{ratio:.2f}'}  "
                  f"({time.perf_counter() - t0:.0f}s)\n    {st}", flush=True)
    doc = dict(meta=meta, cases=rows)
    doc["meta"]["wall_s"] = time.perf_counter() - t_start

    resdir = ROOT / "results"
    resdir.mkdir(exist_ok=True)
    (resdir / f"{tag}.json").write_text(json.dumps(doc, indent=1))
    (resdir / f"{tag}.md").write_text(to_markdown(doc))
    print(f"wrote results/{tag}.json and results/{tag}.md "
          f"({doc['meta']['wall_s']:.0f}s total)")


if __name__ == "__main__":
    main(sys.argv[1:])
