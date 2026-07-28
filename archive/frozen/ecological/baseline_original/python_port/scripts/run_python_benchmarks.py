"""Run Python-only MC-UAMS vs baselines and save artifacts in python_port/results.

This script does not write into data/POLICYX or other R-oriented output paths.
All generated artifacts are saved under python_port/results/<example>/.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _project_root() -> Path:
    # python_port/scripts -> project root is two levels up
    return Path(__file__).resolve().parents[2]


def _python_port_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
PYTHON_PORT = _python_port_root()

# Make ported modules importable.
for rel in (
    "src/building hmMDP",
    "src/simulations",
    "src/Dirichlet solver",
    "src/utils",
    "data",
):
    p = PYTHON_PORT / rel
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dirichlet_solver import dirichlet_solver_from_config
from generate_pomdpx import write_hmMDPx
from generate_random_mdp import generate_random_mdp
from mdp_tools import finite_horizon
from mean_parameters import mean_parameters
from read_policyx import read_policyx2
from sim_mdp_momdp_policy import sim_mdp_momdp_policy
from sim_mdp_parameter_uncertainty import sim_mdp_parameter_uncertainty


@dataclass
class PerfRow:
    algorithm: str
    mean_gap_tmax: float
    ci95_gap_tmax: float
    mean_gap_hmmdp: float | None
    ci95_gap_hmmdp: float | None


def _ci95(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def _resolve_sarsop_binary(root: Path) -> Path:
    candidates = [
        root / "sarsop" / "src" / "pomdpsol",
        root / "sarsop" / "src" / "pomdpsol.exe",
        root / "sarsop" / "src" / "pomdpsol.bin",
    ]
    for c in candidates:
        if c.exists():
            return c
    tried = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find SARSOP solver binary. Tried:\n{tried}")


def _build_transition_models(mean_df: pd.DataFrame, reward: np.ndarray) -> list[np.ndarray]:
    num_s, num_a = reward.shape
    p_cols = [c for c in mean_df.columns if c.startswith("p")]
    models: list[np.ndarray] = []
    for _, row in mean_df.sort_values("opt").iterrows():
        params = row[p_cols].to_numpy(dtype=float)
        model = np.zeros((num_s, num_s, num_a), dtype=float)
        for act_id in range(1, num_a + 1):
            idx = (act_id - 1) * 2
            mat = np.array(
                [[params[idx], 1 - params[idx]], [params[idx + 1], 1 - params[idx + 1]]],
                dtype=float,
            )
            model[:, :, act_id - 1] = mat
        models.append(model)
    return models


def _build_momdp_matrices(mean_df: pd.DataFrame, reward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_s, num_a = reward.shape
    models = _build_transition_models(mean_df, reward)
    num_mod = len(models)
    tr_by_action = [np.zeros((num_mod * num_s, num_mod * num_s), dtype=float) for _ in range(num_a)]
    for i, model in enumerate(models):
        block = slice(i * num_s, (i + 1) * num_s)
        for a in range(num_a):
            tr_by_action[a][block, block] = model[:, :, a]
    tr_momdp = np.stack(tr_by_action, axis=2)
    obs_base = np.vstack([np.eye(num_s) for _ in range(num_mod)])
    obs_momdp = np.stack([obs_base for _ in range(num_a)], axis=2)
    return tr_momdp, obs_momdp


def _build_true_mdp(row: np.ndarray, s: int, a: int) -> np.ndarray:
    mats = []
    for act in range(a):
        vals = row[act * s * s : (act + 1) * s * s]
        mats.append(np.asarray(vals, dtype=float).reshape(s, s))
    return np.stack(mats, axis=2)


def _load_config(example: str) -> Dict:
    module = importlib.import_module(example)
    cfg = dict(module.CONFIG)
    cfg["module_name"] = example
    return cfg


def _extract_sarsop_summary(log_text: str) -> str:
    """Extract final SARSOP summary block with timeout/table metrics."""
    lines = log_text.splitlines()
    # Prefer block starting at "SARSOP finishing ..." (or truncated variant).
    start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if "SARSOP finishing" in line or "ARSOP finishing" in line:
            start_idx = i
            break

    if start_idx is not None:
        end_idx = min(len(lines), start_idx + 20)
        return "\n".join(lines[start_idx:end_idx]).strip() + "\n"

    # Fallback: use the last table block separator.
    sep = "-------------------------------------------------------------------------------"
    sep_idx = [i for i, line in enumerate(lines) if line.strip() == sep]
    if len(sep_idx) >= 3:
        # usually table is bounded by three separators in the footer section
        start = sep_idx[-3]
        end = min(len(lines), sep_idx[-1] + 1)
        return "\n".join(lines[start:end]).strip() + "\n"

    # Last resort: tail of the log.
    return "\n".join(lines[-20:]).strip() + "\n"


def _format_float(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{digits}f}"


def _write_methods_summary(
    out_path: Path,
    example: str,
    tmax: int,
    tmax_hmmdp: int,
    df_opt: pd.DataFrame,
    df_mc_mean: pd.DataFrame,
    df_pubd_mean: pd.DataFrame,
) -> None:
    """Write compact cross-method summary in a human-readable table."""
    c_tmax = f"V{tmax}"
    c_h = f"V{tmax_hmmdp}"

    # Common horizon (all methods available)
    v_opt_t = df_opt[c_tmax].to_numpy()
    v_mc_t = df_mc_mean[c_tmax].to_numpy()
    v_pubd_t = df_pubd_mean[c_tmax].to_numpy()

    gap_mc_t = (v_opt_t - v_mc_t) / v_opt_t
    gap_pubd_t = (v_opt_t - v_pubd_t) / v_opt_t

    rows_common = [
        ("MC-UAMS", np.mean(v_mc_t), _ci95(v_mc_t), np.mean(gap_mc_t), _ci95(gap_mc_t)),
        ("PUBD", np.mean(v_pubd_t), _ci95(v_pubd_t), np.mean(gap_pubd_t), _ci95(gap_pubd_t)),
        ("Optimal", np.mean(v_opt_t), _ci95(v_opt_t), 0.0, 0.0),
    ]

    # Long horizon (PUBD typically unavailable)
    rows_long = []
    if c_h in df_opt.columns and c_h in df_mc_mean.columns:
        v_opt_h = df_opt[c_h].to_numpy()
        v_mc_h = df_mc_mean[c_h].to_numpy()
        gap_mc_h = (v_opt_h - v_mc_h) / v_opt_h
        rows_long.append(("MC-UAMS", np.mean(v_mc_h), _ci95(v_mc_h), np.mean(gap_mc_h), _ci95(gap_mc_h)))
        rows_long.append(("Optimal", np.mean(v_opt_h), _ci95(v_opt_h), 0.0, 0.0))

    sep = "-------------------------------------------------------------------------------"
    lines = [
        sep,
        " Method Comparison Summary",
        f" Example     : {example}",
        f" #MDP        : {len(df_opt)}",
        sep,
        f" Common Horizon (Tmax = {tmax})",
        sep,
        " Method   |MeanValue |CI95(Value)|MeanGap   |CI95(Gap)",
        sep,
    ]
    for method, mean_v, ci_v, mean_gap, ci_gap in rows_common:
        lines.append(
            f" {method:<8}|{_format_float(mean_v):>9} |{_format_float(ci_v):>11} |{_format_float(mean_gap):>9} |{_format_float(ci_gap):>9}"
        )
    lines.append(sep)

    if rows_long:
        lines.extend(
            [
                "",
                f" Extended Horizon (Tmax_hmMDP = {tmax_hmmdp})",
                sep,
                " Method   |MeanValue |CI95(Value)|MeanGap   |CI95(Gap)",
                sep,
            ]
        )
        for method, mean_v, ci_v, mean_gap, ci_gap in rows_long:
            lines.append(
                f" {method:<8}|{_format_float(mean_v):>9} |{_format_float(ci_v):>11} |{_format_float(mean_gap):>9} |{_format_float(ci_gap):>9}"
            )
        lines.append(sep)
        lines.append(" Note: PUBD is finite-horizon (Tmax) and is not reported at Tmax_hmMDP.")
    else:
        lines.append(" Note: Extended horizon summary unavailable for this run.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    cfg = _load_config(args.example)
    reward = np.asarray(cfg["reward"], dtype=float)
    b_full = np.asarray(cfg["b_full"], dtype=float)
    gamma = float(cfg["gamma"])
    tmax = int(cfg["Tmax"])
    tmax_hmmdp = int(cfg["Tmax_hmMDP"])
    n_mdp = int(args.n_mdp if args.n_mdp is not None else cfg["N_MDP"])
    s, a = reward.shape

    out_dir = (PYTHON_PORT / "results" / args.example).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Building MC-UAMS model for '{args.example}'...")

    # 1) Build MC-UAMS mean parameters + local POMDPX + local policyx
    mean_df = mean_parameters(reward, gamma, n_trials=args.n_trials, seed=args.seed)
    mean_csv = out_dir / f"{args.example}_meanparams.csv"
    mean_df.to_csv(mean_csv, index=False)

    transitions = _build_transition_models(mean_df, reward)
    b_par = np.repeat(1.0 / len(transitions), len(transitions))
    pomdpx = out_dir / f"{args.example}.pomdpx"
    policyx = out_dir / f"{args.example}.policyx"
    write_hmMDPx(transitions, reward, b_full, b_par, gamma, str(pomdpx))

    solver = _resolve_sarsop_binary(ROOT)
    cmd = [
        str(solver),
        str(pomdpx),
        "--precision",
        str(args.precision),
        "--timeout",
        str(args.timeout),
        "--output",
        str(policyx),
    ]
    sarsop_log_path = out_dir / f"{args.example}_sarsop.log"
    sarsop_summary_path = out_dir / f"{args.example}_sarsop_summary.txt"
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        full_log = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    except subprocess.CalledProcessError as exc:
        full_log = (exc.stdout or "") + ("\n" if exc.stdout and exc.stderr else "") + (exc.stderr or "")
        sarsop_log_path.write_text(full_log, encoding="utf-8")
        sarsop_summary_path.write_text(_extract_sarsop_summary(full_log), encoding="utf-8")
        raise

    sarsop_log_path.write_text(full_log, encoding="utf-8")
    sarsop_summary_path.write_text(_extract_sarsop_summary(full_log), encoding="utf-8")
    print(f"[2/4] SARSOP solved. Summary saved: {sarsop_summary_path}")

    # 2) Build random true MDPs (local file)
    random_mdp_csv = out_dir / f"{args.example}_random_mdp.csv"
    generate_random_mdp(reward, n_mdp, str(random_mdp_csv), seed=args.seed)
    random_mdp = pd.read_csv(random_mdp_csv).to_numpy(dtype=float)
    print(f"[3/4] Running simulations on {n_mdp} random MDPs x {args.n_sim_it} trajectories...")

    # 3) Simulate MC-UAMS vs baselines
    alpha_momdp = read_policyx2(str(policyx))
    tr_momdp, obs_momdp = _build_momdp_matrices(mean_df, reward)

    pubd = dirichlet_solver_from_config(cfg)
    v_mdp = pubd["V_MDP"]
    states = pubd["states"]
    state_index_map = pubd["state_index_map"]

    mc_rows = []
    pubd_rows = []
    opt_rows = []
    progress_every = max(1, n_mdp // 10)
    for i in range(n_mdp):
        tr_mdp = _build_true_mdp(random_mdp[i, :], s, a)

        # Proposal
        sim_mc = sim_mdp_momdp_policy(
            b_full,
            tmax_hmmdp,
            tr_mdp,
            reward,
            tr_momdp,
            obs_momdp,
            alpha_momdp,
            disc=gamma,
            n_it=args.n_sim_it,
            seed=None if args.seed is None else args.seed + i,
        )
        n_mod = alpha_momdp["vectors"].shape[0]
        mc_rows.append(sim_mc[:, n_mod : tmax_hmmdp + n_mod])

        # Baseline 1: PUBD
        sim_pubd = sim_mdp_parameter_uncertainty(
            b_full,
            tr_mdp,
            reward,
            states,
            state_index_map,
            v_mdp,
            disc=gamma,
            n_it=args.n_sim_it,
            seed=None if args.seed is None else args.seed + 10_000 + i,
        )
        pubd_rows.append(sim_pubd)

        # Baseline 2: True optimal policy value
        opt = finite_horizon(tr_mdp, reward, gamma, tmax_hmmdp)
        v_opt = opt.V[int(np.argmax(b_full)), ::-1]
        opt_rows.append(v_opt)
        if (i + 1) % progress_every == 0 or (i + 1) == n_mdp:
            print(f"  - completed {i + 1}/{n_mdp} MDPs")

    sim_mc_all = np.vstack(mc_rows)
    sim_pubd_all = np.vstack(pubd_rows)
    sim_opt_all = np.vstack(opt_rows)

    sim_mc_csv = out_dir / f"{args.example}_sim_mcuams.csv"
    sim_pubd_csv = out_dir / f"{args.example}_sim_pubd.csv"
    sim_opt_csv = out_dir / f"{args.example}_sim_optimal.csv"
    pd.DataFrame(sim_mc_all).to_csv(sim_mc_csv, index=False)
    pd.DataFrame(sim_pubd_all).to_csv(sim_pubd_csv, index=False)
    pd.DataFrame(sim_opt_all).to_csv(sim_opt_csv, index=False)

    # 4) Performance summary
    # Column mapping mirrors the project convention V1, V2, ...
    cols_opt = [f"V{i+1}" for i in range(sim_opt_all.shape[1])]
    cols_mc = [f"V{i+1}" for i in range(sim_mc_all.shape[1])]
    cols_pubd = [f"V{i+1}" for i in range(sim_pubd_all.shape[1])]
    df_opt = pd.DataFrame(sim_opt_all, columns=cols_opt)
    df_mc = pd.DataFrame(sim_mc_all, columns=cols_mc)
    df_pubd = pd.DataFrame(sim_pubd_all, columns=cols_pubd)

    mdp_ids = np.repeat(np.arange(1, n_mdp + 1), args.n_sim_it)
    df_mc["mdp_id"] = mdp_ids
    df_pubd["mdp_id"] = mdp_ids
    df_mc_mean = df_mc.groupby("mdp_id").mean(numeric_only=True).reset_index(drop=True)
    df_pubd_mean = df_pubd.groupby("mdp_id").mean(numeric_only=True).reset_index(drop=True)

    c_tmax = f"V{tmax}"
    c_h = f"V{tmax_hmmdp}"
    gap_mc_t = (df_opt[c_tmax] - df_mc_mean[c_tmax]) / df_opt[c_tmax]
    gap_pubd_t = (df_opt[c_tmax] - df_pubd_mean[c_tmax]) / df_opt[c_tmax]
    gap_mc_h = (df_opt[c_h] - df_mc_mean[c_h]) / df_opt[c_h]

    perf = [
        PerfRow(
            algorithm="MC-UAMS",
            mean_gap_tmax=float(np.mean(gap_mc_t)),
            ci95_gap_tmax=float(_ci95(gap_mc_t.to_numpy())),
            mean_gap_hmmdp=float(np.mean(gap_mc_h)),
            ci95_gap_hmmdp=float(_ci95(gap_mc_h.to_numpy())),
        ),
        PerfRow(
            algorithm="PUBD",
            mean_gap_tmax=float(np.mean(gap_pubd_t)),
            ci95_gap_tmax=float(_ci95(gap_pubd_t.to_numpy())),
            mean_gap_hmmdp=None,
            ci95_gap_hmmdp=None,
        ),
    ]
    perf_csv = out_dir / f"{args.example}_performance_summary.csv"
    pd.DataFrame([asdict(x) for x in perf]).to_csv(perf_csv, index=False)
    methods_summary_txt = out_dir / f"{args.example}_methods_summary.txt"
    _write_methods_summary(
        methods_summary_txt,
        args.example,
        tmax,
        tmax_hmmdp,
        df_opt,
        df_mc_mean,
        df_pubd_mean,
    )

    print("[4/4] Writing summary files...")
    metadata = {
        "example": args.example,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "n_mdp": n_mdp,
        "n_sim_it": args.n_sim_it,
        "precision": args.precision,
        "timeout": args.timeout,
        "artifacts": {
            "meanparams_csv": str(mean_csv),
            "pomdpx": str(pomdpx),
            "policyx": str(policyx),
            "sarsop_log": str(sarsop_log_path),
            "sarsop_summary": str(sarsop_summary_path),
            "random_mdp_csv": str(random_mdp_csv),
            "sim_mcuams_csv": str(sim_mc_csv),
            "sim_pubd_csv": str(sim_pubd_csv),
            "sim_optimal_csv": str(sim_opt_csv),
            "performance_summary_csv": str(perf_csv),
            "methods_summary_txt": str(methods_summary_txt),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved all Python-only benchmark artifacts to: {out_dir}")
    print(f"SARSOP summary file: {sarsop_summary_path}")
    print(f"Methods summary file: {methods_summary_txt}")
    print(f"MC-UAMS mean gap @Tmax: {np.mean(gap_mc_t):.4f}")
    print(f"PUBD   mean gap @Tmax: {np.mean(gap_pubd_t):.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Python-only MC-UAMS vs baselines.")
    p.add_argument(
        "--example",
        default=os.environ.get("EXAMPLE_MODULE", "gouldian"),
        help="Example module name in python_port/data (e.g., gouldian, potoroo).",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-trials", type=int, default=10_000, help="MC draws for mean parameter estimation.")
    p.add_argument("--n-mdp", type=int, default=None, help="Override number of random true MDPs.")
    p.add_argument("--n-sim-it", type=int, default=100, help="Sim trajectories per true MDP.")
    p.add_argument("--precision", type=float, default=0.1, help="SARSOP precision.")
    p.add_argument("--timeout", type=int, default=3600, help="SARSOP timeout in seconds.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
