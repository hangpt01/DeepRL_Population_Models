#!/usr/bin/env python3
"""Instrumented diagnostic replay of accepted P=10 policies (parity-gated).

Replays an already-accepted (cell, method) policy with side-effect-free logging.
Parity gate: process_noise_sigma=0 and fixed seeds => the replay must reproduce the
accepted 7 summary fields to <= 1e-9, else the run is VOID.

Design for parity safety:
  * The exact accepted policy is rebuilt from the frozen fit-cache (no refit).
  * The UNMODIFIED frozen `ContinuousEvaluator.run` drives the rollout.
  * Logging is added non-invasively: (a) the env instance's __class__ is swapped to a
    subclass whose step() wraps super().step() (identical float ops / RNG), recording a
    per-step row and an exact one-step shadow rollout (pure `transition_value`, no RNG,
    no mutation); (b) the policy instance's bound `act` is wrapped to copy last_diagnostics.
  * The shadow rollout asserts env state and every RNG bit-generator state are unchanged.

Return-blindness does NOT apply and is not claimed: the parity gate deliberately uses the
accepted returns as the invariant to match.

Outputs (no parquet lib available -> npz + csv.gz, documented in the receipt):
  logs/<cell>__<method>.npz  logs/<cell>__<method>.scalars.csv.gz
  derived/<cell>__<method>.json   parity/PARITY_REPORT.json
  DIAGNOSTIC_REPLAY_RECEIPT.json  manifest.json
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(
    os.environ.get("DEEPRL_REPO_ROOT", Path(__file__).resolve().parents[3])
).resolve()
DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import require_scientific_tables  # noqa: E402

SCRATCH_PROJECT = Path(os.environ.get(
    "DEEPRL_SCRATCH_PROJECT",
    "/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models",
))
ROOT = Path(os.environ.get("DEEPRL_RUNS_ROOT", SCRATCH_PROJECT / "real_ecology_runs"))
P10_DATA = Path(os.environ.get(
    "DEEPRL_P10_DATA_ROOT",
    ROOT / "three_species_ecological_p10_correction_20260723_v1",
))
P10_PACKAGE = Path(os.environ.get(
    "DEEPRL_P10_PACKAGE",
    REPO_ROOT / "experiments" / "accepted_p10",
))
REPLAY = Path(os.environ.get(
    "DEEPRL_REPLAY_OUTPUT",
    REPO_ROOT / ".verification" / "diagnostic_replay",
))
ACCEPTED_CSV = Path(os.environ.get(
    "DEEPRL_ACCEPTED_CSV",
    REPO_ROOT / "results" / "accepted" / "MATCHED_P10_144_METHOD_CELLS.csv",
))
ACCEPTED_CSV_SHA = "7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c"

ECOLOGICAL_SCRIPTS = Path(os.environ.get(
    "DEEPRL_ECOLOGICAL_SCRIPTS", REPO_ROOT / "scripts" / "ecological"
))
ECOLOGICAL_SRC = Path(os.environ.get(
    "DEEPRL_ECOLOGICAL_SRC", REPO_ROOT / "src" / "tracks" / "ecological"
))
sys.path.insert(0, str(ECOLOGICAL_SCRIPTS))
sys.path.insert(0, str(ECOLOGICAL_SRC))
import run_real_manifest_row as rrmr  # noqa: E402
from real_ecology_benchmark import evaluator as evaluator_mod  # noqa: E402
from real_ecology_benchmark.evaluator import ContinuousEvaluator  # noqa: E402
from real_ecology_benchmark.envs import make_env as real_make_env  # noqa: E402
from real_ecology_benchmark.config import load_config, hides_rk  # noqa: E402
from real_ecology_benchmark.controls import advance_public_controls, private_r_eff  # noqa: E402
from real_ecology_benchmark.pipeline import (  # noqa: E402
    ensure_dataset, _load_or_fit_public_surrogate, _hidden_method_context,
    make_filter_factory, build_method,
)

# Embedded confirmed constants (audits 1&2). Used for LOG/evaluator diagnostics only,
# never fed to any method. Verified against cfg at runtime.
SSAFE = {"Amur tiger": 25.0, "Crab-eating fox": 10.25, "Egyptian vulture": 81.25}
KREF = {"Amur tiger": 250.0, "Crab-eating fox": 41.0, "Egyptian vulture": 325.0}
METHOD_ID = {
    "moor": "moor_adapted_ricker_misspec_pbvi",
    "plus": "plus_adapted_ricker_only_pbvi",
}
CONFIG = {"moor": "moor_ricker_p10.yaml", "plus": "plus_ricker_only_p10.yaml"}
CELLS = {
    "A1": ("Crab-eating fox", "ricker", "0.2"), "A2": ("Crab-eating fox", "allee", "0.2"),
    "A3": ("Crab-eating fox", "regime", "0.2"), "A4": ("Crab-eating fox", "theta", "0.2"),
    "A5": ("Amur tiger", "theta", "0.1"), "A6": ("Egyptian vulture", "ricker", "0.1"),
}
SEEDS = [b + i for b in (7001, 7051, 7101, 7151, 7201) for i in range(4)]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def slug(v: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", v.lower()).strip("_")


def sig_slug(v):
    return f"{float(v):g}".replace("-", "m").replace(".", "p")


def accepted_fields(pop, env, sigma, method_id):
    for r in csv.DictReader(open(ACCEPTED_CSV)):
        if (r["population"] == pop and r["environment"] == env
                and r["sigma_obs"] == sigma and r["method"] == method_id):
            return {
                "return_mean": float(r["operational_return_mean"]),
                "return_sd": float(r["operational_return_sd"]),
                "unsafe_fraction_mean": float(r["unsafe_fraction"]),
                "persistence_mean": float(r["persistence_mean"]),
                "collapse_entry_mean": float(r["collapse_rate"]),
                "min_population_mean": float(r["min_population_mean"]),
                "economic_cost_mean": float(r["economic_cost_mean"]),
                "dataset_sha256": r["dataset_sha256"],
            }
    raise SystemExit(f"accepted row not found: {pop}/{env}/{sigma}/{method_id}")


def manifest_row(side, pop, env, sigma):
    name = f"{side}_p10_plan_24.csv"
    for r in csv.DictReader(open(P10_PACKAGE / "manifests" / name)):
        if r["population"] == pop and r["environment"] == env and r["sigma_obs"] == sigma:
            return r
    raise SystemExit(f"manifest row not found: {side} {pop}/{env}/{sigma}")


# --------------------------------------------------------------------------- logger
class ReplayLogger:
    def __init__(self, cfg, method, pop):
        self.cfg = cfg
        self.env_cfg = cfg.environment
        self.method = method
        self.s_safe = float(cfg.environment.safety_threshold)
        self.k_ref = float(cfg.environment.K_ref)
        assert abs(self.s_safe - SSAFE[pop]) < 1e-9, (self.s_safe, SSAFE[pop])
        assert abs(self.k_ref - KREF[pop]) < 1e-9, (self.k_ref, KREF[pop])
        self.surrogate = None          # set by driver (for surrogate_reward_t)
        self.pop_id = None
        self.rows = []                 # list of dict per step
        self._pending = None           # policy-side pending for the current step
        self._seed = None
        self._t = 0
        self._prev_obs = None
        self._C_ep = None
        self._theta_ep = None

    # ---- policy side ----
    def on_reset(self, env):
        self._seed = int(env._rngs and 0)  # placeholder; set by env.reset override
        self._t = 0
        self._prev_obs = None
        self._C_ep = float(env._C)
        self._theta_ep = float(env._theta)

    def on_act(self, policy, belief, observation, action):
        d = dict(getattr(policy, "last_diagnostics", {}) or {})
        p = {"obs_t": float(observation), "action_t": int(action)}
        if self.method == "moor":
            q = np.asarray(d.get("action_scores"), dtype=np.float64)
            p["q"] = q
            p["belief_entropy"] = float(d.get("internal_belief_entropy", np.nan))
            ib = getattr(policy, "internal_belief", None)
            pom = getattr(policy, "pomdp", None)
            if ib is not None and pom is not None:
                sa = pom.state_abundances()
                p["belief_mean"] = float(np.dot(ib.probabilities, sa))
            else:
                p["belief_mean"] = float("nan")
            order = np.argsort(q)[::-1]
            p["argmax"] = int(order[0])
            p["margin_top1_top2"] = float(q[order[0]] - q[order[1]])
        elif self.method == "plus":
            w = np.asarray(policy.posterior, dtype=np.float64).copy()
            planners, ibs, poms = policy.planners, policy.internal_beliefs, policy.pomdps
            # q_cand is CAPTURED (not recomputed) from the frozen act()'s own action_values
            # calls (see the planner wrapping in run_cell): zero extra PBVI work and the exact
            # decision-path values. Purely observational; does not alter the rollout.
            q_cand = np.array([np.asarray(planners[j]._replay_last_av, dtype=np.float64)
                               for j in range(len(planners))], dtype=np.float64)
            q_weighted = w @ q_cand
            bmean = np.array([float(np.dot(ibs[j].probabilities, poms[j].state_abundances()))
                              for j in range(len(ibs))])
            bent = np.array([float(-np.sum(ibs[j].probabilities * np.log(ibs[j].probabilities + 1e-300)))
                             for j in range(len(ibs))])
            order = np.argsort(q_weighted)[::-1]
            p.update(w=w, q_cand=q_cand, q_weighted=q_weighted, belief_mean=bmean, belief_entropy=bent,
                     argmax_weighted=int(order[0]), argmax_map_only=int(np.argmax(q_cand[0])),
                     argmax_uniform=int(np.argmax(q_cand.mean(axis=0))),
                     argmax_cand=np.argmax(q_cand, axis=1).astype(int),
                     margin_top1_top2=float(q_weighted[order[0]] - q_weighted[order[1]]))
        self._pending = p

    # ---- env side ----
    def shadow(self, env, action_id):
        """Exact one-step reward for every action; pure, no RNG, no mutation."""
        na = env.num_actions
        x_next = np.empty(na, dtype=np.float64)
        for a in range(na):
            x_next[a] = env.transition_value(
                env._state, env.actions[a], env._r_base, env._C, env._theta,
                env._regime, 0.0, env._rho, env._kappa,
            )
        util = x_next / (x_next + self.k_ref)
        cost = np.asarray([env.actions[a].cost for a in range(na)], dtype=np.float64)
        pen = (x_next <= self.s_safe).astype(np.float64)
        true_reward_all = util - cost - 10.0 * pen
        return x_next, true_reward_all

    def taken_dynamics(self, env, action_id):
        a = env.actions[action_id]
        rho_next, _kap, K_eff = advance_public_controls(self.cfg.environment, a.id, env._rho, env._kappa)
        rho_next = float(np.asarray(rho_next).reshape(-1)[0]); K_eff = float(np.asarray(K_eff).reshape(-1)[0])
        r_eff = float(np.asarray(private_r_eff(self.cfg.environment, env._r_base, rho_next, env._theta)).reshape(-1)[0])
        managed = max(env._state + a.stocking_delta, 0.0)
        return dict(r_setpoint_t=rho_next, r_pos_t=max(r_eff, 0.0), r_mort_t=min(r_eff, 0.0),
                    k_t=K_eff, m_t=managed)

    def on_step(self, env, action_id, result, x_next, true_reward_all, regime_prev):
        info = result.evaluator_info
        p = self._pending
        row = dict(
            seed=self._seed, t=self._t,
            x_true_t=float(info["state_previous"]), x_true_next=float(info["state"]),
            obs_t=p["obs_t"], action_t=p["action_t"],
            cost_t=float(env.actions[action_id].cost),
            utility_t=float(info["state"] / (info["state"] + self.k_ref)),
            penalty_flag_t=float(bool(info["safety_penalty_applied"])),
            reward_t=float(result.reward),
            regime_z_t=int(regime_prev), regime_switched_t=int(regime_prev != env._regime),
            allee_C_episode=self._C_ep, theta_exponent_episode=self._theta_ep,
        )
        row.update(self.taken_dynamics_cached)
        row["x_next_all"] = x_next
        row["true_reward_all_actions"] = true_reward_all
        # myopic-oracle diagnostics (state-level discrimination, M15)
        order = np.argsort(true_reward_all)[::-1]
        row["oracle_action"] = int(order[0])
        row["true_margin_top1_top2"] = float(true_reward_all[order[0]] - true_reward_all[order[1]])
        # surrogate reward for the taken action along the visited trajectory (M13)
        if self.method in ("moor", "plus") and self.surrogate is not None:
            nxt = float(result.observation)
            prev = self._prev_obs if self._prev_obs is not None else p["obs_t"]
            rew, _risk = self.surrogate.predict(
                np.array([prev]), np.array([p["obs_t"]]), np.array([nxt]),
                np.array([action_id], dtype=np.int64), np.array([min(self._t, self.surrogate.feature_spec.public_horizon - 1)], dtype=np.int64),
                np.array([self.pop_id]),
            )
            row["surrogate_reward_t"] = float(rew[0])
        else:
            row["surrogate_reward_t"] = float("nan")
        if self.method == "moor":
            row["q"] = p["q"]; row["belief_mean"] = p["belief_mean"]
            row["belief_entropy"] = p["belief_entropy"]; row["argmax"] = p["argmax"]
            row["margin_top1_top2"] = p["margin_top1_top2"]
        elif self.method == "plus":
            for k in ("w", "q_cand", "q_weighted", "belief_mean", "belief_entropy",
                      "argmax_weighted", "argmax_map_only", "argmax_uniform", "argmax_cand",
                      "margin_top1_top2"):
                row[k] = p[k]
            row["argmax_weighted_matches_deployed"] = int(p["argmax_weighted"] == p["action_t"])
        # consistency assert: recomputed reward == evaluator reward
        rr = row["utility_t"] - row["cost_t"] - 10.0 * row["penalty_flag_t"]
        assert abs(rr - row["reward_t"]) < 1e-9, (rr, row["reward_t"])
        self.rows.append(row)
        self._prev_obs = float(result.observation)
        self._t += 1
        self._pending = None


# module-global bound to the current logger for the __class__-swapped env
_LOG: ReplayLogger | None = None


def make_instrumented_class(base):
    class InstrumentedEnv(base):
        def reset(self, seed, state_override=None):
            res = base.reset(self, seed, state_override)
            _LOG.on_reset(self)
            _LOG._seed = int(seed)
            _LOG._C_ep = float(self._C); _LOG._theta_ep = float(self._theta)
            return res

        def step(self, action_id):
            pre_state = self._state
            pre_rng = [g.bit_generator.state for g in self._rngs.values()]
            regime_prev = self._regime
            x_next, true_reward_all = _LOG.shadow(self, action_id)
            # acceptance #7: shadow advanced nothing and consumed no RNG
            assert self._state == pre_state
            assert [g.bit_generator.state for g in self._rngs.values()] == pre_rng
            _LOG.taken_dynamics_cached = _LOG.taken_dynamics(self, action_id)
            result = base.step(self, action_id)
            _LOG.on_step(self, action_id, result, x_next, true_reward_all, regime_prev)
            return result
    return InstrumentedEnv


def build_policy(side, pop, env, sigma):
    method_full = METHOD_ID[side]
    row = manifest_row(side, pop, env, sigma)
    assert row["method"] == method_full
    cfg = load_config(str(P10_PACKAGE / "configs" / CONFIG[side]))
    cfg, cell, _ = rrmr.apply_row_config(cfg, row, P10_DATA / side)
    cfg.validate()
    assert hides_rk(cfg.environment)
    dataset = ensure_dataset(cfg, regenerate=False)
    receipt_gate = rrmr.validate_adapted_fit_receipt_gate(row, P10_DATA / side)
    surrogate, _sp, _st = _load_or_fit_public_surrogate(cfg, dataset)
    ctx = _hidden_method_context(cfg, dataset, surrogate)
    factory, _ = make_filter_factory(cfg, dataset, "faithful_internal", ctx)
    split_info = {"training_enabled": bool(cfg.training.enabled), "holdout_fraction": 0.0,
                  "fit_history_fraction": float(cfg.faithful.fit.history_fraction),
                  "faithful_internal_beliefs": True}
    policy, _ = build_method(method_full, cfg, dataset, factory, cache=None, timings={},
                             holdout_dataset=None, holdout_cache=None,
                             split_info=split_info, method_context=ctx)
    reuse = rrmr.validate_adapted_plan_cache_hit({"fit_diagnostics": policy.fit_diagnostics}, receipt_gate)
    return cfg, factory, policy, dataset, ctx, surrogate, receipt_gate, reuse


def summarize7(rows):
    def col(k):
        return np.asarray([float(r[k]) for r in rows])
    r = col("operational_return")
    return {
        "return_mean": float(r.mean()),
        "return_sd": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "unsafe_fraction_mean": float(col("unsafe_fraction").mean()),
        "persistence_mean": float(col("persistence").mean()),
        "collapse_entry_mean": float(col("collapse_entry").mean()),
        "min_population_mean": float(col("min_true_state").mean()),
        "economic_cost_mean": float(col("economic_cost").mean()),
    }


def parity(replayed, accepted):
    fields = ["return_mean", "return_sd", "unsafe_fraction_mean", "persistence_mean",
              "collapse_entry_mean", "min_population_mean", "economic_cost_mean"]
    diffs = {f: abs(replayed[f] - accepted[f]) for f in fields}
    m = max(diffs.values())
    verdict = "PASS" if m <= 1e-9 else ("INVESTIGATE" if m <= 1e-6 else "FAIL")
    return {"verdict": verdict, "max_abs_diff": m,
            "per_field": {f: {"replay": replayed[f], "accepted": accepted[f], "abs_diff": diffs[f]} for f in fields}}


def recomputed_fit_count(fit_diagnostics):
    """Return the number of demographic fits performed instead of loaded."""
    diagnostics = dict(fit_diagnostics or {})
    if "fit_cache_misses" in diagnostics:
        return int(float(diagnostics["fit_cache_misses"]))
    if "fit_cache_hit" in diagnostics:
        return int(float(diagnostics["fit_cache_hit"]) != 1.0)
    raise RuntimeError("fit diagnostics do not expose cache hit/miss information")


def enforce_acceptance(parity_result, reuse_ok, recomputed_fits):
    """Raise a non-zero process exit unless every accepted-cell gate passes."""
    if (
        parity_result.get("verdict") != "PASS"
        or not reuse_ok
        or int(recomputed_fits) != 0
    ):
        raise SystemExit(
            "accepted-cell gate failed: "
            f"verdict={parity_result.get('verdict')} "
            f"reuse_ok={reuse_ok} recomputed_fits={recomputed_fits}"
        )


def run_cell(cell_key, side):
    global _LOG
    pop, env, sigma = CELLS[cell_key]
    method_full = METHOD_ID[side]
    cell_id = f"{cell_key}_{slug(pop)}_{env}_s{sig_slug(sigma)}"
    acc = accepted_fields(pop, env, sigma, method_full)
    cfg, factory, policy, dataset, ctx, surrogate, receipt_gate, reuse = build_policy(side, pop, env, sigma)
    assert (dataset.metadata.get("dataset_sha256") or "") == acc["dataset_sha256"], "dataset hash mismatch"

    _LOG = ReplayLogger(cfg, side, pop)
    _LOG.surrogate = surrogate; _LOG.pop_id = ctx.pop_id
    orig_act = policy.act
    def logged_act(belief, observation):
        a = orig_act(belief, observation); _LOG.on_act(policy, belief, observation, a); return a
    policy.act = logged_act
    instrumented = make_instrumented_class

    # PLUS: wrap each candidate planner's action_values so the frozen act()'s own per-candidate
    # scores are captured (no recompute). The wrapper calls the original unchanged and returns
    # its result unchanged (zero perturbation); it only stashes a reference for logging.
    orig_av = []
    if side == "plus":
        for pl in policy.planners:
            o = pl.action_values; orig_av.append((pl, o)); pl._replay_last_av = None
            def _mk(o, pl):
                def _w(belief):
                    r = o(belief); pl._replay_last_av = r; return r
                return _w
            pl.action_values = _mk(o, pl)

    def patched_make_env(env_cfg):
        e = real_make_env(env_cfg); e.__class__ = instrumented(type(e)); return e

    t0 = time.perf_counter()
    evaluator_mod.make_env = patched_make_env
    try:
        rows = ContinuousEvaluator(cfg, factory, "faithful_internal").run(policy)
    finally:
        evaluator_mod.make_env = real_make_env
        policy.act = orig_act
        for pl, o in orig_av:
            pl.action_values = o
    elapsed = time.perf_counter() - t0

    replayed = summarize7(rows)
    par = parity(replayed, acc)
    # acceptance checks
    n_steps = {r["n_steps"] for r in rows}
    assert n_steps == {50}, f"episode length != 50: {n_steps}"
    assert len(_LOG.rows) == len(SEEDS) * 50, (len(_LOG.rows), len(SEEDS) * 50)
    na = cfg.environment.num_actions
    assert all(0 <= r["action_t"] < na for r in _LOG.rows)
    reuse_ok = reuse.get("fit_cache_reuse_gate") == "passed"
    recomputed_fits = recomputed_fit_count(policy.fit_diagnostics)
    return dict(cell_id=cell_id, side=side, method_full=method_full, pop=pop, env=env, sigma=sigma,
                elapsed=elapsed, parity=par, replayed=replayed, accepted=acc,
                recomputed_fits=recomputed_fits, reuse_ok=reuse_ok, log=_LOG)


def write_outputs(res):
    cell_id, side = res["cell_id"], res["side"]
    label = f"{cell_id}__{res['method_full']}__diagnostic_replay"
    log = res["log"]; rows = log.rows
    # npz (array-heavy) + scalar csv.gz
    scal_cols = ["seed", "t", "x_true_t", "x_true_next", "obs_t", "action_t", "cost_t",
                 "utility_t", "penalty_flag_t", "reward_t", "r_setpoint_t", "r_pos_t",
                 "r_mort_t", "k_t", "m_t", "regime_z_t", "regime_switched_t",
                 "allee_C_episode", "theta_exponent_episode", "surrogate_reward_t",
                 "oracle_action", "true_margin_top1_top2"]
    if side == "moor":
        scal_cols += ["belief_mean", "belief_entropy", "argmax", "margin_top1_top2"]
    elif side == "plus":
        scal_cols += ["argmax_weighted", "argmax_map_only", "argmax_uniform",
                      "margin_top1_top2", "argmax_weighted_matches_deployed"]
    npz = {c: np.asarray([r[c] for r in rows]) for c in scal_cols}
    npz["x_next_all"] = np.vstack([r["x_next_all"] for r in rows])
    npz["true_reward_all_actions"] = np.vstack([r["true_reward_all_actions"] for r in rows])
    if side == "moor":
        npz["q"] = np.vstack([r["q"] for r in rows])
    elif side == "plus":
        npz["w"] = np.vstack([r["w"] for r in rows])                       # N x 8
        npz["q_cand"] = np.stack([r["q_cand"] for r in rows])              # N x 8 x 11
        npz["q_weighted"] = np.vstack([r["q_weighted"] for r in rows])     # N x 11
        npz["belief_mean"] = np.vstack([r["belief_mean"] for r in rows])   # N x 8
        npz["belief_entropy"] = np.vstack([r["belief_entropy"] for r in rows])
        npz["argmax_cand"] = np.vstack([r["argmax_cand"] for r in rows])   # N x 8
    (REPLAY / "logs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPLAY / "logs" / f"{label}.npz", **npz)
    with gzip.open(REPLAY / "logs" / f"{label}.scalars.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh); w.writerow(scal_cols)
        for r in rows:
            w.writerow([r[c] for c in scal_cols])
    return label


def plus_metrics(rows, by_seed):
    W = np.stack([r["w"] for r in rows])              # N x 8
    QC = np.stack([r["q_cand"] for r in rows])        # N x 8 x 11
    AW = np.array([r["argmax_weighted"] for r in rows])
    AM = np.array([r["argmax_map_only"] for r in rows])
    AU = np.array([r["argmax_uniform"] for r in rows])
    AC = np.stack([r["argmax_cand"] for r in rows])   # N x 8
    dep = np.array([r["action_t"] for r in rows])
    n8 = W.shape[1]; unif = 1.0 / n8
    H = -np.sum(W * np.log(W + 1e-300), axis=1)
    TV = 0.5 * np.sum(np.abs(W - unif), axis=1)
    l1_move, first_conf = [], []
    for _s, ep in by_seed.items():
        ep = sorted(ep, key=lambda r: r["t"])
        l1_move.append(float(np.sum(np.abs(np.asarray(ep[-1]["w"]) - np.asarray(ep[0]["w"])))))
        fc = -1
        for r in ep:
            if float(np.max(r["w"])) > 0.5:
                fc = int(r["t"]); break
        first_conf.append(fc)
    m1 = {"mean_entropy": float(H.mean()), "max_entropy_ln8": float(np.log(n8)),
          "mean_exp_entropy": float(np.mean(np.exp(H))), "mean_TV_from_uniform": float(TV.mean()),
          "mean_L1_move_w_last_minus_first": float(np.mean(l1_move)),
          "mean_first_step_max_w_gt_0p5": float(np.mean(first_conf)),
          "frac_episodes_ever_confident": float(np.mean([f >= 0 for f in first_conf]))}
    m2 = {"switch_vs_MAP": float(np.mean(AW != AM)), "switch_vs_uniform": float(np.mean(AW != AU)),
          "argmax_weighted_matches_deployed_frac": float(np.mean(AW == dep))}
    pair, unan, modal = [], [], []
    for i in range(len(rows)):
        a = AC[i]
        pair.append(float(np.mean([a[x] == a[y] for x in range(n8) for y in range(x + 1, n8)])))
        unan.append(float(len(set(a.tolist())) == 1))
        _v, c = np.unique(a, return_counts=True); modal.append(float(c.max()) / n8)
    m3 = {"mean_pairwise_agreement": float(np.mean(pair)), "frac_unanimous": float(np.mean(unan)),
          "mean_modal_action_share": float(np.mean(modal))}
    raw = np.mean(np.var(QC, axis=1), axis=1)
    cen = np.mean(np.var(QC - QC.mean(axis=2, keepdims=True), axis=1), axis=1)
    m5 = {"raw_disagreement_mean": float(raw.mean()), "centred_disagreement_mean": float(cen.mean()),
          "ratio_centred_over_raw": float(cen.mean() / raw.mean()) if raw.mean() > 0 else None}
    return {"M1_posterior_movement": m1, "M2_switch_fractions": m2,
            "M3_candidate_agreement": m3, "M5_disagreement": m5}


def derived_metrics(res):
    """Applicable M-metrics (MOOR: M4/M6/M7/M13/M15; PLUS: + M1/M2/M3/M5)."""
    log = res["log"]; rows = log.rows
    gamma = 0.95
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    # M6 reward decomposition per episode
    util_d, cost_d, pen_d, ret_d = [], [], [], []
    for s, ep in by_seed.items():
        ep = sorted(ep, key=lambda r: r["t"])
        g = np.array([gamma ** i for i in range(len(ep))])
        util_d.append(float(np.sum(g * np.array([r["utility_t"] for r in ep]))))
        cost_d.append(float(np.sum(g * np.array([r["cost_t"] for r in ep]))))
        pen_d.append(float(np.sum(g * 10.0 * np.array([r["penalty_flag_t"] for r in ep]))))
        ret_d.append(float(np.sum(g * np.array([r["reward_t"] for r in ep]))))
    ret_mean = float(np.mean(ret_d))
    m6 = {"return_reconstructed_mean": ret_mean,
          "abs_diff_vs_accepted": abs(ret_mean - res["accepted"]["return_mean"]),
          "discounted_utility_mean": float(np.mean(util_d)),
          "discounted_cost_mean": float(np.mean(cost_d)),
          "discounted_penalty_mean": float(np.mean(pen_d))}
    rpos0 = float(np.mean([r["r_pos_t"] == 0.0 for r in rows]))          # M7
    # M13 surrogate error along visited trajectory (MOOR consumes surrogate)
    sr = np.array([r["surrogate_reward_t"] for r in rows]); tr = np.array([r["reward_t"] for r in rows])
    pen = np.array([r["penalty_flag_t"] for r in rows]).astype(bool)
    def se(mask):
        e = (sr[mask] - tr[mask])
        return {"n": int(mask.sum()), "mean_signed_error": float(e.mean()) if mask.any() else None,
                "rmse": float(np.sqrt(np.mean(e ** 2))) if mask.any() else None}
    # surrogate top-1 vs true top-1 (per step, on the 11-action shadow)
    sur_top = []
    for r in rows:
        # surrogate reward per action would need 11 predicts/step; report deployed-vs-oracle instead
        sur_top.append(r["action_t"] == r["oracle_action"])
    m13 = {"overall": se(np.ones(len(rows), bool)), "safe": se(~pen), "unsafe": se(pen)}
    m15 = {"mean_true_margin_top1_top2": float(np.mean([r["true_margin_top1_top2"] for r in rows])),
           "frac_deployed_eq_myopic_oracle": float(np.mean(sur_top)),
           "frac_true_margin_below_1e-3": float(np.mean([r["true_margin_top1_top2"] < 1e-3 for r in rows]))}
    # M4 margin distribution (deployed q top1-top2)
    marg = np.array([r["margin_top1_top2"] for r in rows]) if res["side"] in ("moor", "plus") else None
    m4 = None if marg is None else {"min": float(marg.min()), "p5": float(np.percentile(marg, 5)),
                                    "median": float(np.median(marg)),
                                    "frac_below_1e-3": float(np.mean(marg < 1e-3)),
                                    "frac_below_1e-2": float(np.mean(marg < 1e-2))}
    out = {"M4_margin": m4, "M6_reward_decomposition": m6, "M7_frac_r_pos_zero": rpos0,
           "M8_depensation": "n/a (ricker)" if res["env"] == "ricker" else "TODO",
           "M9_regime": "n/a (ricker)" if res["env"] == "ricker" else "TODO",
           "M13_surrogate_error_visited": m13, "M15_state_discrimination": m15}
    if res["side"] == "plus":
        out.update(plus_metrics(rows, by_seed))
        out["note"] = "PLUS run: M1/M2/M3/M4/M5 + common M6/M7/M13/M15 included."
    else:
        out["note"] = "PLUS-only metrics M1/M2/M3/M5 are not applicable to MOOR."
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", nargs="?", choices=list(CELLS))
    ap.add_argument("side", nargs="?", choices=["moor", "plus"])
    ap.add_argument(
        "--gate-self-test-corrupt",
        action="store_true",
        help="exercise the enforced gate with a deliberately corrupted comparison",
    )
    args = ap.parse_args()
    require_scientific_tables()
    if args.gate_self_test_corrupt:
        accepted = {
            "return_mean": 1.0,
            "return_sd": 0.0,
            "unsafe_fraction_mean": 0.0,
            "persistence_mean": 1.0,
            "collapse_entry_mean": 0.0,
            "min_population_mean": 2.0,
            "economic_cost_mean": 0.0,
        }
        corrupted = dict(accepted)
        corrupted["return_mean"] += 1e-6
        result = parity(corrupted, accepted)
        print(json.dumps(result, indent=2))
        enforce_acceptance(result, reuse_ok=True, recomputed_fits=0)
        raise AssertionError("corrupted comparison unexpectedly passed")
    if args.cell is None or args.side is None:
        ap.error("cell and side are required unless --gate-self-test-corrupt is used")
    assert sha(ACCEPTED_CSV) == ACCEPTED_CSV_SHA, "accepted CSV hash changed!"
    res = run_cell(args.cell, args.side)
    label = write_outputs(res)
    dm = derived_metrics(res)
    (REPLAY / "derived").mkdir(parents=True, exist_ok=True)
    (REPLAY / "derived" / f"{label}.json").write_text(json.dumps(dm, indent=2))
    (REPLAY / "parity").mkdir(parents=True, exist_ok=True)
    prep = {"cell": args.cell, "side": args.side, "method": res["method_full"],
            "parity": res["parity"], "elapsed_seconds": res["elapsed"],
            "recomputed_fits": res["recomputed_fits"], "fit_cache_reuse": res["reuse_ok"]}
    (REPLAY / "parity" / f"PARITY_{label}.json").write_text(json.dumps(prep, indent=2))
    print(json.dumps({"cell": args.cell, "side": args.side, "parity": res["parity"]["verdict"],
                      "max_abs_diff": res["parity"]["max_abs_diff"], "elapsed_s": round(res["elapsed"], 1),
                      "recomputed_fits": res["recomputed_fits"], "reuse_ok": res["reuse_ok"]}, indent=2))
    # full parity detail
    print(json.dumps(res["parity"]["per_field"], indent=2))
    enforce_acceptance(res["parity"], res["reuse_ok"], res["recomputed_fits"])


if __name__ == "__main__":
    main()
