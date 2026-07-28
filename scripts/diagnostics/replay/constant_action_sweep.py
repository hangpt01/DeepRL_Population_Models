#!/usr/bin/env python3
"""Stage S1: full constant-action sweep. Evaluator-only REFERENCE policies (no fitting,
no planning, no method code). 11 constant actions x 24 cells x 20 registered seeds.

Metrics replicate the frozen ContinuousEvaluator exactly (unsafe over the 51-state trace,
persistence = final>s_safe, collapse_entry from-above, economic_cost undiscounted). Also the
M6 decomposition (discounted utility/cost/penalty) and per-cell headroom vs the best accepted
learned method. Constant policies are reference-only; they are never inserted into the
six-method ranking tables.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import (  # noqa: E402
    ACCEPTED_CSV,
    ECOLOGICAL_SCRIPTS,
    ECOLOGICAL_SRC,
    P10_DATA_ROOT,
    P10_PACKAGE,
    REPLAY_OUTPUT,
    require_scientific_tables,
)

REPLAY = REPLAY_OUTPUT
sys.path[:0] = [str(ECOLOGICAL_SCRIPTS), str(ECOLOGICAL_SRC)]
import run_real_manifest_row as rrmr
from real_ecology_benchmark.config import load_config
from real_ecology_benchmark.envs import make_env

SEEDS = [b + i for b in (7001, 7051, 7101, 7151, 7201) for i in range(4)]
GAMMA = 0.95
MLABEL = {"refplan": "RefPlan", "ogsrl": "OGSRL", "bamcts": "BA-MCTS",
          "ensemble_value_disagreement_pessimism": "EVD",
          "plus_adapted_ricker_only_pbvi": "PLUS", "moor_adapted_ricker_misspec_pbvi": "MOOR"}


def cells_and_cfgs():
    rows = list(csv.DictReader(open(P10_PACKAGE / "manifests" / "moor_p10_plan_24.csv")))
    out = []
    for r in rows:
        cfg = load_config(str(P10_PACKAGE / "configs" / "moor_ricker_p10.yaml"))
        cfg, _c, _e = rrmr.apply_row_config(cfg, r, P10_DATA_ROOT / "moor")
        cfg.validate()
        out.append((r["population"], r["environment"], r["sigma_obs"], cfg.environment))
    return out


def roll(env_cfg, action):
    s_safe = float(env_cfg.safety_threshold); k_ref = float(env_cfg.K_ref)
    R = dict(op=[], unsafe=[], persist=[], collapse=[], minp=[], econ=[], u=[], c=[], p=[])
    for seed in SEEDS:
        e = make_env(env_cfg); reset = e.reset(seed)
        s0 = float(reset.evaluator_info["state"]); tv = [s0]
        unsafe = int(s0 <= s_safe); entries = 0; econ = 0.0
        g = 1.0; op = u = c = p = 0.0
        for _t in range(50):
            res = e.step(action); info = res.evaluator_info; st = float(info["state"])
            op += g * float(res.reward)
            u += g * (st / (st + k_ref)); c += g * float(e.actions[action].cost)
            p += g * 10.0 * float(bool(info["safety_penalty_applied"]))
            entries += int(bool(info["entered_safety_region"]))
            econ += float(e.actions[action].cost); tv.append(st); unsafe += int(st <= s_safe)
            g *= GAMMA
        R["op"].append(op); R["unsafe"].append(unsafe / 51.0)
        R["persist"].append(int(tv[-1] > s_safe)); R["collapse"].append(int(entries > 0))
        R["minp"].append(min(tv)); R["econ"].append(econ)
        R["u"].append(u); R["c"].append(c); R["p"].append(p)
    m = {k: float(np.mean(v)) for k, v in R.items()}
    m["op_sd"] = float(np.std(R["op"], ddof=1)) if len(set(R["op"])) > 1 else 0.0
    return m


def best_learned():
    """Per (pop,env,sigma): best accepted learned return + method; and PLUS/MOOR modal action."""
    best = {}
    for r in csv.DictReader(open(ACCEPTED_CSV)):
        k = (r["population"], r["environment"], r["sigma_obs"])
        v = float(r["operational_return_mean"])
        rec = best.setdefault(k, {"v": -1e18, "m": None, "modal": {}})
        if v > rec["v"]:
            rec["v"] = v; rec["m"] = MLABEL[r["method"]]
        if r["method"] in ("plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi") and r["pbvi_argmax_actions"]:
            acts = [int(x) for x in r["pbvi_argmax_actions"].split(";") if x != ""]
            vals, cnts = np.unique(acts, return_counts=True)
            rec["modal"][MLABEL[r["method"]]] = int(vals[cnts.argmax()])
    return best


def dominated_table(species):
    """Analytic: within a (r_setpoint, dN) group with r_setpoint<=0, dK is inert => lowest-cost
    member dominates. Uses ricker column (ricker/allee/regime); theta uses lgm (same pattern)."""
    rows = [r for r in csv.DictReader(open(require_scientific_tables() / "action_effects_long.csv"))
            if r["common_name"] == species]
    groups = {}
    for r in rows:
        key = (round(float(r["r_setpoint_ricker"]), 6), round(float(r["dN"]), 6))
        groups.setdefault(key, []).append((r["action"], float(r["dK_step"]), float(r["cost_step"])))
    dominated = {}
    for (rset, dN), members in groups.items():
        if rset <= 0 and len(members) > 1:
            dom = min(members, key=lambda m: m[2])  # lowest cost
            for a, dK, cost in members:
                if a != dom[0]:
                    dominated[a] = {"dominated_by": dom[0], "r_setpoint": rset, "dN": dN,
                                    "cost": cost, "dominant_cost": dom[2]}
    return dominated


def main():
    require_scientific_tables()
    bl = best_learned()
    out_rows = []; per_cell = []
    for pop, env, sig, ec in cells_and_cfgs():
        cellres = {}
        for a in range(11):
            m = roll(ec, a)
            cellres[a] = m
            out_rows.append({"population": pop, "environment": env, "sigma_obs": sig, "action": a,
                             "return_mean": m["op"], "return_sd": m["op_sd"],
                             "unsafe_fraction": m["unsafe"], "persistence": m["persist"],
                             "collapse_entry": m["collapse"], "min_population": m["minp"],
                             "economic_cost": m["econ"], "disc_utility": m["u"],
                             "disc_cost": m["c"], "disc_penalty": m["p"]})
        ba = max(cellres, key=lambda a: cellres[a]["op"]); bv = cellres[ba]["op"]
        blr = bl[(pop, env, sig)]
        headroom = blr["v"] - bv
        per_cell.append({"population": pop, "environment": env, "sigma_obs": sig,
                         "best_constant_action": ba, "best_constant_return": bv,
                         "best_learned_method": blr["m"], "best_learned_return": blr["v"],
                         "headroom": headroom, "FLAG_headroom_le_0": bool(headroom <= 0),
                         "plus_modal": blr["modal"].get("PLUS"), "moor_modal": blr["modal"].get("MOOR")})
    # write sweep CSV
    (REPLAY / "derived").mkdir(exist_ok=True)
    with open(REPLAY / "derived" / "S1_constant_action_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    # dominated tables + accepted-method-deploys-dominated check
    dom = {sp: dominated_table(sp) for sp in ("Amur tiger", "Crab-eating fox", "Egyptian vulture")}
    for pc in per_cell:
        sp = pc["population"]; ds = {int(a[1:]) for a in dom[sp]}
        pc["plus_modal_is_dominated"] = (pc["plus_modal"] in ds) if pc["plus_modal"] is not None else None
        pc["moor_modal_is_dominated"] = (pc["moor_modal"] in ds) if pc["moor_modal"] is not None else None
    summary = {"schema": "S1_constant_action_sweep_v1", "reference_policies_only": True,
               "n_cell_action_rows": len(out_rows), "per_cell": per_cell,
               "dominated_actions": {sp: dom[sp] for sp in dom}}
    (REPLAY / "derived" / "S1_constant_action_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    # console report
    print(f"sweep rows: {len(out_rows)} (11 actions x 24 cells)\n")
    print("=== per-cell: best constant vs best learned (FLAG = constant >= learned) ===")
    for pc in sorted(per_cell, key=lambda x: (x["population"], x["environment"], x["sigma_obs"])):
        flag = "  <<< FLAG headroom<=0" if pc["FLAG_headroom_le_0"] else ""
        print(f"{pc['population']:16} {pc['environment']:6} s{pc['sigma_obs']}: "
              f"best_const a{pc['best_constant_action']}={pc['best_constant_return']:.3f} | "
              f"best_learned {pc['best_learned_method']}={pc['best_learned_return']:.3f} | "
              f"headroom={pc['headroom']:+.3f}{flag}")
    n_flag = sum(pc["FLAG_headroom_le_0"] for pc in per_cell)
    print(f"\ncells where a constant reference matches/beats the best learned method: {n_flag}/24")
    print("\n=== dominated-action table (analytic) ===")
    for sp in dom:
        print(f"{sp}: dominated = {sorted(dom[sp], key=lambda a:int(a[1:]))}")
    print("\n=== accepted PLUS/MOOR modal action vs dominated set ===")
    for pc in sorted(per_cell, key=lambda x: (x["population"], x["environment"], x["sigma_obs"])):
        if pc["plus_modal_is_dominated"] or pc["moor_modal_is_dominated"]:
            print(f"{pc['population']:16} {pc['environment']:6} s{pc['sigma_obs']}: "
                  f"PLUS a{pc['plus_modal']} dominated={pc['plus_modal_is_dominated']} | "
                  f"MOOR a{pc['moor_modal']} dominated={pc['moor_modal_is_dominated']}")


if __name__ == "__main__":
    main()
