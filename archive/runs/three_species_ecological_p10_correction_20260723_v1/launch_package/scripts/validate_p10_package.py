#!/usr/bin/env python3
"""Pre-submission validation for the P=10 correction package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
RUN_ROOT = PACKAGE.parent
WORKSPACE = RUN_ROOT.parents[1]
PYTHON = WORKSPACE / ".venv-paper-faithful/bin/python"
DISPATCH = PACKAGE / "scripts/dispatch_p10_plan.py"
ACCEPTANCE = PACKAGE / "scripts/accept_p10_method.py"
PLUS_CODE = (
    WORKSPACE
    / "real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix"
)
MOOR_CODE = WORKSPACE / "real_ecology_runs/adapted_32cell_diagnostic_pending/code"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dynamic_cache_check(group: str) -> dict[str, object]:
    code_root = PLUS_CODE if group == "plus" else MOOR_CODE
    manifest = PACKAGE / f"manifests/{group}_p10_plan_24.csv"
    config = PACKAGE / (
        "configs/plus_ricker_only_p10.yaml"
        if group == "plus"
        else "configs/moor_ricker_p10.yaml"
    )
    output_root = RUN_ROOT / group
    snippet = r'''
import csv, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); manifest=pathlib.Path(sys.argv[2])
config=pathlib.Path(sys.argv[3]); output=pathlib.Path(sys.argv[4]); group=sys.argv[5]
sys.path.insert(0,str(root/"src")); sys.path.insert(0,str(root/"scripts"))
from real_ecology_benchmark.config import load_config
from real_ecology_benchmark.pipeline import ensure_dataset,_load_or_fit_public_surrogate,_hidden_method_context
from real_ecology_benchmark.faithful_fit import fit_cache_key,split_history_episodes
from run_real_manifest_row import apply_row_config
items=[]
with manifest.open(newline="",encoding="utf-8") as h: manifest_rows=list(csv.DictReader(h))
for position,row in enumerate(manifest_rows):
    cfg=load_config(config); cfg,cell,_=apply_row_config(cfg,row,output)
    if cfg.environment.reward_mode!="safe" or cfg.environment.collapse_penalty!=10.0:
        raise SystemExit("resolved configuration is not safe P=10")
    if cfg.evaluation.horizon!=50 or cfg.evaluation.discount!=0.95:
        raise SystemExit("resolved evaluation protocol mismatch")
    dataset=ensure_dataset(cfg,regenerate=False)
    surrogate,_path,status=_load_or_fit_public_surrogate(cfg,dataset)
    if status!="loaded": raise SystemExit("surrogate was not a verified load")
    context=_hidden_method_context(cfg,dataset,surrogate)
    if group=="plus":
        from run_ricker_only_fit_locked import candidate_lock_specs
        specs=candidate_lock_specs(cfg,row,dataset,config)
        keys=[spec["cache_key"] for spec in specs]
    else:
        seed=cfg.seed+47000
        history,holdout=split_history_episodes(dataset,cfg.faithful.fit.history_fraction,seed)
        keys=[fit_cache_key(dataset,context,"ricker",cfg.faithful.model,cfg.faithful.fit,
             seed,"moor_ricker_00",history,holdout,0.90)]
    receipt=json.loads((output/row["fit_receipt_path"]).read_text())
    if keys!=receipt["fit_cache_keys"]: raise SystemExit(f"cache key mismatch row {position}")
    for key in keys:
        if not (output/"fit_cache"/f"{key}.json").is_file(): raise SystemExit("cache metadata missing")
        if not (output/"fit_cache"/f"{key}.npz").is_file(): raise SystemExit("cache array missing")
    items.append({"position":position,"keys":keys,"transition_data_hash":receipt["transition_data_hash"]})
print(json.dumps({"group":group,"rows":len(items),"candidate_slots":sum(len(x["keys"]) for x in items)}))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root / "src")
    result = subprocess.run(
        [
            str(PYTHON),
            "-c",
            snippet,
            str(code_root),
            str(manifest),
            str(config),
            str(output_root),
            group,
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return json.loads(result.stdout)


def main() -> None:
    preparation = json.loads(
        (PACKAGE / "PREPARATION_RECEIPT.json").read_text(encoding="utf-8")
    )
    if preparation["new_outcomes_opened"] is not False:
        raise RuntimeError("preparation receipt is not return-blind")
    dry_runs = {}
    output_dirs = set()
    dependencies = set()
    for group in ("plus", "moor"):
        result = subprocess.run(
            [str(PYTHON), str(DISPATCH), "--method", group, "--dry-run"],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)
        if payload["dry_run_count"] != 24 or payload["return_fields_opened"] is not False:
            raise RuntimeError(f"dry-run coverage mismatch: {group}")
        for command in payload["commands"]:
            if command["output_directory"] in output_dirs:
                raise RuntimeError("output directory collision")
            output_dirs.add(command["output_directory"])
            dependency = (group, command["dependency_key"])
            if dependency in dependencies:
                raise RuntimeError("dependency key collision within method")
            dependencies.add(dependency)
            if command["collapse_penalty"] != 10.0:
                raise RuntimeError("dry run resolved a non-P=10 row")
        dry_runs[group] = payload
        (PACKAGE / f"DRY_RUN_{group.upper()}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    dynamic = {group: dynamic_cache_check(group) for group in ("plus", "moor")}
    if dynamic["plus"] != {"group": "plus", "rows": 24, "candidate_slots": 192}:
        raise RuntimeError("PLUS dynamic cache coverage mismatch")
    if dynamic["moor"] != {"group": "moor", "rows": 24, "candidate_slots": 24}:
        raise RuntimeError("MOOR dynamic cache coverage mismatch")
    ledger = PACKAGE / "provenance/fit_reuse_ledger_216.csv"
    ledger_rows = rows(ledger)
    if len(ledger_rows) != 216:
        raise RuntimeError("fit-reuse ledger does not contain 216 candidate slots")
    if any(row["reuse_status"] != "verified_reward_independent_reuse" for row in ledger_rows):
        raise RuntimeError("fit-reuse ledger contains an unverified row")
    # Prove the acceptance reader rejects both forbidden filenames and fields.
    spec = importlib.util.spec_from_file_location("p10_acceptance", ACCEPTANCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sentinel_results = []
    with tempfile.TemporaryDirectory(prefix="p10-acceptance-sentinel-") as folder:
        root = Path(folder)
        forbidden_name = root / "summary.json"
        forbidden_name.write_text('{"status":"complete"}\n', encoding="utf-8")
        forbidden_field = root / "structural.json"
        forbidden_field.write_text('{"operational_return":12345}\n', encoding="utf-8")
        for path in (forbidden_name, forbidden_field):
            try:
                module.guarded_json(path)
            except RuntimeError:
                sentinel_results.append(True)
            else:
                sentinel_results.append(False)
    if sentinel_results != [True, True]:
        raise RuntimeError("return-blind acceptance sentinel test failed")
    receipt = {
        "validation_schema": "three_species_p10_pre_submission_v1",
        "decision": "PASS",
        "plan_rows": {"plus": 24, "moor": 24},
        "planners_and_evaluations": 48,
        "fit_reuse_candidate_slots": 216,
        "fits_recomputed": 0,
        "dynamic_cache_identity": dynamic,
        "unique_output_directories": len(output_dirs),
        "unique_method_dependency_keys": len(dependencies),
        "acceptance_forbidden_sentinel_passed": True,
        "return_fields_opened": False,
        "hashes": {
            "plus_manifest": sha256(PACKAGE / "manifests/plus_p10_plan_24.csv"),
            "moor_manifest": sha256(PACKAGE / "manifests/moor_p10_plan_24.csv"),
            "plus_config": sha256(PACKAGE / "configs/plus_ricker_only_p10.yaml"),
            "moor_config": sha256(PACKAGE / "configs/moor_ricker_p10.yaml"),
            "fit_reuse_ledger": sha256(ledger),
            "dry_run_plus": sha256(PACKAGE / "DRY_RUN_PLUS.json"),
            "dry_run_moor": sha256(PACKAGE / "DRY_RUN_MOOR.json"),
        },
    }
    target = PACKAGE / "VALIDATION_RECEIPT.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
