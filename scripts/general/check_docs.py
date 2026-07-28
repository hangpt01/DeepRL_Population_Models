"""Lightweight checks for the canonical benchmark documentation bundle."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "docs" / "benchmark"
FOOTER_RE = re.compile(r"^Verified against: .+ @ [0-9a-f]{7,40}\s*$", re.MULTILINE)
BANNED = {
    "Tier-2/3/4 label": re.compile(r"Tier-?[234]"),
    "claude_build": re.compile(r"claude_build"),
    "discrete_action_cont_obser": re.compile(r"discrete_action_cont_obser"),
    "standalone repo": re.compile(r"standalone repo"),
}
REAL_SETPOINT = re.compile(r"real_setpoint")
MD_LINK = re.compile(r"\[[^\]]+\]\(([^)#][^)]+)\)")
TEX_HREF = re.compile(r"\\href\{([^}]+)\}")


def fail(message: str) -> None:
    print(f"docs-check: {message}", file=sys.stderr)
    raise SystemExit(1)


def bundle_docs() -> list[Path]:
    expected = [
        "README.md",
        "01_problem_setting_and_design.tex",
        "02_real_ecology_data_actions_and_costs.tex",
        "03_state_action_reward_reference.tex",
        "04_algorithm_adaptations_and_claims.tex",
        "05_implementation_and_code_map.md",
        "06_experiment_protocol_and_reproducibility.md",
        "07_recent_real_ecology_results.md",
        "08_limitations_tracker.md",
    ]
    paths = [BUNDLE / name for name in expected]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        fail(f"missing bundle documents: {missing}")
    return paths


def check_footers(paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if not FOOTER_RE.search(text):
            fail(f"missing Verified against footer in {path.relative_to(ROOT)}")


def check_banned_terms(paths: list[Path]) -> None:
    hits: list[str] = []
    real_setpoint_hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for label, pattern in BANNED.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {label}")
        for match in REAL_SETPOINT.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            line = text.splitlines()[line_no - 1]
            real_setpoint_hits.append(f"{path.relative_to(ROOT)}:{line_no}: {line}")
    if hits:
        fail("banned terms found:\n" + "\n".join(hits))
    if len(real_setpoint_hits) != 1:
        fail(
            "expected exactly one real_setpoint legacy-alias sentence; found "
            f"{len(real_setpoint_hits)}:\n" + "\n".join(real_setpoint_hits)
        )
    allowed = real_setpoint_hits[0].lower()
    if "legacy" not in allowed or "alias" not in allowed or "setpoint_cumulative" not in allowed:
        fail("the real_setpoint occurrence is not the allowed legacy-alias sentence")


def is_external(target: str) -> bool:
    return (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
        or target.startswith("#")
    )


def check_links(paths: list[Path]) -> None:
    broken: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        targets = [m.group(1) for m in MD_LINK.finditer(text)]
        targets.extend(m.group(1) for m in TEX_HREF.finditer(text))
        for target in targets:
            target = target.split("#", 1)[0]
            if not target or is_external(target):
                continue
            candidate = (path.parent / target).resolve()
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError:
                broken.append(f"{path.relative_to(ROOT)} -> {target} leaves repo")
                continue
            if not candidate.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    if broken:
        fail("broken relative links:\n" + "\n".join(broken))


def check_code_constants() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from real_ecology_benchmark import realdata  # noqa: PLC0415
    from real_ecology_benchmark.methods import (  # noqa: PLC0415
        FAITHFUL_METHODS,
        METHODS,
        NATIVE_METHODS,
    )

    if realdata.NUM_REAL_ACTIONS != 11:
        fail("NUM_REAL_ACTIONS no longer equals 11")
    if realdata.NUM_REAL_POPULATIONS != 9:
        fail("NUM_REAL_POPULATIONS no longer equals 9")
    # The docs bundle describes the seven benchmark methods.  The native ecological
    # solvers are a documented addition for the motivation experiment, so assert the
    # documented seven by name rather than pinning the registry size.
    documented = {
        "mopo", "refplan", "bamcts", "moor", "plus",
        "ensemble_value_disagreement_pessimism", "ogsrl",
    }
    missing = documented - set(METHODS)
    if missing:
        fail(f"method registry is missing documented methods: {sorted(missing)}")
    deprecated_aliases = {"delphic"}
    undocumented = set(METHODS) - documented - set(NATIVE_METHODS) - deprecated_aliases
    documented_adapted = {
        "plus_adapted_mechanistic_pbvi",
        "moor_adapted_ricker_misspec_pbvi",
    }
    if set(FAITHFUL_METHODS) != documented_adapted:
        fail(
            "adapted method registry differs from documented IDs: "
            f"registry={sorted(FAITHFUL_METHODS)}, docs={sorted(documented_adapted)}"
        )
    undocumented -= documented_adapted
    if undocumented:
        fail(f"method registry has undocumented methods: {sorted(undocumented)}")


def main() -> int:
    paths = bundle_docs()
    check_footers(paths)
    check_banned_terms(paths)
    check_links(paths)
    check_code_constants()
    print("docs-check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
