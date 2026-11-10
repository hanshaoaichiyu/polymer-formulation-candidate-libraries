from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common import CASES, RESULTS, WORK, atomic_json, load_npz, sha256, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    checks = []
    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    input_audit = json.loads((WORK / "status" / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
    check("input audit", input_audit["ok"], input_audit)
    protocol_hash = sha256(WORK / "configs" / "FROZEN_PROTOCOL.json")
    for case in CASES:
        expected_seeds = 16 if case == "natural_fiber_pp_hdpe" else 4
        expected_structures = 4 if case == "natural_fiber_pp_hdpe" else 1
        expected_portfolio = 32 if case == "natural_fiber_pp_hdpe" else 40
        for replicate in range(1, 6):
            root = RESULTS / "formal" / case / f"replicate_{replicate}" / "MPG_GA"
            completion = json.loads((root / "COMPLETED.json").read_text(encoding="utf-8"))
            check(f"{case} run {replicate} complete formal", not completion["pilot"])
            check(f"{case} run {replicate} 8000 starts", completion["starts"] == 8000)
            check(f"{case} run {replicate} 40000 GA attempts", completion["ga"]["attempts"] == 40000)
            check(f"{case} run {replicate} selected seeds", completion["selected_seeds"] == expected_seeds)
            check(f"{case} run {replicate} selected structures", completion["selected_structures"] == expected_structures)
            check(f"{case} run {replicate} monotone", completion["monotonic_violations"] == 0)
            check(f"{case} run {replicate} no anchors", not completion["numeric_database_anchor_used"] and not completion["nearest_row_repair_used"])
            check(f"{case} run {replicate} protocol hash", completion["protocol_sha256"] == protocol_hash)
            pool = load_npz(root / "representative_pool.npz"); library = load_npz(root / "library.npz"); portfolio = load_npz(root / "portfolio.npz")
            check(f"{case} run {replicate} representative identity", len(pool["design"]) == completion["successful_starts"] and np.all(pool["feasible"]))
            check(f"{case} run {replicate} library size", len(library["design"]) == completion["library_size"])
            check(f"{case} run {replicate} portfolio size", len(portfolio["design"]) == expected_portfolio)
            check(f"{case} run {replicate} source bits", np.all((library["source_bits"] & 8) != 0))
            check(f"{case} run {replicate} no temporary files", not any(root.rglob("*.tmp")))
    expected_rows = {
        "two_case_full_supply_by_run.csv": 40,
        "two_case_membership_by_run.csv": 70,
        "two_case_fixed_selection_by_run.csv": 80,
        "two_case_fixed_selection_by_support.csv": 200,
        "two_case_five_run_summary.csv": 192,
        "two_case_MPG_GA_resources_by_run.csv": 10,
    }
    for name, expected in expected_rows.items():
        actual = len(read_csv(RESULTS / name)); check(f"row count {name}", actual == expected, actual)
    supply = read_csv(RESULTS / "two_case_full_supply_by_run.csv")
    for case in CASES:
        for replicate in range(1, 6):
            rows = [row for row in supply if row["case"] == case and int(row["replicate"]) == replicate]
            tri = next(row for row in rows if row["arm"] == "TRI_GA"); singles = [row for row in rows if row["arm"] != "TRI_GA"]
            check(f"{case} run {replicate} union bounds",
                  max(int(row["audited_unique_candidates"]) for row in singles) <= int(tri["audited_unique_candidates"]) <= sum(int(row["audited_unique_candidates"]) for row in singles))
    manifest = read_csv(WORK / "status" / "ANALYSIS_INPUT_MANIFEST.csv"); changed = []
    for row in manifest:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            changed.append(row["path"])
    check("analysis inputs unchanged", not changed, changed)
    check("report present", (WORK / "report" / "V38两个补充聚合物案例MPG_GA统一实验报告_20261105.md").is_file())
    ok = all(row["passed"] for row in checks)
    verification = {"ok": ok, "checks": len(checks), "failed": [row for row in checks if not row["passed"]]}
    atomic_json(WORK / "status" / "verification.json", verification)
    completion = {"ok": ok, "formal_case_runs": 10, "total_GA_attempts": 400000,
                  "verification_checks": len(checks),
                  "protocol_sha256": protocol_hash,
                  "report": "report/V38两个补充聚合物案例MPG_GA统一实验报告_20261105.md"}
    atomic_json(WORK / "status" / "COMPLETED.json", completion)
    manifest_rows = []
    manifest_path = WORK / "status" / "DELIVERY_MANIFEST.csv"
    for path in sorted(WORK.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or ".git" in path.parts or path == manifest_path:
            continue
        manifest_rows.append({"relative_path": path.relative_to(WORK).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(manifest_path, manifest_rows)
    print(json.dumps({**completion, "manifest_files": len(manifest_rows)}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
