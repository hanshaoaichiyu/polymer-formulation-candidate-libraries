from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from common import INPUTS, RESULTS, WORK, atomic_json, load_npz, sha256, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def native(path: Path) -> Path:
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and path.is_absolute() else path


def file_hash(path: Path) -> str:
    return sha256(native(path))


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: object = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    audit = json.loads((WORK / "status" / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
    check("input audit passed", audit["ok"], audit)
    frozen_hash = sha256(WORK / "configs" / "FROZEN_PROTOCOL.json")
    all_input_hashes = []
    for replicate in range(1, 6):
        route = RESULTS / "formal" / f"replicate_{replicate}" / "MPG_GA"
        completion = json.loads((route / "COMPLETED.json").read_text(encoding="utf-8"))
        all_input_hashes.append(completion["input_sha256"])
        check(f"run {replicate} attempts", completion["attempts"] == 40000, completion["attempts"])
        check(f"run {replicate} seed", completion["seed"] == 3101000 + replicate * 100 + 3, completion["seed"])
        check(f"run {replicate} selected seeds", completion["selected_seeds"] == 208, completion["selected_seeds"])
        check(f"run {replicate} supports", completion["selected_structures"] == 52, completion["selected_structures"])
        check(f"run {replicate} no numeric anchor", not completion["numeric_database_anchor_used"] and not completion["nearest_row_repair_used"])
        check(f"run {replicate} protocol hash", completion["input_sha256"]["frozen_protocol"] == frozen_hash)
        ledger = read_csv(route / "selected_seed_ledger.csv")
        counts = {structure: 0 for structure in range(80)}
        for row in ledger:
            counts[int(row["structure"])] += 1
        nonzero = {key: value for key, value in counts.items() if value}
        check(f"run {replicate} four seeds per represented support", len(nonzero) == 52 and set(nonzero.values()) == {4}, nonzero)
        library = load_npz(route / "library.npz")
        portfolio = load_npz(route / "portfolio.npz")
        check(f"run {replicate} library size", len(library["design"]) == completion["library_size"])
        check(f"run {replicate} library source bits", np.all((library["source_bits"] & 8) != 0))
        check(f"run {replicate} portfolio", len(portfolio["design"]) == 260)
        check(f"run {replicate} no temporary files", not any(route.rglob("*.tmp")))
    check("formal runs share identical frozen inputs", all(item == all_input_hashes[0] for item in all_input_hashes))
    expected_rows = {
        "three_route_full_supply_by_run.csv": 20,
        "three_route_membership_by_run.csv": 35,
        "three_route_fixed_selection_by_run.csv": 40,
        "three_route_fixed_selection_by_support.csv": 1973,
        "three_route_five_run_summary.csv": 80,
        "mpg_ga_lineage_by_run.csv": 5,
    }
    for name, expected in expected_rows.items():
        actual = len(read_csv(RESULTS / name))
        check(f"row count {name}", actual == expected, actual)
    supply = read_csv(RESULTS / "three_route_full_supply_by_run.csv")
    for replicate in range(1, 6):
        rows = [row for row in supply if int(row["replicate"]) == replicate]
        tri = next(row for row in rows if row["arm"] == "TRI_GA")
        singles = [row for row in rows if row["arm"] != "TRI_GA"]
        check(f"run {replicate} union bounds", max(int(row["audited_unique_candidates"]) for row in singles) <= int(tri["audited_unique_candidates"]) <= sum(int(row["audited_unique_candidates"]) for row in singles))
        support_by_arm = {row["arm"]: int(row["supports"]) for row in rows}
        check(
            f"run {replicate} support coverage is internally valid",
            support_by_arm["MPG_GA"] == 52
            and support_by_arm["TRI_GA"] == 52
            and 1 <= support_by_arm["Generator_GA"] <= 52
            and 1 <= support_by_arm["R_GA"] <= 52,
            support_by_arm,
        )
    analysis_manifest = read_csv(WORK / "status" / "ANALYSIS_INPUT_MANIFEST.csv")
    hash_failures = []
    for row in analysis_manifest:
        path = Path(row["path"])
        if file_hash(path) != row["sha256"]:
            hash_failures.append(row["path"])
    check("analysis inputs unchanged", not hash_failures, hash_failures)
    check("report present", (WORK / "report" / "V35三种种子来源同一GA统一实验报告_20261104.md").is_file())
    ok = all(row["passed"] for row in checks)
    verification = {"ok": ok, "checks": len(checks), "failed": [row for row in checks if not row["passed"]]}
    atomic_json(WORK / "status" / "verification.json", verification)
    manifest_rows = []
    for path in sorted(WORK.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or ".git" in path.parts or path == WORK / "status" / "DELIVERY_MANIFEST.csv":
            continue
        manifest_rows.append({"relative_path": path.relative_to(WORK).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(WORK / "status" / "DELIVERY_MANIFEST.csv", manifest_rows)
    completion = {
        "ok": ok,
        "formal_runs": 5,
        "total_ga_attempts": 200000,
        "manifest_files": len(manifest_rows),
        "verification_checks": len(checks),
        "frozen_protocol_sha256": frozen_hash,
        "report": "report/V35三种种子来源同一GA统一实验报告_20261104.md",
    }
    atomic_json(WORK / "status" / "COMPLETED.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("delivery verification failed")


if __name__ == "__main__":
    main()
