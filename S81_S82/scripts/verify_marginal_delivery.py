from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


WORK = Path(__file__).resolve().parents[1]
MARGINAL = WORK / "results" / "marginal"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    checks = []
    def check(name, value):
        checks.append({"check": name, "ok": bool(value)})
    for replicate in range(1, 6):
        formal = MARGINAL / "formal" / f"replicate_{replicate}"
        receipt = json.loads((formal / "COMPLETED.json").read_text(encoding="utf-8"))
        check(f"run {replicate} endpoint identity", receipt["reconstruction"]["exact_projected_endpoint_identity"])
        check(f"run {replicate} endpoint difference zero", receipt["reconstruction"]["max_abs_projected_endpoint_difference"] == 0)
        for name in ("A2_continued_MPGA_visits.npz", "A3_new_start_MPGA_visits.npz", "B2_additional_generator_candidates.npz"):
            check(f"run {replicate} {name}", (formal / name).is_file())
        cache = json.loads((MARGINAL / "state_cache" / f"replicate_{replicate}" / "CACHE_RECEIPT.json").read_text(encoding="utf-8"))
        check(f"run {replicate} cache verified", cache["verified"])
        check(f"run {replicate} cost supplement", (MARGINAL / "cost_supplement" / f"replicate_{replicate}" / "COMPLETED.json").is_file())
    expected = {
        "supply_and_marginal_by_run.csv": 60,
        "fixed_selection_by_run.csv": 60,
        "work_fraction_curves_by_run.csv": 135,
        "cost_and_budget_match_by_run.csv": 30,
        "paired_differences_by_run.csv": 20,
        "paired_differences_five_run_summary.csv": 32,
        "five_run_summary.csv": 156,
    }
    for name, count in expected.items():
        path = MARGINAL / "analysis" / name
        check(f"analysis {name} exists", path.is_file())
        check(f"analysis {name} rows", len(rows(path)) == count)
    cost = rows(MARGINAL / "analysis" / "cost_and_budget_match_by_run.csv")
    six = [float(r["ratio_to_GA"]) for r in cost if r["arm"] == "B2_more_generator_6_batches"]
    check("B2 six batches budget within all runs", len(six) == 5 and all(value <= 1 for value in six))
    check("report exists", (WORK / "report" / "工业PP等预算边际价值对照实验报告_20261106.md").is_file())
    files = sorted([path for path in [WORK / "configs" / "MARGINAL_GAIN_PROTOCOL.json",
                                      WORK / "configs" / "CALIBRATED_BUDGET.json",
                                      WORK / "configs" / "COST_ONLY_AMENDMENT_BEFORE_OUTCOME_ANALYSIS.json",
                                      WORK / "scripts" / "run_marginal_contrasts.py",
                                      WORK / "scripts" / "run_cost_supplement.py",
                                      WORK / "scripts" / "analyze_marginal_contrasts.py",
                                      WORK / "report" / "工业PP等预算边际价值对照实验报告_20261106.md"]
                    + list((MARGINAL / "analysis").glob("*.csv"))])
    manifest = [{"path": str(path.relative_to(WORK)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
    status = {"ok": all(item["ok"] for item in checks), "checks": len(checks),
              "failed": [item for item in checks if not item["ok"]], "manifest_files": len(manifest),
              "formal_replicates": 5, "endpoint_identity_verified": 5,
              "manuscript_or_figure_modified": False}
    (WORK / "status").mkdir(exist_ok=True)
    with (WORK / "status" / "MARGINAL_INPUT_OUTPUT_MANIFEST.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"]); writer.writeheader(); writer.writerows(manifest)
    (WORK / "status" / "MARGINAL_COMPLETED.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
