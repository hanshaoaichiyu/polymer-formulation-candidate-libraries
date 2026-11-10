from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import HISTORICAL, INPUTS, WORK, Problem, atomic_json, load_npz, sha256, write_csv


def file_record(path: Path, role: str) -> dict:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    problem = Problem()
    manifest = [
        file_record(WORK / "configs" / "AUDIT_PLAN.json", "predeclared audit plan"),
        file_record(WORK / "configs" / "FROZEN_PROTOCOL.json", "frozen protocol"),
        file_record(INPUTS / "generation_protocol_v1.json", "frozen generation protocol"),
        file_record(INPUTS / "industrial_pp_100_recipes.csv", "frozen industrial evaluator data"),
        file_record(INPUTS / "best_feasible_per_successful_start.npz", "MPG best-feasible representatives"),
    ]
    comparison_rows = []
    maximum_normalized = 0.0
    maximum_quality = 0.0
    total_window_mismatch = 0
    for replicate in range(1, 6):
        for route in ("G_GA", "R_GA"):
            path = HISTORICAL / f"replicate_{replicate}" / "methods" / route / "library.npz"
            manifest.append(file_record(path, f"historical {route} library run {replicate}"))
            saved = load_npz(path)
            replay = problem.audit32(saved["design"], saved["structure"])
            width = np.ptp(problem.windows, axis=1)
            normalized = np.abs(replay["predicted_properties"].astype(float) - saved["predicted_properties"].astype(float)) / width
            pred_diff = float(np.max(normalized))
            q_diff = float(np.max(np.abs(replay["task_quality"].astype(float) - saved["task_quality"].astype(float))))
            saved_window = np.all(
                (saved["predicted_properties"] >= problem.windows[:, 0])
                & (saved["predicted_properties"] <= problem.windows[:, 1]),
                axis=1,
            )
            mismatch = int(np.sum(replay["window_ok"] != saved_window))
            structural_fail = int(np.sum(~replay["structural_ok"]))
            comparison_rows.append(
                {
                    "replicate": replicate,
                    "route": route,
                    "rows": len(saved["design"]),
                    "max_window_normalized_prediction_difference": pred_diff,
                    "max_quality_difference": q_diff,
                    "window_flag_mismatches": mismatch,
                    "structural_failures": structural_fail,
                }
            )
            maximum_normalized = max(maximum_normalized, pred_diff)
            maximum_quality = max(maximum_quality, q_diff)
            total_window_mismatch += mismatch

    compact = load_npz(INPUTS / "best_feasible_per_successful_start.npz")
    representative_rows = []
    representative_mismatch = 0
    representative_boundary_max = 0.0
    for replicate in range(1, 6):
        rows = compact["replicate"] == replicate
        design = np.rint(compact["design"][rows].astype(float) / problem.step) * problem.step
        replay = problem.audit32(design, compact["structure"][rows])
        mismatch = int(np.sum(~replay["feasible"]))
        representative_mismatch += mismatch
        width = np.ptp(problem.windows, axis=1)
        representative_boundary_max = max(
            representative_boundary_max,
            float(np.max(np.abs(replay["predicted_properties"].astype(float) - compact["predicted_properties"][rows].astype(float)) / width)),
        )
        representative_rows.append(
            {
                "replicate": replicate,
                "representatives": int(rows.sum()),
                "unique_start_ids": int(np.unique(compact["global_start_id"][rows]).size),
                "reaudited_feasible": int(replay["feasible"].sum()),
                "reaudited_nonfeasible": mismatch,
                "structures": int(np.unique(compact["structure"][rows]).size),
                "max_archived_vs_replay_quality_difference": float(
                    np.max(np.abs(replay["task_quality"].astype(float) - compact["task_quality"][rows].astype(float)))
                ),
            }
        )
    targets = json.loads((WORK / "configs" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))["numeric_replay"]["validation_targets"]
    passed = bool(
        maximum_normalized <= targets["maximum_window_normalized_prediction_difference"]
        and maximum_quality <= targets["maximum_quality_difference"]
        and total_window_mismatch <= targets["historical_saved_library_window_flag_mismatches"]
        and representative_mismatch <= targets["maximum_archived_mpg_boundary_representatives_with_replay_flag_change"]
        and representative_boundary_max <= targets["maximum_window_normalized_prediction_difference"]
    )
    write_csv(WORK / "status" / "INPUT_MANIFEST.csv", manifest)
    write_csv(WORK / "status" / "FLOAT32_REPLAY_VALIDATION.csv", comparison_rows)
    write_csv(WORK / "status" / "MPG_REPRESENTATIVE_AUDIT.csv", representative_rows)
    result = {
        "ok": passed,
        "historical_rows_checked": int(sum(row["rows"] for row in comparison_rows)),
        "maximum_window_normalized_prediction_difference": maximum_normalized,
        "maximum_quality_difference": maximum_quality,
        "historical_window_flag_mismatches": total_window_mismatch,
        "mpg_representatives_checked": len(compact["design"]),
        "mpg_representative_nonfeasible_after_canonicalization": representative_mismatch,
        "mpg_representative_max_window_normalized_prediction_difference": representative_boundary_max,
        "mpg_boundary_policy": "preserve archived feasible identity/prediction/Q; final shared float64 audit remains authoritative for comparison",
        "files_hashed": len(manifest),
        "frozen_protocol_sha256": sha256(WORK / "configs" / "FROZEN_PROTOCOL.json"),
        "audit_plan_sha256": sha256(WORK / "configs" / "AUDIT_PLAN.json"),
    }
    atomic_json(WORK / "status" / "INPUT_AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("input/replay audit failed; formal runs are blocked")


if __name__ == "__main__":
    main()
