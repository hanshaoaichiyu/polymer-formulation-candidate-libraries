from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import CASES, FROZEN, INPUTS, SOURCE, WORK, Problem, atomic_json, load_npz, sha256, write_csv


def record(path: Path, role: str) -> dict:
    return {"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    manifest = [
        record(WORK / "configs" / "AUDIT_PLAN.json", "predeclared audit plan"),
        record(WORK / "configs" / "FROZEN_PROTOCOL.json", "frozen protocol"),
        record(INPUTS / "generation_protocol_v1.json", "generation protocol"),
        record(INPUTS / "data" / "pp_hdpe_banana_recipes.csv", "natural-fiber evaluator data"),
        record(INPUTS / "data" / "pcm_material_designs.csv", "PCM evaluator data"),
    ]
    rows = []; maximum_prediction = maximum_quality = 0.0; mismatches = 0; structural_failures = 0
    targets = FROZEN["numeric_replay_validation"]
    for case in CASES:
        problem = Problem(case)
        for replicate in range(1, 6):
            start_path = SOURCE / "results_raw" / "formal_v1" / case / f"replicate_{replicate}" / "upstream" / "R" / "candidates.npz"
            manifest.append(record(start_path, f"{case} run {replicate} MPG starts"))
            starts = load_npz(start_path)
            start_structural = np.fromiter((problem.structural(x, int(s)) for x, s in zip(starts["design"], starts["structure"])), bool, len(starts["design"]))
            if not np.all(start_structural):
                raise AssertionError(f"{case} run {replicate}: nonstructural formal R start")
            for route in ("G_GA", "R_GA"):
                path = SOURCE / "results_raw" / "formal_v1" / case / f"replicate_{replicate}" / "methods" / route / "library.npz"
                manifest.append(record(path, f"{case} run {replicate} historical {route}"))
                saved = load_npz(path); replay = problem.audit(saved["design"], saved["structure"], dtype=np.float32)
                width = np.ptp(problem.windows, axis=1)
                pred_difference = float(np.max(np.abs(replay["predicted_properties"].astype(float) - saved["predicted_properties"].astype(float)) / width))
                q_difference = float(np.max(np.abs(replay["task_quality"].astype(float) - saved["task_quality"].astype(float))))
                saved_window = np.all((saved["predicted_properties"] >= problem.windows[:, 0]) & (saved["predicted_properties"] <= problem.windows[:, 1]), axis=1)
                mismatch = int(np.sum(replay["window_ok"] != saved_window)); structural = int(np.sum(~replay["structural_ok"]))
                rows.append({"case": case, "replicate": replicate, "route": route, "rows": len(saved["design"]),
                             "max_window_normalized_prediction_difference": pred_difference,
                             "max_quality_difference": q_difference, "window_flag_mismatches": mismatch,
                             "structural_failures": structural, "R_starts": len(starts["design"]),
                             "R_start_saved_feasible": int(starts["feasible"].sum())})
                maximum_prediction = max(maximum_prediction, pred_difference); maximum_quality = max(maximum_quality, q_difference)
                mismatches += mismatch; structural_failures += structural
    ok = (
        maximum_prediction <= float(targets["maximum_window_normalized_prediction_difference"])
        and maximum_quality <= float(targets["maximum_quality_difference"])
        and mismatches <= int(targets["maximum_window_flag_mismatches"])
        and structural_failures == 0
    )
    write_csv(WORK / "status" / "INPUT_MANIFEST.csv", manifest)
    write_csv(WORK / "status" / "NUMPY_FLOAT32_REPLAY_VALIDATION.csv", rows)
    result = {"ok": ok, "historical_rows_checked": sum(row["rows"] for row in rows),
              "maximum_window_normalized_prediction_difference": maximum_prediction,
              "maximum_quality_difference": maximum_quality, "window_flag_mismatches": mismatches,
              "structural_failures": structural_failures, "files_hashed": len(manifest),
              "protocol_sha256": sha256(WORK / "configs" / "FROZEN_PROTOCOL.json")}
    atomic_json(WORK / "status" / "INPUT_AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("input audit failed; pilot/formal runs blocked")


if __name__ == "__main__":
    main()
