from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    INPUTS,
    RESULTS,
    WORK,
    Problem,
    atomic_json,
    build_library,
    load_npz,
    run_ga,
    select_seeds,
    sha256,
    write_csv,
)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, required=True, choices=range(1, 6))
    args = parser.parse_args()
    replicate = args.replicate
    output = RESULTS / "formal" / f"replicate_{replicate}" / "MPG_GA"
    if output.exists():
        raise FileExistsError(f"immutable formal output already exists: {output}")
    output.mkdir(parents=True)
    started = time.time()
    problem = Problem()
    compact = load_npz(INPUTS / "best_feasible_per_successful_start.npz")
    rows = compact["replicate"] == replicate
    original_design = np.asarray(compact["design"][rows], dtype=np.float64)
    structures = np.asarray(compact["structure"][rows], dtype=int)
    # Canonicalize archived float32 storage tails to the native composition lattice.
    design = np.rint(original_design / problem.step) * problem.step
    replay = problem.audit32(design, structures)
    archived_feasible = np.asarray(compact["feasible"][rows], dtype=bool)
    if not np.all(archived_feasible):
        raise AssertionError("compact representative archive contains a non-feasible row")
    boundary_changed = archived_feasible & ~replay["feasible"]
    if int(boundary_changed.sum()) > 2:
        raise AssertionError("unexpected number of float32 replay boundary changes")
    pool = {
        "design": design,
        "structure": structures,
        "source": np.full(len(design), 8, dtype=np.uint8),
        "global_start_id": compact["global_start_id"][rows],
        "accepted_step": compact["accepted_step"][rows],
        "task_quality": compact["task_quality"][rows],
        "predicted_properties": compact["predicted_properties"][rows],
        "nearest_database_distance": replay["nearest_database_distance"],
        "structural_ok": replay["structural_ok"],
        "window_ok": archived_feasible,
        "feasible": archived_feasible,
        "replay32_task_quality": replay["task_quality"],
        "replay32_predicted_properties": replay["predicted_properties"],
        "replay32_window_ok": replay["window_ok"],
        "replay32_feasible": replay["feasible"],
        "numerical_boundary_flag_change": boundary_changed,
    }
    selected = select_seeds(problem, pool)
    seed_rows = []
    for rank, index in enumerate(selected):
        seed_rows.append(
            {
                "replicate": replicate,
                "selection_rank": rank,
                "pool_index": int(index),
                "structure": int(pool["structure"][index]),
                "global_start_id": int(pool["global_start_id"][index]),
                "accepted_step": int(pool["accepted_step"][index]),
                "replayed_quality": float(pool["task_quality"][index]),
                "archived_quality": float(pool["task_quality"][index]),
                "replay32_quality": float(pool["replay32_task_quality"][index]),
                "numerical_boundary_flag_change": bool(pool["numerical_boundary_flag_change"][index]),
            }
        )
    write_csv(output / "selected_seed_ledger.csv", seed_rows)
    np.save(output / "selected_indices.npy", selected)
    save_npz_atomic(output / "upstream_pool.npz", **pool)
    search, lineages, receipt = run_ga(problem, pool, selected, replicate)
    write_csv(output / "lineages.csv", lineages)
    save_npz_atomic(output / "accepted.npz", **search)
    library, portfolio_indices = build_library(problem, pool, search)
    save_npz_atomic(output / "library.npz", **library)
    save_npz_atomic(output / "portfolio.npz", **{key: value[portfolio_indices] for key, value in library.items()})
    upstream_feasible = int(pool["feasible"].sum())
    completion = {
        **receipt,
        "case": "industrial_pp_100",
        "route": "MPG_GA",
        "upstream_archived_representatives": len(pool["design"]),
        "upstream_reaudited_feasible": upstream_feasible,
        "upstream_reaudited_feasible_rate": upstream_feasible / len(pool["design"]),
        "selected_seeds": len(selected),
        "upstream_numerical_boundary_flag_changes": int(boundary_changed.sum()),
        "selected_structures": int(np.unique(pool["structure"][selected]).size),
        "library_size": len(library["design"]),
        "library_structures": int(np.unique(library["structure"]).size),
        "library_q_ge_0_75": int(np.sum(library["task_quality"] >= 0.75)),
        "portfolio_size": len(portfolio_indices),
        "total_seconds": time.time() - started,
        "input_sha256": {
            "frozen_protocol": sha256(WORK / "configs" / "FROZEN_PROTOCOL.json"),
            "audit_plan": sha256(WORK / "configs" / "AUDIT_PLAN.json"),
            "generation_protocol": sha256(INPUTS / "generation_protocol_v1.json"),
            "recipe_data": sha256(INPUTS / "industrial_pp_100_recipes.csv"),
            "mpg_representatives": sha256(INPUTS / "best_feasible_per_successful_start.npz"),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "numeric_database_anchor_used": False,
        "nearest_row_repair_used": False,
    }
    atomic_json(output / "COMPLETED.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
