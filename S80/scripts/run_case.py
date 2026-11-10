from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

from common import (
    CASES, FROZEN, RESULTS, SOURCE, WORK, Problem, atomic_json, build_library, load_npz,
    optimize_structure, run_ga, save_npz, select_seeds, sha256, write_csv,
)


def run(case: str, replicate: int, pilot: bool) -> None:
    mode = "pilot" if pilot else "formal"
    output = RESULTS / mode / case / f"replicate_{replicate}" / "MPG_GA"
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter(); problem = Problem(case)
    start_path = SOURCE / "results_raw" / "formal_v1" / case / f"replicate_{replicate}" / "upstream" / "R" / "candidates.npz"
    source = load_npz(start_path); all_design = np.asarray(source["design"], dtype=np.float64); all_structure = np.asarray(source["structure"], dtype=int)
    if pilot:
        keep = np.concatenate([
            np.flatnonzero(all_structure == structure)[: int(FROZEN["pilot"]["starts_per_structure"])]
            for structure in np.unique(all_structure)
        ])
        design = all_design[keep]; structures = all_structure[keep]; global_ids = keep
        maximum_updates = int(FROZEN["pilot"]["maximum_updates"]); ga_budget = int(FROZEN["pilot"]["GA_attempts"])
    else:
        design = all_design; structures = all_structure; global_ids = np.arange(len(design), dtype=int)
        maximum_updates = int(FROZEN["mpg_ascent"]["maximum_updates_per_start"]); ga_budget = int(FROZEN["ga"]["attempts_per_run"])
    n = len(design); best_found = np.zeros(n, bool); best_q = np.full(n, -np.inf, np.float32)
    best_design = np.full((n, problem.n), np.nan); best_prediction = np.full((n, len(problem.property_names)), np.nan, np.float32)
    best_margin = np.full(n, np.nan, np.float32); best_step = np.full(n, -1, np.int16); best_objective = np.full(n, np.nan)
    feasible_visits = np.zeros(n, np.int32); audited_visits = np.zeros(n, np.int32); updates = np.zeros(n, np.int16)
    gradients = np.zeros(n, np.int32); backtracks = np.zeros(n, np.int32); stop = np.full(n, -1, np.int8); violations = np.zeros(n, np.int16)
    step_rows = []; counters = {"audited_visits": 0, "feasible_visits": 0}
    for structure in np.unique(structures):
        ids = np.flatnonzero(structures == structure)
        result = optimize_structure(problem, design[ids], int(structure), maximum_updates); collector = result["collector"]
        best_found[ids] = collector.best_found; best_q[ids] = collector.best_q; best_design[ids] = collector.best_design
        best_prediction[ids] = collector.best_prediction; best_margin[ids] = collector.best_margin; best_step[ids] = collector.best_step
        best_objective[ids] = collector.best_objective; feasible_visits[ids] = collector.feasible_visit_count
        audited_visits[ids] = collector.audited_visit_count; updates[ids] = result["updates"]
        gradients[ids] = result["gradient_evaluations"]; backtracks[ids] = result["backtracking"]
        stop[ids] = result["stop"]; violations[ids] = result["monotonic_violations"]
        counters["audited_visits"] += collector.counters["audited_visits"]; counters["feasible_visits"] += collector.counters["feasible_visits"]
        for row in collector.step_rows:
            step_rows.append({"case": case, "replicate": replicate, **row})
        print(f"{case} run {replicate}: MPG structure {structure} complete", flush=True)
    save_npz(output / "per_start_summary.npz", global_start_id=global_ids, structure=structures,
             audited_visit_count=audited_visits, feasible_visit_count=feasible_visits, best_found=best_found,
             best_design=best_design, best_task_quality=best_q, best_predicted_properties=best_prediction,
             best_margin=best_margin, best_accepted_step=best_step, best_continuous_objective=best_objective,
             accepted_updates=updates, gradient_evaluations=gradients, backtracking_forward_evaluations=backtracks,
             stop_reason_code=stop, monotonic_violations=violations)
    write_csv(output / "MPG_visit_summary.csv", step_rows)
    successful = np.flatnonzero(best_found)
    representative_structure = structures[successful]
    representative_audit = problem.audit(best_design[successful], representative_structure, dtype=np.float32)
    if not np.all(representative_audit["feasible"]):
        raise AssertionError("saved best representatives failed immediate replay")
    pool = {"design": best_design[successful], "structure": representative_structure,
            "source": np.full(len(successful), 8, np.uint8), "global_start_id": global_ids[successful],
            "accepted_step": best_step[successful], "feasible_visit_count": feasible_visits[successful], **representative_audit}
    save_npz(output / "representative_pool.npz", **pool)
    selected = select_seeds(problem, pool); np.save(output / "selected_indices.npy", selected)
    seed_rows = [{"case": case, "replicate": replicate, "selection_rank": rank, "pool_index": int(index),
                  "structure": int(pool["structure"][index]), "global_start_id": int(pool["global_start_id"][index]),
                  "accepted_step": int(pool["accepted_step"][index]), "task_quality": float(pool["task_quality"][index]),
                  "feasible_visit_count": int(pool["feasible_visit_count"][index])}
                 for rank, index in enumerate(selected)]
    write_csv(output / "selected_seed_ledger.csv", seed_rows)
    search, lineages, ga_receipt = run_ga(problem, pool, selected, replicate, ga_budget)
    save_npz(output / "accepted.npz", **search); write_csv(output / "lineages.csv", lineages)
    library, portfolio_indices = build_library(problem, pool, search)
    save_npz(output / "library.npz", **library)
    save_npz(output / "portfolio.npz", **{key: value[portfolio_indices] for key, value in library.items()})
    completion = {"case": case, "replicate": replicate, "pilot": pilot, "starts": n,
                  "start_source": str(start_path.resolve()), "start_source_sha256": sha256(start_path),
                  "maximum_updates": maximum_updates, "mpg_audited_visits": counters["audited_visits"],
                  "mpg_feasible_visits": counters["feasible_visits"], "successful_starts": len(successful),
                  "successful_start_rate": len(successful) / n, "selected_seeds": len(selected),
                  "selected_structures": int(np.unique(pool["structure"][selected]).size),
                  "accepted_updates": int(updates.sum()), "gradient_evaluations": int(gradients.sum()),
                  "backtracking_forward_evaluations": int(backtracks.sum()), "monotonic_violations": int(violations.sum()),
                  "ga": ga_receipt, "library_size": len(library["design"]),
                  "library_structures": int(np.unique(library["structure"]).size),
                  "library_Q_ge_0_75_saved_float32": int(np.sum(library["task_quality"] >= 0.75)),
                  "portfolio_size": len(portfolio_indices), "total_seconds": time.perf_counter() - started,
                  "numeric_database_anchor_used": False, "nearest_row_repair_used": False,
                  "protocol_sha256": sha256(WORK / "configs" / "FROZEN_PROTOCOL.json"),
                  "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__}}
    atomic_json(output / "COMPLETED.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--replicate", required=True, type=int, choices=range(1, 6)); parser.add_argument("--pilot", action="store_true")
    arguments = parser.parse_args(); run(arguments.case, arguments.replicate, arguments.pilot)


if __name__ == "__main__":
    main()
