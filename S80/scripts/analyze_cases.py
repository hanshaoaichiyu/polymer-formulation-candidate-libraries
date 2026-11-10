from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import CASES, FROZEN, RESULTS, SOURCE, WORK, Problem, atomic_json, load_npz, sha256, write_csv


ROUTES = ("Generator_GA", "R_GA", "MPG_GA")
BITS = {"Generator_GA": 1, "R_GA": 2, "MPG_GA": 4}
ARMS = (*ROUTES, "TRI_GA")
BOOTSTRAP_ROOT = WORK.parent / "PES_v11_review_supplement_20261012_v1" / "results_summary" / "case_modules"


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def select_order(problem: Problem, ids: np.ndarray, union: dict, cap: int) -> list[int]:
    if not len(ids):
        return []
    quality = union["task_quality"]; threshold = float(np.quantile(quality[ids], 0.5))
    eligible = ids[quality[ids] >= threshold]
    first_candidates = eligible[quality[eligible] == np.max(quality[eligible])]
    first = int(first_candidates[np.lexsort(union["integer_coords"][first_candidates].T[::-1])[0]])
    selected = [first]
    while len(selected) < min(cap, len(eligible)):
        remaining = np.asarray([index for index in eligible if index not in selected], dtype=int)
        normalized = union["design"][remaining] / problem.scale
        chosen = union["design"][np.asarray(selected)] / problem.scale
        minimum = np.min(np.linalg.norm(normalized[:, None, :] - chosen[None, :, :], axis=2), axis=1)
        best_distance = np.max(minimum); candidates = remaining[np.isclose(minimum, best_distance, rtol=0, atol=1e-14)]
        best_quality = np.max(quality[candidates]); candidates = candidates[np.isclose(quality[candidates], best_quality, rtol=0, atol=1e-14)]
        selected.append(int(candidates[np.lexsort(union["integer_coords"][candidates].T[::-1])[0]]))
    return selected


def round_robin(orders: dict[int, list[int]], target: int) -> list[int]:
    structures = sorted(orders); cursors = {structure: 0 for structure in structures}; selected = []
    while len(selected) < target:
        progressed = False
        for structure in structures:
            cursor = cursors[structure]
            if cursor < len(orders[structure]):
                selected.append(orders[structure][cursor]); cursors[structure] += 1; progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
    return selected


def bootstrap_frequency(problem: Problem, design: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if not len(design):
        return np.empty(0)
    normalized = design / problem.scale; centers = problem.x_obs / problem.scale
    d2 = np.maximum(np.sum(normalized * normalized, axis=1)[:, None] + np.sum(centers * centers, axis=1)[None, :] - 2.0 * normalized @ centers.T, 0.0)
    base = -d2 / (2.0 * problem.bandwidth**2 + 1e-12); passed = np.zeros(len(design), dtype=np.int32)
    for count in counts:
        prediction = np.full((len(design), len(problem.property_names)), np.nan)
        for response in range(len(problem.property_names)):
            valid = np.isfinite(problem.y_obs[:, response]) & (count > 0)
            log_weight = base[:, valid] + np.log(count[valid].astype(float))[None, :]
            maximum = log_weight.max(axis=1, keepdims=True); weight = np.exp(log_weight - maximum)
            prediction[:, response] = (weight @ problem.y_obs[valid, response]) / weight.sum(axis=1)
        passed += np.all((prediction >= problem.windows[:, 0]) & (prediction <= problem.windows[:, 1]), axis=1)
    return passed / len(counts)


def distances(design: np.ndarray, scale: np.ndarray) -> dict:
    if len(design) < 2:
        return {"pair_n": 0, "pairwise_mean": None, "pairwise_rms": None, "nearest_neighbor_mean": None}
    z = design / scale; d2 = np.maximum(np.sum(z * z, axis=1)[:, None] + np.sum(z * z, axis=1)[None, :] - 2.0 * z @ z.T, 0.0)
    upper = np.triu_indices(len(z), 1); pair = np.sqrt(d2[upper]); np.fill_diagonal(d2, np.inf)
    return {"pair_n": len(pair), "pairwise_mean": float(pair.mean()), "pairwise_rms": float(np.sqrt(np.mean(pair * pair))),
            "nearest_neighbor_mean": float(np.mean(np.sqrt(np.min(d2, axis=1))))}


def selection_metrics(case: str, replicate: int, layer: str, arm: str, ids: list[int], union: dict,
                      problem: Problem, bootstrap_map: dict[int, float]) -> tuple[dict, list[dict]]:
    indices = np.asarray(ids, dtype=int); quality = union["task_quality"][indices]
    margin = problem.margin(union["predicted_properties"][indices]); bootstrap = np.asarray([bootstrap_map[int(index)] for index in indices])
    nominal = (quality >= 0.75) & (margin >= 0.05); sensitive = nominal & (bootstrap >= 0.80)
    support_rows = []
    for structure in sorted(map(int, np.unique(union["structure"][indices]))):
        local = np.flatnonzero(union["structure"][indices] == structure); local_ids = indices[local]; metric = distances(union["design"][local_ids], problem.scale)
        support_rows.append({"case": case, "replicate": replicate, "layer": layer, "arm": arm, "structure": structure,
                             "selected_n": len(local), "Q_mean": float(quality[local].mean()), "Q_median": float(np.median(quality[local])),
                             "minimum_margin_mean": float(margin[local].mean()), "nominal_rate": float(nominal[local].mean()),
                             "sensitive_rate": float(sensitive[local].mean()), "bootstrap_frequency_mean": float(bootstrap[local].mean()), **metric})
    finite = [row for row in support_rows if row["pair_n"]]
    run = {"case": case, "replicate": replicate, "layer": layer, "arm": arm, "selected_n": len(indices),
           "selected_support_n": len(support_rows), "Q_mean_candidate_weighted": float(quality.mean()),
           "Q_mean_support_equal": float(np.mean([row["Q_mean"] for row in support_rows])),
           "pairwise_rms_support_equal": float(np.mean([row["pairwise_rms"] for row in finite])) if finite else None,
           "nearest_neighbor_support_equal": float(np.mean([row["nearest_neighbor_mean"] for row in finite])) if finite else None,
           "nominal_rate_candidate_weighted": float(nominal.mean()),
           "nominal_rate_support_equal": float(np.mean([row["nominal_rate"] for row in support_rows])),
           "sensitive_rate_candidate_weighted": float(sensitive.mean()),
           "sensitive_rate_support_equal": float(np.mean([row["sensitive_rate"] for row in support_rows]))}
    return run, support_rows


def main() -> None:
    supply_rows = []; membership_rows = []; selection_rows = []; support_rows = []; resource_rows = []; input_paths = []
    for case in CASES:
        problem = Problem(case); cap = int(FROZEN["portfolio_per_structure"][case])
        bootstrap_path = BOOTSTRAP_ROOT / case / f"B_bootstrap_indices_{case}.npz"; input_paths.append(bootstrap_path)
        counts = load_npz(bootstrap_path)["counts"]
        for replicate in range(1, 6):
            paths = {
                "Generator_GA": SOURCE / "results_raw" / "formal_v1" / case / f"replicate_{replicate}" / "methods" / "G_GA" / "library.npz",
                "R_GA": SOURCE / "results_raw" / "formal_v1" / case / f"replicate_{replicate}" / "methods" / "R_GA" / "library.npz",
                "MPG_GA": RESULTS / "formal" / case / f"replicate_{replicate}" / "MPG_GA" / "library.npz",
            }
            libraries = []; lengths = []
            for route in ROUTES:
                input_paths.append(paths[route]); library = load_npz(paths[route]); libraries.append(library); lengths.append(len(library["design"]))
            all_design = np.concatenate([library["design"] for library in libraries]); all_structure = np.concatenate([library["structure"] for library in libraries])
            all_keys = problem.coordinate_key(all_design, all_structure); packed = np.ascontiguousarray(all_keys).view(np.dtype((np.void, all_keys.dtype.itemsize * all_keys.shape[1]))).ravel()
            _, first, inverse = np.unique(packed, return_index=True, return_inverse=True); design = all_design[first]; structure = all_structure[first]; integer_coords = all_keys[first, 1:]
            membership = np.zeros(len(first), dtype=np.uint8); splits = np.split(inverse, np.cumsum(lengths)[:-1])
            for route, ids in zip(ROUTES, splits):
                membership[ids] |= BITS[route]
            audit = problem.audit(design, structure, dtype=np.float64); membership = np.where(audit["feasible"], membership, 0).astype(np.uint8)
            union = {"design": design, "structure": structure, "integer_coords": integer_coords,
                     "task_quality": audit["task_quality"], "predicted_properties": audit["predicted_properties"], "membership_bits": membership}
            masks = {route: (membership & BITS[route]) != 0 for route in ROUTES}; masks["TRI_GA"] = membership != 0
            for arm in ARMS:
                ids = np.flatnonzero(masks[arm]); supply_rows.append({"case": case, "replicate": replicate, "arm": arm,
                    "audited_unique_candidates": len(ids), "supports": int(np.unique(structure[ids]).size),
                    "Q_ge_0_75": int(np.sum(audit["task_quality"][ids] >= 0.75)), "Q_mean_candidate_weighted": float(audit["task_quality"][ids].mean())})
            for pattern in range(1, 8):
                ids = np.flatnonzero(membership == pattern); membership_rows.append({"case": case, "replicate": replicate,
                    "membership_bits": pattern, "membership_label": "+".join(route for route in ROUTES if pattern & BITS[route]),
                    "candidates": len(ids), "Q_ge_0_75": int(np.sum(audit["task_quality"][ids] >= 0.75)),
                    "supports": int(np.unique(structure[ids]).size) if len(ids) else 0})
            orders = {}
            for arm in ARMS:
                ids = np.flatnonzero(masks[arm]); orders[arm] = {
                    int(s): select_order(problem, ids[structure[ids] == s], union, cap) for s in np.unique(structure[ids])
                }
            common = sorted(set.intersection(*(set(orders[arm]) for arm in ARMS)))
            per_support = {s: min(len(orders[arm][s]) for arm in ARMS) for s in common}
            selections = {("common_support", arm): [index for s in common for index in orders[arm][s][: per_support[s]]] for arm in ARMS}
            capacities = {arm: sum(len(values) for values in orders[arm].values()) for arm in ARMS}; target = min(capacities.values())
            for arm in ARMS:
                selections[("full_domain", arm)] = round_robin(orders[arm], target)
            selected_union = np.unique(np.concatenate([np.asarray(ids, dtype=int) for ids in selections.values()]))
            frequency = bootstrap_frequency(problem, design[selected_union], counts); bootstrap_map = dict(zip(map(int, selected_union), frequency))
            for (layer, arm), ids in selections.items():
                run, details = selection_metrics(case, replicate, layer, arm, ids, union, problem, bootstrap_map)
                selection_rows.append(run); support_rows.extend(details)
            completion_path = RESULTS / "formal" / case / f"replicate_{replicate}" / "MPG_GA" / "COMPLETED.json"
            input_paths.append(completion_path); completion = json.loads(completion_path.read_text(encoding="utf-8"))
            resource_rows.append({"case": case, "replicate": replicate, "successful_start_rate": completion["successful_start_rate"],
                                  "mpg_audited_visits": completion["mpg_audited_visits"], "mpg_feasible_visits": completion["mpg_feasible_visits"],
                                  "GA_acceptance_rate": completion["ga"]["accepted_rate"], "GA_improved_lineages": completion["ga"]["improved_lineages"],
                                  "GA_lineages": completion["ga"]["lineages"], "total_seconds": completion["total_seconds"]})
    summary = []
    for case in CASES:
        for arm in ARMS:
            subset = [row for row in supply_rows if row["case"] == case and row["arm"] == arm]
            for metric in ("audited_unique_candidates", "supports", "Q_ge_0_75", "Q_mean_candidate_weighted"):
                mean, sd = mean_sd([float(row[metric]) for row in subset]); summary.append({"case": case, "section": "full_supply", "layer": "full", "arm": arm, "metric": metric, "mean": mean, "sample_sd": sd})
        for layer in ("common_support", "full_domain"):
            for arm in ARMS:
                subset = [row for row in selection_rows if row["case"] == case and row["layer"] == layer and row["arm"] == arm]
                for metric in ("selected_n", "selected_support_n", "Q_mean_candidate_weighted", "Q_mean_support_equal",
                               "pairwise_rms_support_equal", "nearest_neighbor_support_equal", "nominal_rate_candidate_weighted",
                               "nominal_rate_support_equal", "sensitive_rate_candidate_weighted", "sensitive_rate_support_equal"):
                    values = [float(row[metric]) for row in subset if row[metric] is not None]
                    mean, sd = mean_sd(values); summary.append({"case": case, "section": "fixed_selection", "layer": layer, "arm": arm, "metric": metric, "mean": mean, "sample_sd": sd})
    write_csv(RESULTS / "two_case_full_supply_by_run.csv", supply_rows)
    write_csv(RESULTS / "two_case_membership_by_run.csv", membership_rows)
    write_csv(RESULTS / "two_case_fixed_selection_by_run.csv", selection_rows)
    write_csv(RESULTS / "two_case_fixed_selection_by_support.csv", support_rows)
    write_csv(RESULTS / "two_case_five_run_summary.csv", summary)
    write_csv(RESULTS / "two_case_MPG_GA_resources_by_run.csv", resource_rows)
    manifest = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(set(input_paths), key=str)]
    write_csv(WORK / "status" / "ANALYSIS_INPUT_MANIFEST.csv", manifest)
    def get(case, section, layer, arm, metric):
        return next(row["mean"] for row in summary if row["case"] == case and row["section"] == section and row["layer"] == layer and row["arm"] == arm and row["metric"] == metric)
    judgment = {}
    for case in CASES:
        judgment[case] = {
            "full_unique": {arm: get(case, "full_supply", "full", arm, "audited_unique_candidates") for arm in ARMS},
            "full_Q_ge_0_75": {arm: get(case, "full_supply", "full", arm, "Q_ge_0_75") for arm in ARMS},
            "common_support_Q": {arm: get(case, "fixed_selection", "common_support", arm, "Q_mean_support_equal") for arm in ARMS},
            "common_support_RMS": {arm: get(case, "fixed_selection", "common_support", arm, "pairwise_rms_support_equal") for arm in ARMS},
            "common_support_sensitive_rate": {arm: get(case, "fixed_selection", "common_support", arm, "sensitive_rate_support_equal") for arm in ARMS},
        }
    atomic_json(RESULTS / "TWO_CASE_JUDGMENT.json", judgment)
    print(json.dumps(judgment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
