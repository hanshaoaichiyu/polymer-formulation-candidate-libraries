from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import HISTORICAL, RESULTS, WORK, atomic_json, load_npz, sha256, write_csv


CMS = Path(__file__).resolve().parents[2]
V34 = CMS / "output_cms_v34" / "analysis" / "recombine.py"
ROUTES = ("Generator_GA", "R_GA", "MPG_GA")
BITS = {"Generator_GA": 1, "R_GA": 2, "MPG_GA": 4}
ARMS = (*ROUTES, "TRI_GA")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def round_robin(orders: dict[int, list[int]], target: int) -> list[int]:
    structures = sorted(orders)
    cursors = {structure: 0 for structure in structures}
    selected: list[int] = []
    while len(selected) < target:
        progressed = False
        for structure in structures:
            cursor = cursors[structure]
            if cursor < len(orders[structure]):
                selected.append(orders[structure][cursor])
                cursors[structure] += 1
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
    return selected


def main() -> None:
    v34 = load_module("v35_parent_v34", V34)
    historical = v34.historical
    problem = v34.frozen.FrozenProblem()
    bootstrap_path = next((v34.ROUTE / "full_archive").glob(
        "*/results_summary/case_modules/industrial_pp_100/B_bootstrap_indices_industrial_pp_100.npz"
    ))
    with np.load(v34.native(bootstrap_path), allow_pickle=False) as bootstrap:
        counts = bootstrap["counts"]
    input_paths = [V34, bootstrap_path]
    supply_rows: list[dict] = []
    overlap_rows: list[dict] = []
    selection_rows: list[dict] = []
    support_rows: list[dict] = []
    lineage_rows: list[dict] = []
    for replicate in range(1, 6):
        paths = {
            "Generator_GA": HISTORICAL / f"replicate_{replicate}" / "methods" / "G_GA" / "library.npz",
            "R_GA": HISTORICAL / f"replicate_{replicate}" / "methods" / "R_GA" / "library.npz",
            "MPG_GA": RESULTS / "formal" / f"replicate_{replicate}" / "MPG_GA" / "library.npz",
        }
        libraries = []
        key_parts = []
        for route in ROUTES:
            input_paths.append(paths[route])
            library = load_npz(paths[route])
            libraries.append(library)
            key_parts.append(v34.keys(library["design"], library["structure"], problem.step)[0])
        all_keys = np.concatenate(key_parts)
        packed = all_keys.view(np.dtype((np.void, all_keys.dtype.itemsize * all_keys.shape[1]))).ravel()
        _, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
        unique = all_keys[first]
        split = np.split(inverse, np.cumsum([len(keys) for keys in key_parts])[:-1])
        original_membership = np.zeros(len(unique), dtype=np.uint8)
        for route, ids in zip(ROUTES, split):
            original_membership[ids] |= BITS[route]
        design, structure, composition, window, predicted, quality = v34.audit_union(unique, problem)
        keep = composition & window
        membership = np.where(keep, original_membership, 0).astype(np.uint8)
        union = {
            "design": design,
            "integer_coords": unique[:, 1:],
            "structure": structure,
            "task_quality": quality,
            "predicted_properties": predicted,
            "membership_bits": membership,
        }
        masks = {route: (membership & BITS[route]) != 0 for route in ROUTES}
        masks["TRI_GA"] = membership != 0
        for arm in ARMS:
            ids = np.flatnonzero(masks[arm])
            supply_rows.append({
                "replicate": replicate,
                "arm": arm,
                "audited_unique_candidates": len(ids),
                "supports": int(np.unique(structure[ids]).size),
                "Q_ge_0_75": int(np.sum(quality[ids] >= 0.75)),
                "Q_mean_candidate_weighted": float(np.mean(quality[ids])),
            })
        for pattern in range(1, 8):
            ids = np.flatnonzero(membership == pattern)
            overlap_rows.append({
                "replicate": replicate,
                "membership_bits": pattern,
                "membership_label": "+".join(route for route in ROUTES if pattern & BITS[route]),
                "candidates": len(ids),
                "Q_ge_0_75": int(np.sum(quality[ids] >= 0.75)),
                "supports": int(np.unique(structure[ids]).size) if len(ids) else 0,
            })
        cap = int(historical.PROTOCOL["selection"]["case_template_caps"]["industrial_pp_100"])
        orders: dict[str, dict[int, list[int]]] = {}
        for arm in ARMS:
            ids = np.flatnonzero(masks[arm])
            orders[arm] = {
                int(s): historical.select_template_order(
                    ids[structure[ids] == s], union, np.asarray(problem.scale, float), cap
                )
                for s in np.unique(structure[ids])
            }
        common = sorted(set.intersection(*(set(orders[arm]) for arm in ARMS)))
        common_n = {s: min(len(orders[arm][s]) for arm in ARMS) for s in common}
        selections: dict[tuple[str, str], list[int]] = {}
        for arm in ARMS:
            selections[("common_support", arm)] = [
                index for s in common for index in orders[arm][s][: common_n[s]]
            ]
        capacities = {arm: sum(len(values) for values in orders[arm].values()) for arm in ARMS}
        full_target = min(capacities.values())
        for arm in ARMS:
            selections[("full_domain", arm)] = round_robin(orders[arm], full_target)
        selected_union = np.unique(np.concatenate([np.asarray(ids, dtype=int) for ids in selections.values()]))
        frequencies = historical.bootstrap_frequency(problem, design[selected_union], counts)
        bootstrap_map = dict(zip(map(int, selected_union), frequencies))
        for (layer, arm), ids in selections.items():
            metric, details = historical.metrics_for_selection(
                "industrial_pp_100", replicate, layer, arm, ids, union, problem, bootstrap_map
            )
            metric["quota_shortfall"] = 0
            selection_rows.append(metric)
            support_rows.extend(details)
        line_path = RESULTS / "formal" / f"replicate_{replicate}" / "MPG_GA" / "lineages.csv"
        input_paths.append(line_path)
        for row in read_csv(line_path):
            lineage_rows.append({
                "replicate": replicate,
                "lineage": int(row["lineage"]),
                "structure": int(row["structure"]),
                "global_start_id": int(row["global_start_id"]),
                "initial_quality": float(row["initial_quality"]),
                "best_quality": float(row["best_quality"]),
                "quality_gain": float(row["quality_gain"]),
                "improved": float(row["quality_gain"]) > 0,
                "accepted": int(row["accepted"]),
            })
    summary_rows: list[dict] = []
    for arm in ARMS:
        subset = [row for row in supply_rows if row["arm"] == arm]
        for metric in ("audited_unique_candidates", "supports", "Q_ge_0_75", "Q_mean_candidate_weighted"):
            mean, sd = mean_sd([float(row[metric]) for row in subset])
            summary_rows.append({"section": "full_supply", "layer": "full", "arm": arm, "metric": metric, "mean": mean, "sample_sd": sd})
    selection_metrics = (
        "selected_n", "selected_support_n", "Q_mean_candidate_weighted", "Q_mean_template_equal",
        "pairwise_rms_template_equal", "nearest_neighbor_mean_template_equal",
        "nominal_rate_candidate_weighted", "sensitive_rate_candidate_weighted",
    )
    for layer in ("common_support", "full_domain"):
        for arm in ARMS:
            subset = [row for row in selection_rows if row["layer"] == layer and row["arm"] == arm]
            for metric in selection_metrics:
                mean, sd = mean_sd([float(row[metric]) for row in subset])
                summary_rows.append({"section": "fixed_selection", "layer": layer, "arm": arm, "metric": metric, "mean": mean, "sample_sd": sd})
    lineage_summary = []
    for replicate in range(1, 6):
        rows = [row for row in lineage_rows if row["replicate"] == replicate]
        lineage_summary.append({
            "replicate": replicate,
            "lineages": len(rows),
            "improved": int(sum(row["improved"] for row in rows)),
            "improved_rate": float(np.mean([row["improved"] for row in rows])),
            "mean_quality_gain": float(np.mean([row["quality_gain"] for row in rows])),
            "median_quality_gain": float(np.median([row["quality_gain"] for row in rows])),
            "mean_initial_quality": float(np.mean([row["initial_quality"] for row in rows])),
            "mean_best_quality": float(np.mean([row["best_quality"] for row in rows])),
        })
    write_csv(RESULTS / "three_route_full_supply_by_run.csv", supply_rows)
    write_csv(RESULTS / "three_route_membership_by_run.csv", overlap_rows)
    write_csv(RESULTS / "three_route_fixed_selection_by_run.csv", selection_rows)
    write_csv(RESULTS / "three_route_fixed_selection_by_support.csv", support_rows)
    write_csv(RESULTS / "three_route_five_run_summary.csv", summary_rows)
    write_csv(RESULTS / "mpg_ga_lineage_by_run.csv", lineage_summary)
    manifest = []
    for path in sorted(set(input_paths), key=str):
        native_path = v34.native(path)
        manifest.append({"path": str(path.resolve()), "bytes": native_path.stat().st_size, "sha256": historical.sha256(native_path)})
    write_csv(WORK / "status" / "ANALYSIS_INPUT_MANIFEST.csv", manifest)
    get = lambda section, layer, arm, metric: next(
        row for row in summary_rows if row["section"] == section and row["layer"] == layer and row["arm"] == arm and row["metric"] == metric
    )["mean"]
    judgment = {
        "core_system": "Generator_GA + R_GA + MPG_GA",
        "mean_full_unique": {arm: get("full_supply", "full", arm, "audited_unique_candidates") for arm in ARMS},
        "mean_full_Q_ge_0_75": {arm: get("full_supply", "full", arm, "Q_ge_0_75") for arm in ARMS},
        "common_support_Q_template_equal": {arm: get("fixed_selection", "common_support", arm, "Q_mean_template_equal") for arm in ARMS},
        "common_support_RMS_template_equal": {arm: get("fixed_selection", "common_support", arm, "pairwise_rms_template_equal") for arm in ARMS},
        "common_support_nominal_rate": {arm: get("fixed_selection", "common_support", arm, "nominal_rate_candidate_weighted") for arm in ARMS},
        "common_support_sensitive_rate": {arm: get("fixed_selection", "common_support", arm, "sensitive_rate_candidate_weighted") for arm in ARMS},
        "mpg_ga_improved_lineage_rate_mean": float(np.mean([row["improved_rate"] for row in lineage_summary])),
        "interpretation": "Three seed sources now share one downstream GA. Differences can be attributed to initialization/search-path supply under the frozen evaluator, subject to equal-attempt rather than equal-time scope. TRI_GA is assessed as a traceable candidate-supply union, not required to dominate every single-route fixed-count metric.",
    }
    atomic_json(RESULTS / "THREE_ROUTE_JUDGMENT.json", judgment)
    print(json.dumps(judgment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
