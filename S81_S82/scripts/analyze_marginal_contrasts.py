from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parents[1]
RESULTS = WORK / "results" / "marginal"
V16 = WORK.parent / "CMS_v16_focused_analysis_20261014_v1" / "open_source_package" / "results_raw" / "mpga_t_paired_v16_formal" / "formal_v1"
CMS = Path(__file__).resolve().parents[2]
V34 = CMS / "output_cms_v34" / "analysis" / "recombine.py"
FRACTIONS = (0.25, 0.5, 1.0)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def packed(keys: np.ndarray) -> np.ndarray:
    keys = np.ascontiguousarray(keys)
    return keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()


def subset_keys(v34, problem, data: dict, feasible_only: bool = False) -> np.ndarray:
    keep = np.ones(len(data["design"]), bool)
    if feasible_only and "feasible" in data:
        keep &= data["feasible"]
    return v34.keys(data["design"][keep], data["structure"][keep], problem.step)[0]


def ga_keys(v34, problem, replicate: int, route: str) -> tuple[np.ndarray, np.ndarray]:
    data = load(WORK / "results" / "formal" / f"replicate_{replicate}" / route / "proposals.npz")
    keys = subset_keys(v34, problem, data, feasible_only=True)
    candidate = data["candidate_id"][data["feasible"]]
    return keys[np.argsort(candidate)], candidate[np.argsort(candidate)]


def metrics(ids: np.ndarray, structure: np.ndarray, quality: np.ndarray) -> dict:
    return {
        "audited_unique_candidates": int(len(ids)),
        "Q_ge_0_75": int(np.sum(quality[ids] >= 0.75)),
        "supports": int(np.unique(structure[ids]).size) if len(ids) else 0,
        "Q_mean_candidate_weighted": float(np.mean(quality[ids])) if len(ids) else float("nan"),
        "Q_max": float(np.max(quality[ids])) if len(ids) else float("nan"),
    }


def analyze_group(v34, historical, problem, counts, replicate: int, group: str,
                  baseline_name: str, baseline_keys: np.ndarray,
                  additions: dict[str, np.ndarray], costs: dict[str, float],
                  supply_rows: list[dict], selection_rows: list[dict], curve_rows: list[dict]) -> None:
    arms = {baseline_name: baseline_keys}
    for arm, extra in additions.items():
        arms[arm] = np.concatenate((baseline_keys, extra))
    all_keys = np.concatenate(list(arms.values()))
    all_packed = packed(all_keys)
    unique_packed, first, inverse = np.unique(all_packed, return_index=True, return_inverse=True)
    unique = all_keys[first]
    design, structure, composition, window, prediction, quality = v34.audit_union(unique, problem)
    valid = composition & window
    split = np.split(inverse, np.cumsum([len(value) for value in arms.values()])[:-1])
    arm_ids = {}
    for (arm, _), ids in zip(arms.items(), split):
        ids = np.unique(ids); ids = ids[valid[ids]]; arm_ids[arm] = ids
        row = {"replicate": replicate, "group": group, "arm": arm, **metrics(ids, structure, quality)}
        base_ids = arm_ids.get(baseline_name, ids)
        if arm != baseline_name:
            row["new_unique_vs_baseline"] = int(len(np.setdiff1d(ids, base_ids, assume_unique=True)))
            row["new_Q_ge_0_75_vs_baseline"] = int(np.sum(quality[np.setdiff1d(ids, base_ids, assume_unique=True)] >= 0.75))
            row["seconds"] = float(costs[arm])
            row["new_unique_per_second"] = row["new_unique_vs_baseline"] / row["seconds"]
            row["new_Q_ge_0_75_per_second"] = row["new_Q_ge_0_75_vs_baseline"] / row["seconds"]
        supply_rows.append(row)
    cap = int(historical.PROTOCOL["selection"]["case_template_caps"]["industrial_pp_100"])
    union = {"design": design, "integer_coords": unique[:, 1:], "structure": structure,
             "task_quality": quality, "predicted_properties": prediction}
    orders = {
        arm: {int(s): historical.select_template_order(ids[structure[ids] == s], union, problem.scale, cap)
              for s in np.unique(structure[ids])}
        for arm, ids in arm_ids.items()
    }
    common = sorted(set.intersection(*(set(value) for value in orders.values())))
    common_n = {s: min(cap, *(len(orders[arm][s]) for arm in orders)) for s in common}
    selections = {arm: [index for s in common for index in orders[arm][s][:common_n[s]]] for arm in orders}
    selected_union = np.unique(np.concatenate([np.asarray(ids, int) for ids in selections.values()]))
    frequencies = historical.bootstrap_frequency(problem, design[selected_union], counts)
    bootstrap_map = dict(zip(map(int, selected_union), frequencies))
    for arm, ids in selections.items():
        result, _ = historical.metrics_for_selection(
            "industrial_pp_100", replicate, f"{group}_common_support", arm, ids, union, problem, bootstrap_map
        )
        result.update({"group": group, "baseline": baseline_name, "common_support_n": len(common)})
        selection_rows.append(result)
    baseline_ids = arm_ids[baseline_name]
    baseline_packed = unique_packed[baseline_ids]
    for arm, extra in additions.items():
        extra_packed = packed(extra)
        for fraction in FRACTIONS:
            n = max(1, int(np.floor(len(extra_packed) * fraction)))
            prefix_ids = np.searchsorted(unique_packed, extra_packed[:n])
            ids = np.unique(np.concatenate((baseline_ids, prefix_ids)))
            ids = ids[valid[ids]]
            row = {"replicate": replicate, "group": group, "arm": arm, "work_fraction": fraction,
                   "prefix_records": n, "cost_axis": "proportional_work_fraction_not_measured_prefix_time",
                   "estimated_seconds": float(costs[arm]) * fraction, **metrics(ids, structure, quality)}
            row["new_unique_vs_baseline"] = int(len(np.setdiff1d(ids, baseline_ids, assume_unique=True)))
            curve_rows.append(row)


def main() -> None:
    v34 = load_module("v42_parent_v34", V34)
    historical = v34.historical
    problem = v34.frozen.FrozenProblem()
    bootstrap_path = WORK.parent / "PES_v11_review_supplement_20261012_v1" / "results_summary" / "case_modules" / "industrial_pp_100" / "B_bootstrap_indices_industrial_pp_100.npz"
    with np.load(bootstrap_path, allow_pickle=False) as z:
        counts = z["counts"]
    representatives = load(WORK / "inputs" / "best_feasible_per_successful_start.npz")
    supply_rows: list[dict] = []; selection_rows: list[dict] = []; curve_rows: list[dict] = []; cost_rows: list[dict] = []
    for replicate in range(1, 6):
        main = read_json(RESULTS / "formal" / f"replicate_{replicate}" / "COMPLETED.json")
        supplement = read_json(RESULTS / "cost_supplement" / f"replicate_{replicate}" / "COMPLETED.json")
        mpg_ga_seconds = float(supplement["GA_no_logging_cost"]["MPG_GA"]["seconds"])
        gen_ga_seconds = float(supplement["GA_no_logging_cost"]["Generator_GA"]["seconds"])
        a2_seconds = float(main["A2_cost"]["seconds"])
        a3_seconds = float(sum(row["seconds"] for row in main["A3_cost"]))
        b2_5_seconds = float(sum(row["seconds"] for row in main["B2_cost"]))
        b2_6_seconds = b2_5_seconds + float(supplement["B2_sixth_batch_cost"]["seconds"])
        cost_rows.extend([
            {"replicate": replicate, "arm": "A1_MPG_GA", "seconds": mpg_ga_seconds, "budget_reference": mpg_ga_seconds, "ratio_to_GA": 1.0},
            {"replicate": replicate, "arm": "A2_continue_MPGA", "seconds": a2_seconds, "budget_reference": mpg_ga_seconds, "ratio_to_GA": a2_seconds / mpg_ga_seconds},
            {"replicate": replicate, "arm": "A3_new_MPGA_starts", "seconds": a3_seconds, "budget_reference": mpg_ga_seconds, "ratio_to_GA": a3_seconds / mpg_ga_seconds},
            {"replicate": replicate, "arm": "B1_Generator_GA", "seconds": gen_ga_seconds, "budget_reference": gen_ga_seconds, "ratio_to_GA": 1.0},
            {"replicate": replicate, "arm": "B2_more_generator_5_batches", "seconds": b2_5_seconds, "budget_reference": gen_ga_seconds, "ratio_to_GA": b2_5_seconds / gen_ga_seconds},
            {"replicate": replicate, "arm": "B2_more_generator_6_batches", "seconds": b2_6_seconds, "budget_reference": gen_ga_seconds, "ratio_to_GA": b2_6_seconds / gen_ga_seconds},
        ])
        rep_keep = representatives["replicate"] == replicate
        a0_rep = subset_keys(v34, problem, {key: value[rep_keep] for key, value in representatives.items()}, True)
        a0_full = subset_keys(v34, problem, load(V16 / f"replicate_{replicate}" / "trajectory_feasible_candidates.npz"), False)
        a1, _ = ga_keys(v34, problem, replicate, "MPG_GA")
        a2_data = load(RESULTS / "formal" / f"replicate_{replicate}" / "A2_continued_MPGA_visits.npz")
        a2 = subset_keys(v34, problem, a2_data, True)
        a3_data = load(RESULTS / "formal" / f"replicate_{replicate}" / "A3_new_start_MPGA_visits.npz")
        a3 = subset_keys(v34, problem, a3_data, True)
        a_add = {"A1_MPG_GA": a1, "A2_continue_MPGA": a2, "A3_new_MPGA_starts": a3}
        a_cost = {"A1_MPG_GA": mpg_ga_seconds, "A2_continue_MPGA": a2_seconds, "A3_new_MPGA_starts": a3_seconds}
        analyze_group(v34, historical, problem, counts, replicate, "A_representative_baseline", "A0_representatives", a0_rep, a_add, a_cost, supply_rows, selection_rows, curve_rows)
        analyze_group(v34, historical, problem, counts, replicate, "A_full_visit_baseline", "A0_full_visits", a0_full, a_add, a_cost, supply_rows, selection_rows, curve_rows)
        b0 = subset_keys(v34, problem, load(WORK / "inputs" / "historical_formal" / f"replicate_{replicate}" / "G" / "candidates.npz"), True)
        b1, _ = ga_keys(v34, problem, replicate, "Generator_GA")
        b2_5_data = load(RESULTS / "formal" / f"replicate_{replicate}" / "B2_additional_generator_candidates.npz")
        b2_5 = subset_keys(v34, problem, b2_5_data, True)
        sixth_data = load(RESULTS / "cost_supplement" / f"replicate_{replicate}" / "B2_sixth_generator_batch.npz")
        sixth = subset_keys(v34, problem, sixth_data, True)
        analyze_group(v34, historical, problem, counts, replicate, "B_generator_baseline", "B0_generator",
                      b0, {"B1_Generator_GA": b1, "B2_more_generator_5_batches": b2_5,
                           "B2_more_generator_6_batches": np.concatenate((b2_5, sixth))},
                      {"B1_Generator_GA": gen_ga_seconds, "B2_more_generator_5_batches": b2_5_seconds,
                       "B2_more_generator_6_batches": b2_6_seconds}, supply_rows, selection_rows, curve_rows)
    write_csv(RESULTS / "analysis" / "supply_and_marginal_by_run.csv", supply_rows)
    write_csv(RESULTS / "analysis" / "fixed_selection_by_run.csv", selection_rows)
    write_csv(RESULTS / "analysis" / "work_fraction_curves_by_run.csv", curve_rows)
    write_csv(RESULTS / "analysis" / "cost_and_budget_match_by_run.csv", cost_rows)
    supply_lookup = {(r["replicate"], r["group"], r["arm"]): r for r in supply_rows}
    selection_lookup = {(int(r["replicate"]), r["group"], r["arm"]): r for r in selection_rows}
    cost_lookup = {(r["replicate"], r["arm"]): r for r in cost_rows}
    paired_rows = []
    contrasts = [
        ("A_full_visit_baseline", "A1_MPG_GA", "A2_continue_MPGA"),
        ("A_full_visit_baseline", "A1_MPG_GA", "A3_new_MPGA_starts"),
        ("B_generator_baseline", "B1_Generator_GA", "B2_more_generator_5_batches"),
        ("B_generator_baseline", "B1_Generator_GA", "B2_more_generator_6_batches"),
    ]
    for replicate in range(1, 6):
        for group, left, right in contrasts:
            l = supply_lookup[(replicate, group, left)]; r = supply_lookup[(replicate, group, right)]
            ls = selection_lookup[(replicate, group, left)]; rs = selection_lookup[(replicate, group, right)]
            paired_rows.append({
                "replicate": replicate, "group": group, "left": left, "right": right,
                "delta_new_unique_left_minus_right": float(l["new_unique_vs_baseline"]) - float(r["new_unique_vs_baseline"]),
                "delta_new_Q_ge_0_75_left_minus_right": float(l["new_Q_ge_0_75_vs_baseline"]) - float(r["new_Q_ge_0_75_vs_baseline"]),
                "delta_new_unique_per_second_left_minus_right": float(l["new_unique_per_second"]) - float(r["new_unique_per_second"]),
                "delta_new_Q_ge_0_75_per_second_left_minus_right": float(l["new_Q_ge_0_75_per_second"]) - float(r["new_Q_ge_0_75_per_second"]),
                "delta_selected_Q_left_minus_right": float(ls["Q_mean_template_equal"]) - float(rs["Q_mean_template_equal"]),
                "delta_selected_RMS_left_minus_right": float(ls["pairwise_rms_template_equal"]) - float(rs["pairwise_rms_template_equal"]),
                "delta_nominal_rate_left_minus_right": float(ls["nominal_rate_candidate_weighted"]) - float(rs["nominal_rate_candidate_weighted"]),
                "delta_sensitive_rate_left_minus_right": float(ls["sensitive_rate_candidate_weighted"]) - float(rs["sensitive_rate_candidate_weighted"]),
                "left_seconds": float(cost_lookup[(replicate, left)]["seconds"]),
                "right_seconds": float(cost_lookup[(replicate, right)]["seconds"]),
            })
    write_csv(RESULTS / "analysis" / "paired_differences_by_run.csv", paired_rows)
    paired_summary = []
    for group, left, right in contrasts:
        rows = [row for row in paired_rows if row["group"] == group and row["left"] == left and row["right"] == right]
        for metric in [key for key in rows[0] if key.startswith("delta_")]:
            values = np.asarray([float(row[metric]) for row in rows])
            paired_summary.append({"group": group, "left": left, "right": right, "metric": metric,
                                   "mean_difference": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                                   "left_higher_runs": int(np.sum(values > 0)), "ties": int(np.sum(values == 0)),
                                   "right_higher_runs": int(np.sum(values < 0))})
    write_csv(RESULTS / "analysis" / "paired_differences_five_run_summary.csv", paired_summary)
    summary = []
    for (group, arm), rows in defaultdict(list, {key: [r for r in supply_rows if (r["group"], r["arm"]) == key]
                                                  for key in {(r["group"], r["arm"]) for r in supply_rows}}).items():
        for metric in ("audited_unique_candidates", "Q_ge_0_75", "supports", "Q_mean_candidate_weighted", "Q_max", "new_unique_vs_baseline", "new_Q_ge_0_75_vs_baseline", "new_unique_per_second", "new_Q_ge_0_75_per_second"):
            values = [float(row[metric]) for row in rows if metric in row]
            if values:
                summary.append({"section": "supply", "group": group, "arm": arm, "metric": metric,
                                "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))})
    for (group, arm), rows in defaultdict(list, {key: [r for r in selection_rows if (r["group"], r["arm"]) == key]
                                                  for key in {(r["group"], r["arm"]) for r in selection_rows}}).items():
        for metric in ("selected_n", "Q_mean_template_equal", "pairwise_rms_template_equal",
                       "nominal_rate_candidate_weighted", "sensitive_rate_candidate_weighted"):
            values = [float(row[metric]) for row in rows]
            summary.append({"section": "fixed_selection", "group": group, "arm": arm, "metric": metric,
                            "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))})
    write_csv(RESULTS / "analysis" / "five_run_summary.csv", summary)
    print(json.dumps({"supply_rows": len(supply_rows), "selection_rows": len(selection_rows),
                      "curve_rows": len(curve_rows), "cost_rows": len(cost_rows), "paired_rows": len(paired_rows),
                      "paired_summary_rows": len(paired_summary), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
