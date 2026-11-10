from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
WORK = PACKAGE / "full_archive_recomputed"
PROJECT = PACKAGE / "full_archive"
SOURCE = PROJECT / "PES_constraint_synthesis_no_anchor_20261010_v1"
V11 = PROJECT / "PES_v11_review_supplement_20261012_v1"
FORMAL = SOURCE / "results_raw" / "formal_v1"
PROTOCOL_PATH = PACKAGE / "configs" / "FROZEN_PROTOCOL.json"
FREEZE_PATH = PACKAGE / "provenance" / "PROTOCOL_FROZEN.json"
INPUT_MANIFEST_PATH = PACKAGE / "full_archive_workflow" / "FULL_ARCHIVE_MANIFEST.json"

sys.path.insert(0, str(SOURCE / "scripts"))
import core  # noqa: E402


PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CASES = PROTOCOL["source"]["cases"]
REPLICATES = PROTOCOL["source"]["replicates"]
ROUTES = PROTOCOL["source"]["routes"]
ROUTE_BITS = PROTOCOL["source"]["route_membership_bits"]
ARMS = PROTOCOL["source"]["comparison_arms"]
ARM_MASKS = {
    "G_GA": 1,
    "R_GA": 2,
    "GR_GA": 4,
    "G_GA+R_GA": 3,
    "G_GA+GR_GA": 5,
    "R_GA+GR_GA": 6,
    "ALL": 7,
}
ROUTE_INDEX = {route: idx for idx, route in enumerate(ROUTES)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None, None
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def precision_arrays(problem: core.Problem) -> tuple[np.ndarray, np.ndarray]:
    if problem.space == "simplex":
        return np.zeros(problem.n, dtype=float), np.asarray(problem.step, dtype=float)
    return np.asarray(problem.lower, dtype=float), np.asarray(problem.step, dtype=float)


def membership_label(bits: int) -> str:
    names = [route for route in ROUTES if bits & ROUTE_BITS[route]]
    return "+".join(names)


def arm_ids(membership: np.ndarray, arm: str) -> np.ndarray:
    return np.flatnonzero((membership & ARM_MASKS[arm]) > 0)


def key_hash(structure: int, integer_row: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray([structure], dtype="<i8").tobytes())
    h.update(np.asarray(integer_row, dtype="<i8").tobytes())
    return h.hexdigest()


def build_union(case: str, replicate: int, problem: core.Problem) -> dict[str, np.ndarray]:
    lower, step = precision_arrays(problem)
    table: dict[tuple, dict] = {}
    route_paths = {}
    for route in ROUTES:
        path = FORMAL / case / f"replicate_{replicate}" / "methods" / route / "library.npz"
        route_paths[route] = str(path)
        with np.load(path, allow_pickle=False) as raw:
            data = {k: np.asarray(raw[k]) for k in raw.files}
        integer = np.rint((np.asarray(data["design"], float) - lower) / step).astype(np.int64)
        for idx in range(len(integer)):
            key = (int(data["structure"][idx]), *map(int, integer[idx]))
            if key not in table:
                table[key] = {
                    "design": np.asarray(data["design"][idx], dtype=float),
                    "integer": integer[idx],
                    "structure": int(data["structure"][idx]),
                    "task_quality": float(data["task_quality"][idx]),
                    "predicted_properties": np.asarray(data["predicted_properties"][idx], dtype=float),
                    "nearest_database_distance": float(data["nearest_database_distance"][idx]),
                    "membership_bits": 0,
                    "source_bits": 0,
                    "latest_stage": 0,
                    "route_indices": np.full(3, -1, dtype=np.int64),
                }
            rec = table[key]
            if abs(rec["task_quality"] - float(data["task_quality"][idx])) > 1e-5:
                raise ValueError(f"Q mismatch {case} rep{replicate} {route} row{idx}")
            if np.max(np.abs(rec["predicted_properties"] - data["predicted_properties"][idx])) > 1e-5:
                raise ValueError(f"Prediction mismatch {case} rep{replicate} {route} row{idx}")
            rec["membership_bits"] |= ROUTE_BITS[route]
            rec["source_bits"] |= int(data["source_bits"][idx])
            rec["latest_stage"] = max(rec["latest_stage"], int(data["latest_stage"][idx]))
            rec["route_indices"][ROUTE_INDEX[route]] = idx

    values = list(table.values())
    result = {
        "design": np.asarray([v["design"] for v in values], dtype=np.float64),
        "integer_coords": np.asarray([v["integer"] for v in values], dtype=np.int64),
        "structure": np.asarray([v["structure"] for v in values], dtype=np.int32),
        "task_quality": np.asarray([v["task_quality"] for v in values], dtype=np.float64),
        "predicted_properties": np.asarray([v["predicted_properties"] for v in values], dtype=np.float64),
        "nearest_database_distance": np.asarray([v["nearest_database_distance"] for v in values], dtype=np.float64),
        "membership_bits": np.asarray([v["membership_bits"] for v in values], dtype=np.uint8),
        "source_bits": np.asarray([v["source_bits"] for v in values], dtype=np.uint8),
        "latest_stage": np.asarray([v["latest_stage"] for v in values], dtype=np.uint8),
        "route_indices": np.asarray([v["route_indices"] for v in values], dtype=np.int64),
    }
    out_dir = WORK / "results" / "full_supply" / case
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"replicate_{replicate}_union_ledger.npz", **result)
    compact_rows = []
    for idx in range(len(result["design"])):
        compact_rows.append({
            "case": case,
            "replicate": replicate,
            "union_candidate_id": idx,
            "structure": int(result["structure"][idx]),
            "coordinate_key_sha256": key_hash(int(result["structure"][idx]), result["integer_coords"][idx]),
            "membership_bits": int(result["membership_bits"][idx]),
            "membership_label": membership_label(int(result["membership_bits"][idx])),
            "source_bits_or": int(result["source_bits"][idx]),
            "latest_stage_max": int(result["latest_stage"][idx]),
            "task_quality": float(result["task_quality"][idx]),
            "nearest_database_distance": float(result["nearest_database_distance"][idx]),
            "G_GA_original_index": int(result["route_indices"][idx, 0]),
            "R_GA_original_index": int(result["route_indices"][idx, 1]),
            "GR_GA_original_index": int(result["route_indices"][idx, 2]),
        })
    write_csv(out_dir / f"replicate_{replicate}_union_membership_ledger.csv", compact_rows)
    write_json(out_dir / f"replicate_{replicate}_union_mapping.json", {
        "case": case,
        "replicate": replicate,
        "candidate_n": len(values),
        "route_order_for_route_indices": ROUTES,
        "route_paths": route_paths,
        "npz_fields": list(result),
        "membership_bits": ROUTE_BITS,
        "source_bits_note": PROTOCOL["source"]["source_bits_interpretation"],
    })
    return result


def full_supply_rows(case: str, replicate: int, union: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    q = union["task_quality"]
    structure = union["structure"]
    membership = union["membership_bits"]
    thresholds = PROTOCOL["full_supply"]["Q_thresholds"]
    for arm in ARMS:
        ids = arm_ids(membership, arm)
        row = {
            "case": case,
            "replicate": replicate,
            "arm": arm,
            "arm_route_mask": ARM_MASKS[arm],
            "unique_candidate_n": len(ids),
            "support_n": int(np.unique(structure[ids]).size),
        }
        for threshold in thresholds:
            row[f"Q_ge_{threshold:.2f}_n"] = int(np.sum(q[ids] >= threshold))
        if arm == "ALL":
            counts = Counter(map(int, membership[ids]))
            for bits in range(1, 8):
                row[f"membership_{membership_label(bits)}_n"] = counts.get(bits, 0)
        rows.append(row)
    return rows


def tie_choice(candidates: np.ndarray, primary: np.ndarray, q: np.ndarray,
               integer: np.ndarray, candidate_ids: np.ndarray) -> int:
    maximum = float(np.max(primary))
    tied = candidates[np.isclose(primary, maximum, rtol=0.0, atol=1e-12)]
    if len(tied) > 1:
        best_q = float(np.max(q[tied]))
        tied = tied[np.isclose(q[tied], best_q, rtol=0.0, atol=1e-12)]
    if len(tied) == 1:
        return int(tied[0])
    return int(min(tied, key=lambda idx: (tuple(map(int, integer[idx])), int(candidate_ids[idx]))))


def select_template_order(ids: np.ndarray, union: dict[str, np.ndarray], scale: np.ndarray,
                          cap: int) -> list[int]:
    if not len(ids):
        return []
    q = union["task_quality"]
    threshold = float(np.quantile(q[ids], 0.5))
    eligible = ids[q[ids] >= threshold]
    local_ids = np.arange(len(union["design"]), dtype=int)
    first = tie_choice(eligible, q[eligible], q, union["integer_coords"], local_ids)
    selected = [first]
    if cap == 1 or len(eligible) == 1:
        return selected
    normalized = union["design"][eligible] / scale
    pos = {int(candidate): idx for idx, candidate in enumerate(eligible)}
    active = np.ones(len(eligible), dtype=bool)
    active[pos[first]] = False
    min_distance = np.linalg.norm(normalized - normalized[pos[first]], axis=1)
    min_distance[~active] = -np.inf
    while len(selected) < min(cap, len(eligible)):
        candidate_positions = np.flatnonzero(active)
        chosen_pos = tie_choice(candidate_positions, min_distance[candidate_positions],
                                q[eligible], union["integer_coords"][eligible], eligible)
        chosen = int(eligible[chosen_pos])
        selected.append(chosen)
        active[chosen_pos] = False
        new_distance = np.linalg.norm(normalized - normalized[chosen_pos], axis=1)
        min_distance = np.minimum(min_distance, new_distance)
        min_distance[~active] = -np.inf
    return selected


def build_selection_orders(case: str, union: dict[str, np.ndarray], problem: core.Problem) -> dict:
    cap = int(PROTOCOL["selection"]["case_template_caps"][case])
    orders: dict[str, dict[int, list[int]]] = {}
    for arm in ARMS:
        ids = arm_ids(union["membership_bits"], arm)
        orders[arm] = {}
        for structure in sorted(map(int, np.unique(union["structure"][ids]))):
            group = ids[union["structure"][ids] == structure]
            orders[arm][structure] = select_template_order(group, union, np.asarray(problem.scale, float), cap)
    return orders


def common_support_selection(orders: dict) -> tuple[dict[str, list[int]], dict[int, int]]:
    common = sorted(set.intersection(*(set(orders[arm]) for arm in ARMS)))
    per_template_n = {
        structure: min(len(orders[arm][structure]) for arm in ARMS)
        for structure in common
    }
    selected = {
        arm: [candidate for structure in common for candidate in orders[arm][structure][:per_template_n[structure]]]
        for arm in ARMS
    }
    return selected, per_template_n


def full_domain_selection(orders: dict) -> tuple[dict[str, list[int]], dict[str, int], int]:
    capacity = {arm: sum(len(values) for values in orders[arm].values()) for arm in ARMS}
    target = min(capacity.values())
    selected: dict[str, list[int]] = {}
    for arm in ARMS:
        structures = sorted(orders[arm])
        cursors = {structure: 0 for structure in structures}
        result: list[int] = []
        while len(result) < target:
            progressed = False
            for structure in structures:
                cursor = cursors[structure]
                if cursor < len(orders[arm][structure]):
                    result.append(orders[arm][structure][cursor])
                    cursors[structure] += 1
                    progressed = True
                    if len(result) == target:
                        break
            if not progressed:
                break
        if len(result) != target:
            raise RuntimeError(f"Full-domain allocation failed {arm}: {len(result)} != {target}")
        selected[arm] = result
    return selected, capacity, target


def standardized_margin(problem: core.Problem, pred: np.ndarray) -> np.ndarray:
    width = problem.windows[:, 1] - problem.windows[:, 0]
    lower = (pred - problem.windows[:, 0]) / width
    upper = (problem.windows[:, 1] - pred) / width
    return np.min(np.minimum(lower, upper), axis=1)


def bootstrap_frequency(problem: core.Problem, x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if not len(x):
        return np.empty(0, dtype=float)
    xn = np.asarray(x, float) / problem.scale
    cn = np.asarray(problem.x_obs, float) / problem.scale
    d2 = np.maximum(
        np.sum(xn * xn, axis=1)[:, None] + np.sum(cn * cn, axis=1)[None, :] - 2.0 * (xn @ cn.T),
        0.0,
    )
    base_log = -d2 / (2.0 * float(problem.bandwidth) ** 2 + 1e-12)
    passed = np.zeros(len(x), dtype=np.int32)
    for count in counts:
        pred = np.full((len(x), len(problem.property_names)), np.nan, dtype=float)
        for j in range(len(problem.property_names)):
            valid = np.isfinite(problem.y_obs[:, j]) & (count > 0)
            logw = base_log[:, valid] + np.log(count[valid].astype(float))[None, :]
            maxlog = np.max(logw, axis=1, keepdims=True)
            weights = np.exp(logw - maxlog)
            pred[:, j] = (weights @ problem.y_obs[valid, j]) / np.sum(weights, axis=1)
        passed += np.all((pred >= problem.windows[:, 0]) & (pred <= problem.windows[:, 1]), axis=1)
    return passed.astype(float) / len(counts)


def distance_metrics(x: np.ndarray, scale: np.ndarray) -> dict[str, float | int | None]:
    n = len(x)
    if n < 2:
        return {"pair_n": 0, "pairwise_mean": None, "pairwise_rms": None,
                "nearest_neighbor_mean": None}
    z = np.asarray(x, float) / scale
    d2 = np.maximum(np.sum(z * z, axis=1)[:, None] + np.sum(z * z, axis=1)[None, :] - 2.0 * (z @ z.T), 0.0)
    tri = np.triu_indices(n, 1)
    pair = np.sqrt(d2[tri])
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return {
        "pair_n": len(pair),
        "pairwise_mean": float(np.mean(pair)),
        "pairwise_rms": float(np.sqrt(np.mean(pair * pair))),
        "nearest_neighbor_mean": float(np.mean(nn)),
    }


def metrics_for_selection(case: str, replicate: int, layer: str, arm: str,
                          ids: list[int], union: dict[str, np.ndarray], problem: core.Problem,
                          bootstrap_map: dict[int, float]) -> tuple[dict, list[dict]]:
    ids_a = np.asarray(ids, dtype=int)
    q = union["task_quality"][ids_a]
    margin = standardized_margin(problem, union["predicted_properties"][ids_a])
    nominal = (q >= float(PROTOCOL["screening"]["nominal"]["Q_min"])) & (
        margin >= float(PROTOCOL["screening"]["nominal"]["minimum_standardized_window_margin_min"])
    )
    boot = np.asarray([bootstrap_map[int(idx)] for idx in ids_a], dtype=float)
    sensitive = nominal & (boot >= float(PROTOCOL["screening"]["sensitivity"]["bootstrap_joint_window_pass_frequency_min"]))
    template_rows = []
    pair_sum = pair_sq_sum = 0.0
    pair_n = 0
    nn_weighted_sum = 0.0
    nn_weighted_n = 0
    for structure in sorted(map(int, np.unique(union["structure"][ids_a]))):
        local = np.flatnonzero(union["structure"][ids_a] == structure)
        local_ids = ids_a[local]
        dist = distance_metrics(union["design"][local_ids], np.asarray(problem.scale, float))
        if dist["pair_n"]:
            pair_sum += float(dist["pairwise_mean"]) * int(dist["pair_n"])
            pair_sq_sum += float(dist["pairwise_rms"]) ** 2 * int(dist["pair_n"])
            pair_n += int(dist["pair_n"])
            nn_weighted_sum += float(dist["nearest_neighbor_mean"]) * len(local_ids)
            nn_weighted_n += len(local_ids)
        values = q[local]
        template_rows.append({
            "case": case, "replicate": replicate, "layer": layer, "arm": arm,
            "structure": structure, "selected_n": len(local_ids),
            "Q_mean": float(np.mean(values)), "Q_median": float(np.median(values)),
            "Q_p10": float(np.quantile(values, 0.10)), "Q_max": float(np.max(values)),
            "minimum_margin_mean": float(np.mean(margin[local])),
            "nominal_screen_n": int(np.sum(nominal[local])),
            "nominal_screen_rate": float(np.mean(nominal[local])),
            "sensitive_screen_n": int(np.sum(sensitive[local])),
            "sensitive_screen_rate": float(np.mean(sensitive[local])),
            "bootstrap_joint_pass_frequency_mean": float(np.mean(boot[local])),
            **dist,
        })
    finite_pair = [r for r in template_rows if r["pair_n"] > 0]
    run_row = {
        "case": case, "replicate": replicate, "layer": layer, "arm": arm,
        "selected_n": len(ids_a),
        "selected_support_n": len(template_rows),
        "Q_mean_candidate_weighted": float(np.mean(q)),
        "Q_median_candidate_weighted": float(np.median(q)),
        "Q_p10_candidate_weighted": float(np.quantile(q, 0.10)),
        "Q_max": float(np.max(q)),
        "Q_mean_template_equal": float(np.mean([r["Q_mean"] for r in template_rows])),
        "nominal_n": int(np.sum(nominal)),
        "nominal_rate_candidate_weighted": float(np.mean(nominal)),
        "nominal_rate_template_equal": float(np.mean([r["nominal_screen_rate"] for r in template_rows])),
        "sensitive_n": int(np.sum(sensitive)),
        "sensitive_rate_candidate_weighted": float(np.mean(sensitive)),
        "sensitive_rate_template_equal": float(np.mean([r["sensitive_screen_rate"] for r in template_rows])),
        "pairwise_mean_template_equal": float(np.mean([r["pairwise_mean"] for r in finite_pair])) if finite_pair else None,
        "pairwise_rms_template_equal": float(np.mean([r["pairwise_rms"] for r in finite_pair])) if finite_pair else None,
        "nearest_neighbor_mean_template_equal": float(np.mean([r["nearest_neighbor_mean"] for r in finite_pair])) if finite_pair else None,
        "pairwise_mean_pair_weighted": pair_sum / pair_n if pair_n else None,
        "pairwise_rms_pair_weighted": math.sqrt(pair_sq_sum / pair_n) if pair_n else None,
        "nearest_neighbor_mean_candidate_weighted": nn_weighted_sum / nn_weighted_n if nn_weighted_n else None,
    }
    return run_row, template_rows


def summarize_runs(run_rows: list[dict]) -> list[dict]:
    keys = [
        "selected_n", "selected_support_n", "Q_mean_candidate_weighted",
        "Q_median_candidate_weighted", "Q_p10_candidate_weighted", "Q_max",
        "Q_mean_template_equal", "nominal_n", "nominal_rate_candidate_weighted",
        "nominal_rate_template_equal", "sensitive_n", "sensitive_rate_candidate_weighted",
        "sensitive_rate_template_equal", "pairwise_mean_template_equal",
        "pairwise_rms_template_equal", "nearest_neighbor_mean_template_equal",
        "pairwise_mean_pair_weighted", "pairwise_rms_pair_weighted",
        "nearest_neighbor_mean_candidate_weighted",
    ]
    groups = defaultdict(list)
    for row in run_rows:
        groups[(row["case"], row["layer"], row["arm"])].append(row)
    summary = []
    for (case, layer, arm), rows in sorted(groups.items()):
        out = {"case": case, "layer": layer, "arm": arm, "run_n": len(rows)}
        for key in keys:
            values = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
            mean, sd = mean_sd(values)
            out[f"{key}_mean"] = mean
            out[f"{key}_sample_sd"] = sd
        summary.append(out)
    return summary


def paired_contrasts(run_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metric_keys = [
        "selected_support_n", "Q_mean_candidate_weighted", "Q_p10_candidate_weighted",
        "Q_max", "nominal_rate_candidate_weighted", "nominal_rate_template_equal",
        "sensitive_rate_candidate_weighted", "sensitive_rate_template_equal",
        "pairwise_rms_template_equal", "nearest_neighbor_mean_template_equal",
    ]
    lookup = {(r["case"], r["replicate"], r["layer"], r["arm"]): r for r in run_rows}
    per_run = []
    for case in CASES:
        for replicate in REPLICATES:
            for layer in ["common_support_equal_per_template", "full_domain_equal_total"]:
                all_row = lookup[(case, replicate, layer, "ALL")]
                for comparator in ARMS[:-1]:
                    row = {"case": case, "replicate": replicate, "layer": layer,
                           "reference": "ALL", "comparator": comparator,
                           "delta_direction": "ALL_minus_comparator"}
                    comp = lookup[(case, replicate, layer, comparator)]
                    for key in metric_keys:
                        if all_row.get(key) is None or comp.get(key) is None:
                            row[f"delta_{key}"] = None
                        else:
                            row[f"delta_{key}"] = float(all_row[key]) - float(comp[key])
                    per_run.append(row)
    groups = defaultdict(list)
    for row in per_run:
        groups[(row["case"], row["layer"], row["comparator"])].append(row)
    summary = []
    for (case, layer, comparator), rows in sorted(groups.items()):
        out = {"case": case, "layer": layer, "reference": "ALL", "comparator": comparator,
               "delta_direction": "ALL_minus_comparator", "run_n": len(rows)}
        for key in metric_keys:
            values = [r[f"delta_{key}"] for r in rows if r[f"delta_{key}"] is not None]
            mean, sd = mean_sd(values)
            out[f"delta_{key}_mean"] = mean
            out[f"delta_{key}_sample_sd"] = sd
        summary.append(out)
    return per_run, summary


def route_contributions(selected_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in selected_rows:
        if row["arm"] == "ALL":
            groups[(row["case"], row["replicate"], row["layer"])].append(row)
    output = []
    for (case, replicate, layer), rows in sorted(groups.items()):
        counts = Counter(int(r["membership_bits"]) for r in rows)
        out = {"case": case, "replicate": replicate, "layer": layer,
               "selected_n": len(rows), "support_n": len(set(int(r["structure"]) for r in rows))}
        for bits in range(1, 8):
            out[f"membership_{membership_label(bits)}_n"] = counts.get(bits, 0)
        out["exclusive_any_route_n"] = counts.get(1, 0) + counts.get(2, 0) + counts.get(4, 0)
        out["shared_two_or_three_routes_n"] = sum(counts.get(bits, 0) for bits in [3, 5, 6, 7])
        output.append(out)
    return output


def selected_ledger_rows(case: str, replicate: int, layer: str, selections: dict[str, list[int]],
                         union: dict[str, np.ndarray], bootstrap_map: dict[int, float],
                         problem: core.Problem) -> list[dict]:
    rows = []
    qmin = float(PROTOCOL["screening"]["nominal"]["Q_min"])
    margin_min = float(PROTOCOL["screening"]["nominal"]["minimum_standardized_window_margin_min"])
    stable_min = float(PROTOCOL["screening"]["sensitivity"]["bootstrap_joint_window_pass_frequency_min"])
    for arm in ARMS:
        ids = selections[arm]
        by_structure_rank = Counter()
        margin = standardized_margin(problem, union["predicted_properties"][np.asarray(ids, int)])
        for rank, (candidate, candidate_margin) in enumerate(zip(ids, margin), start=1):
            structure = int(union["structure"][candidate])
            by_structure_rank[structure] += 1
            q = float(union["task_quality"][candidate])
            boot = float(bootstrap_map[int(candidate)])
            nominal = q >= qmin and candidate_margin >= margin_min
            rows.append({
                "case": case, "replicate": replicate, "layer": layer, "arm": arm,
                "selection_rank_global": rank,
                "selection_rank_within_structure": by_structure_rank[structure],
                "union_candidate_id": int(candidate), "structure": structure,
                "coordinate_key_sha256": key_hash(structure, union["integer_coords"][candidate]),
                "membership_bits": int(union["membership_bits"][candidate]),
                "membership_label": membership_label(int(union["membership_bits"][candidate])),
                "source_bits_or": int(union["source_bits"][candidate]),
                "latest_stage_max": int(union["latest_stage"][candidate]),
                "task_quality": q,
                "minimum_standardized_window_margin": float(candidate_margin),
                "bootstrap_joint_window_pass_frequency": boot,
                "nominal_screen": nominal,
                "sensitive_screen": bool(nominal and boot >= stable_min),
                "nearest_database_distance": float(union["nearest_database_distance"][candidate]),
                "G_GA_original_index": int(union["route_indices"][candidate, 0]),
                "R_GA_original_index": int(union["route_indices"][candidate, 1]),
                "GR_GA_original_index": int(union["route_indices"][candidate, 2]),
            })
    return rows


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("Frozen protocol hash mismatch")
    manifest = json.loads(INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    before_hashes = {str(PACKAGE / item["package_relative_path"]): item["sha256"] for item in manifest["files"]}
    for path, expected in before_hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Read-only input changed before analysis: {path}")

    supply_rows: list[dict] = []
    selected_rows: list[dict] = []
    run_metric_rows: list[dict] = []
    template_metric_rows: list[dict] = []
    selection_capacity_rows: list[dict] = []
    bootstrap_rows: list[dict] = []

    for case in CASES:
        problem = core.Problem(case)
        counts_path = (V11 / "results_summary" / "case_modules" / case /
                       f"B_bootstrap_indices_{case}.npz")
        with np.load(counts_path, allow_pickle=False) as b:
            bootstrap_counts = np.asarray(b["counts"], dtype=np.int32)
        for replicate in REPLICATES:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] {case} replicate {replicate}", flush=True)
            union = build_union(case, replicate, problem)
            supply_rows.extend(full_supply_rows(case, replicate, union))
            orders = build_selection_orders(case, union, problem)
            common_selected, common_n = common_support_selection(orders)
            full_selected, capacity, target = full_domain_selection(orders)
            for arm in ARMS:
                selection_capacity_rows.append({
                    "case": case, "replicate": replicate, "arm": arm,
                    "common_support_n": len(common_n),
                    "common_selected_n": len(common_selected[arm]),
                    "common_per_template_min": min(common_n.values()) if common_n else 0,
                    "common_per_template_max": max(common_n.values()) if common_n else 0,
                    "full_domain_capacity": capacity[arm],
                    "full_domain_target_n": target,
                    "full_domain_selected_n": len(full_selected[arm]),
                    "full_domain_unfilled_n": target - len(full_selected[arm]),
                })

            unique_selected = sorted(set(candidate for values in common_selected.values() for candidate in values) |
                                     set(candidate for values in full_selected.values() for candidate in values))
            frequencies = bootstrap_frequency(problem, union["design"][unique_selected], bootstrap_counts)
            bootstrap_map = {candidate: float(freq) for candidate, freq in zip(unique_selected, frequencies)}
            for candidate, freq in zip(unique_selected, frequencies):
                bootstrap_rows.append({
                    "case": case, "replicate": replicate,
                    "union_candidate_id": candidate,
                    "coordinate_key_sha256": key_hash(int(union["structure"][candidate]), union["integer_coords"][candidate]),
                    "bootstrap_repeats": len(bootstrap_counts),
                    "bootstrap_joint_window_pass_frequency": float(freq),
                })

            layers = {
                "common_support_equal_per_template": common_selected,
                "full_domain_equal_total": full_selected,
            }
            index_payload = {}
            for layer, selections in layers.items():
                index_payload[layer] = {arm: np.asarray(ids, dtype=np.int64) for arm, ids in selections.items()}
                selected_rows.extend(selected_ledger_rows(case, replicate, layer, selections, union,
                                                           bootstrap_map, problem))
                for arm in ARMS:
                    run_row, template_rows = metrics_for_selection(case, replicate, layer, arm,
                                                                    selections[arm], union, problem,
                                                                    bootstrap_map)
                    run_metric_rows.append(run_row)
                    template_metric_rows.extend(template_rows)
            index_dir = WORK / "results" / "selections" / case
            index_dir.mkdir(parents=True, exist_ok=True)
            flat = {}
            for layer, arm_map in index_payload.items():
                for arm, values in arm_map.items():
                    flat[f"{layer}__{arm.replace('+', '_plus_')}"] = values
            np.savez_compressed(index_dir / f"replicate_{replicate}_selected_union_indices.npz", **flat)

    write_csv(WORK / "results" / "full_supply" / "full_supply_arm_metrics.csv", supply_rows)
    write_csv(WORK / "results" / "selections" / "selection_capacity_and_counts.csv", selection_capacity_rows)
    write_csv(WORK / "results" / "selections" / "selected_candidate_ledger.csv", selected_rows)
    write_csv(WORK / "results" / "selections" / "selected_bootstrap_frequency.csv", bootstrap_rows)
    write_csv(WORK / "results" / "metrics" / "per_template_metrics.csv", template_metric_rows)
    write_csv(WORK / "results" / "metrics" / "per_run_metrics.csv", run_metric_rows)
    summary_rows = summarize_runs(run_metric_rows)
    write_csv(WORK / "results" / "metrics" / "five_run_summary.csv", summary_rows)
    contrast_rows, contrast_summary = paired_contrasts(run_metric_rows)
    write_csv(WORK / "results" / "contributions" / "paired_ALL_minus_comparator_per_run.csv", contrast_rows)
    write_csv(WORK / "results" / "contributions" / "paired_ALL_minus_comparator_summary.csv", contrast_summary)
    contribution_rows = route_contributions(selected_rows)
    write_csv(WORK / "results" / "contributions" / "ALL_selected_route_membership.csv", contribution_rows)

    after_changed = []
    for path, expected in before_hashes.items():
        if sha256(Path(path)) != expected:
            after_changed.append(path)
    if after_changed:
        raise RuntimeError("Read-only inputs changed during analysis: " + "; ".join(after_changed))

    output_files = []
    for path in sorted((WORK / "results").rglob("*")):
        if path.is_file():
            output_files.append({"path": str(path.relative_to(WORK)).replace("\\", "/"),
                                 "bytes": path.stat().st_size, "sha256": sha256(path)})
    output_manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "read_only_input_manifest_sha256": sha256(INPUT_MANIFEST_PATH),
        "input_hashes_unchanged": True,
        "result_file_count": len(output_files),
        "files": output_files,
    }
    write_json(WORK / "status" / "OUTPUT_MANIFEST.json", output_manifest)
    completed = {
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "cases": CASES,
        "replicates": REPLICATES,
        "arms": ARMS,
        "libraries_used": 45,
        "formal_training_or_optimization_rerun": False,
        "selected_candidate_ledger_rows": len(selected_rows),
        "per_template_metric_rows": len(template_metric_rows),
        "per_run_metric_rows": len(run_metric_rows),
        "bootstrap_selected_unique_rows": len(bootstrap_rows),
        "equal_budget_truncation_performed": False,
        "input_hashes_unchanged": True,
    }
    write_json(WORK / "status" / "COMPLETED.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
