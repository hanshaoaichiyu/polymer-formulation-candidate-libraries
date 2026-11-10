from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import frozen_core
import mpg_common


WORK = Path(__file__).resolve().parents[1]
INPUTS = WORK / "inputs"
RESULTS = WORK / "results"
ROOT_PROJECT = WORK.parent
HISTORICAL_G_R = ROOT_PROJECT / "PES_constraint_synthesis_no_anchor_20261010_v1" / "results_raw" / "formal_v1" / "industrial_pp_100"
HISTORICAL_MPG = ROOT_PROJECT / "CMS_v35_three_seed_ga_routes_20261104_v1" / "results" / "formal"
ROUTE_INFO = {
    "Generator_GA": {"source": "G", "offset": 0, "source_code": 1, "engine": "torch"},
    "R_GA": {"source": "R", "offset": 1, "source_code": 2, "engine": "torch"},
    "MPG_GA": {"source": "MPG", "offset": 3, "source_code": 8, "engine": "numpy"},
}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def coordinate_units(problem, design: np.ndarray) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    if getattr(problem, "space", "simplex") == "simplex":
        return np.rint(design / np.asarray(problem.step)).astype(np.int32)
    return np.rint((design - np.asarray(problem.lower)) / np.asarray(problem.step)).astype(np.int32)


def build_mpg_pool(problem: mpg_common.Problem, replicate: int) -> dict[str, np.ndarray]:
    compact = load_npz(INPUTS / "best_feasible_per_successful_start.npz")
    rows = compact["replicate"] == replicate
    original = np.asarray(compact["design"][rows], dtype=np.float64)
    structures = np.asarray(compact["structure"][rows], dtype=int)
    design = np.rint(original / problem.step) * problem.step
    replay = problem.audit32(design, structures)
    archived_feasible = np.asarray(compact["feasible"][rows], dtype=bool)
    if not np.all(archived_feasible):
        raise AssertionError("MPG representative archive contains non-feasible row")
    return {
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
        "numerical_boundary_flag_change": archived_feasible & ~replay["feasible"],
    }


def load_route(replicate: int, route: str, device):
    info = ROUTE_INFO[route]
    if info["engine"] == "torch":
        problem = frozen_core.Problem("industrial_pp_100")
        root = INPUTS / "historical_formal" / f"replicate_{replicate}"
        pool = load_npz(root / info["source"] / "candidates.npz")
        selected = np.load(root / f"{info['source']}_selected.npy", allow_pickle=False)
        audit = lambda design, structures: problem.audit(design, structures, device)
        mutate = frozen_core.mutate
        protocol = frozen_core.P
        historical = HISTORICAL_G_R / f"replicate_{replicate}" / "methods" / ("G_GA" if route == "Generator_GA" else "R_GA")
    else:
        problem = mpg_common.Problem()
        pool = build_mpg_pool(problem, replicate)
        selected = np.load(INPUTS / "historical_formal" / f"replicate_{replicate}" / "MPG_selected.npy", allow_pickle=False)
        recomputed = mpg_common.select_seeds(problem, pool)
        if not np.array_equal(selected, recomputed):
            raise AssertionError(f"MPG selected identity mismatch in run {replicate}")
        audit = problem.audit32
        mutate = mpg_common.mutate
        protocol = {"ga": mpg_common.PROTOCOL["ga"]}
        historical = HISTORICAL_MPG / f"replicate_{replicate}" / "MPG_GA"
    return problem, pool, np.asarray(selected, dtype=int), audit, mutate, protocol, historical


def snapshot_append(store: dict[str, list], event: int, phase: int, lineage: int, update: int,
                    ids: list[int], designs: list[np.ndarray], qualities: list[float], metadata: dict[int, tuple], problem) -> None:
    store["event"].append(event)
    store["phase"].append(phase)
    store["lineage"].append(lineage)
    store["update"].append(update)
    store["ptr"].append(store["ptr"][-1] + len(ids))
    for candidate_id, design, quality in zip(ids, designs, qualities):
        predicted, structural, window, feasible = metadata[candidate_id]
        store["candidate_id"].append(candidate_id)
        store["design"].append(np.asarray(design, dtype=np.float64))
        store["units"].append(coordinate_units(problem, np.asarray(design)[None, :])[0])
        store["quality"].append(float(quality))
        store["predicted"].append(np.asarray(predicted, dtype=np.float32))
        store["structural"].append(bool(structural))
        store["window"].append(bool(window))
        store["feasible"].append(bool(feasible))


def run_traced(problem, pool, selected, replicate: int, route: str, audit_fn, mutate_fn, protocol: dict,
               budget: int, capture: bool = True):
    info = ROUTE_INFO[route]
    seed = 3101000 + replicate * 100 + int(info["offset"])
    rng = np.random.default_rng(seed)
    rng_start = jsonable(rng.bit_generator.state)
    quotas = np.full(len(selected), budget // len(selected), dtype=int)
    quotas[: budget % len(selected)] += 1
    ga = protocol["ga"]
    elite_population = int(ga["elite_population"])
    crossover_probability = float(ga["crossover_probability"])
    crossover_range = ga.get("crossover_weight_range", [0.15, 0.85])
    mutation_probability = float(ga["mutation_probability"])
    batch_size = int(ga.get("batch_size", 64))

    proposals = {key: [] for key in [
        "design", "units", "structure", "source", "lineage", "update", "proposal_in_update", "candidate_id",
        "parent_primary_id", "parent_secondary_id", "crossover", "crossover_weight", "mutation",
        "predicted", "quality", "nearest", "structural", "window", "feasible", "entered_elite"
    ]}
    snapshots = {key: [] for key in ["event", "phase", "lineage", "update", "candidate_id", "design", "units", "quality", "predicted", "structural", "window", "feasible"]}
    snapshots["ptr"] = [0]
    initial_records = []
    final_records = []
    line_rows = []
    rng_rows = []
    accepted_ids: list[int] = []
    total = 0
    event = 0
    started = time.time()

    for lineage, (pool_index, quota) in enumerate(zip(selected, quotas)):
        structure = int(pool["structure"][pool_index])
        source = int(pool["source"][pool_index])
        initial_id = -(lineage + 1)
        population = [np.asarray(pool["design"][pool_index], dtype=np.float64).copy()]
        population_ids = [initial_id]
        qualities = [float(pool["task_quality"][pool_index])]
        metadata = {
            initial_id: (
                np.asarray(pool["predicted_properties"][pool_index], dtype=np.float32),
                bool(pool["structural_ok"][pool_index]),
                bool(pool["window_ok"][pool_index]),
                bool(pool["feasible"][pool_index]),
            )
        }
        initial_records.append({
            "replicate": replicate, "route": route, "lineage": lineage, "pool_index": int(pool_index),
            "structure": structure, "source": source, "candidate_id": initial_id,
            "quality": qualities[0], "design": population[0].copy(),
            "predicted": metadata[initial_id][0].copy(), "structural": metadata[initial_id][1],
            "window": metadata[initial_id][2], "feasible": metadata[initial_id][3],
        })
        rng_rows.append({"lineage": lineage, "phase": "start", "state": jsonable(rng.bit_generator.state)})
        line_accepted = 0
        used = 0
        update = 0
        while used < int(quota):
            size = min(batch_size, int(quota) - used)
            if capture:
                snapshot_append(snapshots, event, 0, lineage, update, population_ids, population, qualities, metadata, problem)
            children = []
            parent_primary = []
            parent_secondary = []
            crossover_used = []
            crossover_weight = []
            mutation_used = []
            for _ in range(size):
                primary_position = int(rng.integers(len(population)))
                child = population[primary_position].copy()
                primary_id = population_ids[primary_position]
                secondary_id = -2147483648
                crossed = False
                weight = np.nan
                if len(population) > 1 and rng.random() < crossover_probability:
                    weight = float(rng.uniform(*crossover_range))
                    secondary_position = int(rng.integers(len(population)))
                    secondary_id = population_ids[secondary_position]
                    child = weight * child + (1.0 - weight) * population[secondary_position]
                    crossed = True
                mutated = bool(rng.random() < mutation_probability)
                if mutated:
                    child = mutate_fn(problem, child, structure, rng)
                else:
                    child = problem.project(child, structure)
                children.append(child)
                parent_primary.append(primary_id)
                parent_secondary.append(secondary_id)
                crossover_used.append(crossed)
                crossover_weight.append(weight)
                mutation_used.append(mutated)
            batch = np.asarray(children, dtype=np.float64)
            structures = np.full(size, structure, dtype=int)
            audited = audit_fn(batch, structures)
            proposal_start = total
            batch_ids = list(range(proposal_start, proposal_start + size))
            used += size
            total += size
            feasible_local = set(int(index) for index in np.flatnonzero(audited["feasible"]))
            for local, candidate_id in enumerate(batch_ids):
                metadata[candidate_id] = (
                    np.asarray(audited["predicted_properties"][local], dtype=np.float32),
                    bool(audited["structural_ok"][local]), bool(audited["window_ok"][local]), bool(audited["feasible"][local]),
                )
                proposals["design"].append(batch[local])
                proposals["units"].append(coordinate_units(problem, batch[local:local + 1])[0])
                proposals["structure"].append(structure)
                proposals["source"].append(source)
                proposals["lineage"].append(lineage)
                proposals["update"].append(update)
                proposals["proposal_in_update"].append(local)
                proposals["candidate_id"].append(candidate_id)
                proposals["parent_primary_id"].append(parent_primary[local])
                proposals["parent_secondary_id"].append(parent_secondary[local])
                proposals["crossover"].append(crossover_used[local])
                proposals["crossover_weight"].append(crossover_weight[local])
                proposals["mutation"].append(mutation_used[local])
                proposals["predicted"].append(audited["predicted_properties"][local])
                proposals["quality"].append(audited["task_quality"][local])
                proposals["nearest"].append(audited["nearest_database_distance"][local])
                proposals["structural"].append(audited["structural_ok"][local])
                proposals["window"].append(audited["window_ok"][local])
                proposals["feasible"].append(audited["feasible"][local])
                proposals["entered_elite"].append(False)
            for local in np.flatnonzero(audited["feasible"]):
                local = int(local)
                candidate_id = batch_ids[local]
                accepted_ids.append(candidate_id)
                population.append(batch[local])
                population_ids.append(candidate_id)
                qualities.append(float(audited["task_quality"][local]))
                line_accepted += 1
            order = np.argsort(-np.asarray(qualities))[:elite_population]
            population = [population[int(position)] for position in order]
            population_ids = [population_ids[int(position)] for position in order]
            qualities = [qualities[int(position)] for position in order]
            for candidate_id in population_ids:
                if candidate_id >= proposal_start:
                    proposals["entered_elite"][candidate_id] = True
            if capture:
                snapshot_append(snapshots, event, 1, lineage, update, population_ids, population, qualities, metadata, problem)
            event += 1
            update += 1
        initial_quality = float(pool["task_quality"][pool_index])
        line_rows.append({
            "replicate": replicate, "route": route, "lineage": lineage, "pool_index": int(pool_index),
            "structure": structure, "source": source, "budget": int(quota), "updates": update,
            "accepted": line_accepted, "initial_quality": initial_quality, "best_quality": float(max(qualities)),
            "quality_gain": float(max(qualities) - initial_quality), "termination_reason": "lineage_quota_exhausted",
        })
        for candidate_id, design, quality in zip(population_ids, population, qualities):
            predicted, structural, window, feasible = metadata[candidate_id]
            final_records.append({
                "replicate": replicate, "route": route, "lineage": lineage, "structure": structure,
                "source": source, "candidate_id": candidate_id, "quality": float(quality), "design": np.asarray(design).copy(),
                "predicted": np.asarray(predicted).copy(), "structural": structural, "window": window, "feasible": feasible,
            })
        rng_rows.append({"lineage": lineage, "phase": "end", "state": jsonable(rng.bit_generator.state)})

    final_ids = {int(row["candidate_id"]) for row in final_records if int(row["candidate_id"]) >= 0}
    retained_final = np.asarray([int(candidate_id) in final_ids for candidate_id in proposals["candidate_id"]], dtype=bool)
    accepted_mask = np.asarray(proposals["feasible"], dtype=bool)
    proposal_arrays = {
        "design": np.asarray(proposals["design"], dtype=np.float64),
        "coordinate_units": np.asarray(proposals["units"], dtype=np.int32),
        "structure": np.asarray(proposals["structure"], dtype=np.int16),
        "source": np.asarray(proposals["source"], dtype=np.uint8),
        "lineage": np.asarray(proposals["lineage"], dtype=np.int16),
        "update": np.asarray(proposals["update"], dtype=np.int16),
        "proposal_in_update": np.asarray(proposals["proposal_in_update"], dtype=np.int16),
        "candidate_id": np.asarray(proposals["candidate_id"], dtype=np.int32),
        "parent_primary_id": np.asarray(proposals["parent_primary_id"], dtype=np.int32),
        "parent_secondary_id": np.asarray(proposals["parent_secondary_id"], dtype=np.int32),
        "crossover": np.asarray(proposals["crossover"], dtype=bool),
        "crossover_weight": np.asarray(proposals["crossover_weight"], dtype=np.float64),
        "mutation": np.asarray(proposals["mutation"], dtype=bool),
        "predicted_properties": np.asarray(proposals["predicted"], dtype=np.float32),
        "task_quality": np.asarray(proposals["quality"], dtype=np.float32),
        "nearest_database_distance": np.asarray(proposals["nearest"], dtype=np.float32),
        "structural_ok": np.asarray(proposals["structural"], dtype=bool),
        "window_ok": np.asarray(proposals["window"], dtype=bool),
        "feasible": accepted_mask,
        "entered_elite": np.asarray(proposals["entered_elite"], dtype=bool),
        "retained_final": retained_final,
    }
    accepted = {key: value[accepted_mask] for key, value in proposal_arrays.items() if len(value) == len(accepted_mask)}
    snapshot_arrays = {
        "snapshot_ptr": np.asarray(snapshots["ptr"], dtype=np.int64),
        "snapshot_event": np.asarray(snapshots["event"], dtype=np.int32),
        "snapshot_phase": np.asarray(snapshots["phase"], dtype=np.uint8),
        "snapshot_lineage": np.asarray(snapshots["lineage"], dtype=np.int16),
        "snapshot_update": np.asarray(snapshots["update"], dtype=np.int16),
        "candidate_id": np.asarray(snapshots["candidate_id"], dtype=np.int32),
        "design": np.asarray(snapshots["design"], dtype=np.float64),
        "coordinate_units": np.asarray(snapshots["units"], dtype=np.int32),
        "task_quality": np.asarray(snapshots["quality"], dtype=np.float32),
        "predicted_properties": np.asarray(snapshots["predicted"], dtype=np.float32),
        "structural_ok": np.asarray(snapshots["structural"], dtype=bool),
        "window_ok": np.asarray(snapshots["window"], dtype=bool),
        "feasible": np.asarray(snapshots["feasible"], dtype=bool),
    }
    initial_arrays = records_to_arrays(initial_records, problem)
    final_arrays = records_to_arrays(final_records, problem)
    receipt = {
        "replicate": replicate, "route": route, "seed": seed, "attempts": total,
        "accepted": int(accepted_mask.sum()), "accepted_rate": float(accepted_mask.mean()),
        "lineages": len(selected), "real_update_events": event,
        "improved_lineages": int(sum(row["quality_gain"] > 0 for row in line_rows)),
        "final_elite_members": len(final_records), "seconds": time.time() - started,
        "termination_reason": "all_lineage_quotas_exhausted", "random_state_start": rng_start,
        "random_state_final": jsonable(rng.bit_generator.state), "logging_consumed_random_numbers": False,
    }
    return proposal_arrays, accepted, snapshot_arrays, initial_arrays, final_arrays, line_rows, rng_rows, receipt


def records_to_arrays(records: list[dict], problem) -> dict[str, np.ndarray]:
    designs = np.asarray([row["design"] for row in records], dtype=np.float64)
    return {
        "replicate": np.asarray([row["replicate"] for row in records], dtype=np.uint8),
        "lineage": np.asarray([row["lineage"] for row in records], dtype=np.int16),
        "structure": np.asarray([row["structure"] for row in records], dtype=np.int16),
        "source": np.asarray([row["source"] for row in records], dtype=np.uint8),
        "candidate_id": np.asarray([row["candidate_id"] for row in records], dtype=np.int32),
        "design": designs,
        "coordinate_units": coordinate_units(problem, designs),
        "task_quality": np.asarray([row["quality"] for row in records], dtype=np.float32),
        "predicted_properties": np.asarray([row["predicted"] for row in records], dtype=np.float32),
        "structural_ok": np.asarray([row["structural"] for row in records], dtype=bool),
        "window_ok": np.asarray([row["window"] for row in records], dtype=bool),
        "feasible": np.asarray([row["feasible"] for row in records], dtype=bool),
    }


def baseline_short(problem, pool, selected, replicate, route, audit_fn, mutate_fn, protocol, budget):
    # Exact original loop without logging. This is deliberately separate from run_traced.
    info = ROUTE_INFO[route]
    seed = 3101000 + replicate * 100 + int(info["offset"])
    rng = np.random.default_rng(seed)
    quotas = np.full(len(selected), budget // len(selected), dtype=int)
    quotas[: budget % len(selected)] += 1
    ga = protocol["ga"]
    accepted_x, accepted_s, accepted_source = [], [], []
    for pool_index, quota in zip(selected, quotas):
        structure = int(pool["structure"][pool_index])
        source = int(pool["source"][pool_index])
        population = [pool["design"][pool_index].copy()]
        qualities = [float(pool["task_quality"][pool_index])]
        used = 0
        while used < quota:
            size = min(int(ga.get("batch_size", 64)), int(quota - used))
            children = []
            for _ in range(size):
                child = population[int(rng.integers(len(population)))].copy()
                if len(population) > 1 and rng.random() < float(ga["crossover_probability"]):
                    weight = rng.uniform(*ga.get("crossover_weight_range", [0.15, 0.85]))
                    child = weight * child + (1.0 - weight) * population[int(rng.integers(len(population)))]
                if rng.random() < float(ga["mutation_probability"]):
                    child = mutate_fn(problem, child, structure, rng)
                else:
                    child = problem.project(child, structure)
                children.append(child)
            children = np.asarray(children)
            structures = np.full(size, structure, dtype=int)
            audited = audit_fn(children, structures)
            used += size
            for local in np.flatnonzero(audited["feasible"]):
                accepted_x.append(children[local]); accepted_s.append(structure); accepted_source.append(source)
                population.append(children[local]); qualities.append(float(audited["task_quality"][local]))
            order = np.argsort(-np.asarray(qualities))[: int(ga["elite_population"])]
            population = [population[int(position)] for position in order]
            qualities = [qualities[int(position)] for position in order]
    return np.asarray(accepted_x, dtype=np.float64).reshape((-1, problem.n)), np.asarray(accepted_s), np.asarray(accepted_source)


def gate(device) -> None:
    output = RESULTS / "gate"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for route in ROUTE_INFO:
        problem, pool, selected, audit_fn, mutate_fn, protocol, _ = load_route(1, route, device)
        selected = selected[:4]
        budget = 512
        baseline = baseline_short(problem, pool, selected, 1, route, audit_fn, mutate_fn, protocol, budget)
        traced = run_traced(problem, pool, selected, 1, route, audit_fn, mutate_fn, protocol, budget, capture=True)
        accepted = traced[1]
        exact_design = np.array_equal(baseline[0], accepted["design"])
        exact_structure = np.array_equal(baseline[1], accepted["structure"])
        exact_source = np.array_equal(baseline[2], accepted["source"])
        rows.append({"route": route, "budget": budget, "selected_lineages": len(selected),
                     "accepted": len(accepted["design"]), "exact_design": exact_design,
                     "exact_structure": exact_structure, "exact_source": exact_source,
                     "passed": exact_design and exact_structure and exact_source})
    result = {"ok": all(row["passed"] for row in rows), "device": str(device), "routes": rows}
    atomic_json(output / "GATE_RESULT.json", result)
    if not result["ok"]:
        raise SystemExit("trace non-interference gate failed")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_formal(replicate: int, route: str, device) -> None:
    output = RESULTS / "formal" / f"replicate_{replicate}" / route
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    problem, pool, selected, audit_fn, mutate_fn, protocol, historical = load_route(replicate, route, device)
    proposal, accepted, snapshots, initial, final, line_rows, rng_rows, receipt = run_traced(
        problem, pool, selected, replicate, route, audit_fn, mutate_fn, protocol, 40000, capture=True
    )
    save_npz(output / "proposals.npz", **proposal)
    save_npz(output / "accepted_replay.npz", **accepted)
    save_npz(output / "population_events.npz", **snapshots)
    save_npz(output / "initial_population.npz", **initial)
    save_npz(output / "final_elite_population.npz", **final)
    write_csv(output / "lineages.csv", line_rows)
    atomic_json(output / "rng_states.json", rng_rows)
    historical_accepted = historical / ("search/accepted.npz" if route != "MPG_GA" else "accepted.npz")
    old = load_npz(historical_accepted)
    same_count = len(old["design"]) == len(accepted["design"])
    comparison = {
        "historical_path": str(historical_accepted), "historical_sha256": sha256(historical_accepted),
        "same_accepted_count": same_count, "historical_accepted": len(old["design"]), "replay_accepted": len(accepted["design"]),
        "exact_design_order": bool(same_count and np.array_equal(old["design"], accepted["design"])),
        "exact_structure_order": bool(same_count and np.array_equal(old["structure"], accepted["structure"])),
        "exact_source_order": bool(same_count and np.array_equal(old["source"], accepted["source"])),
        "max_abs_quality_difference": float(np.max(np.abs(old["task_quality"] - accepted["task_quality"]))) if same_count else None,
        "max_abs_prediction_difference": float(np.max(np.abs(old["predicted_properties"] - accepted["predicted_properties"]))) if same_count else None,
    }
    comparison["strict_identity_reproduction"] = bool(
        comparison["exact_design_order"] and comparison["exact_structure_order"] and comparison["exact_source_order"]
    )
    atomic_json(output / "historical_consistency.json", comparison)
    environment = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "numpy": np.__version__, "device": str(device), "route_engine": ROUTE_INFO[route]["engine"],
        "torch": getattr(frozen_core.torch, "__version__", None),
        "cuda_available": bool(frozen_core.torch.cuda.is_available()),
        "cuda_device": frozen_core.torch.cuda.get_device_name(0) if frozen_core.torch.cuda.is_available() else None,
    }
    atomic_json(output / "environment.json", environment)
    receipt.update({
        "case": "industrial_pp_100", "protocol_unchanged": True, "logging_only_change": True,
        "historical_strict_identity_reproduction": comparison["strict_identity_reproduction"],
        "input_identity": {
            "trace_protocol": sha256(WORK / "configs" / "TRACE_PROTOCOL.json"),
            "original_protocol": sha256(WORK / "configs" / "protocol_v1.json"),
            "mpg_protocol": sha256(WORK / "configs" / "FROZEN_PROTOCOL.json"),
            "selected_indices": sha256(INPUTS / "historical_formal" / f"replicate_{replicate}" / (f"{ROUTE_INFO[route]['source']}_selected.npy" if route != "MPG_GA" else "MPG_selected.npy")),
        },
    })
    atomic_json(output / "COMPLETED.json", receipt)
    print(json.dumps({"completion": receipt, "historical_consistency": comparison}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--replicate", type=int, choices=range(1, 6))
    parser.add_argument("--route", choices=ROUTE_INFO)
    args = parser.parse_args()
    device = frozen_core.torch.device("cuda" if frozen_core.torch.cuda.is_available() else "cpu")
    if args.gate:
        gate(device)
        return
    if args.replicate is None or args.route is None:
        parser.error("formal mode requires --replicate and --route")
    run_formal(args.replicate, args.route, device)


if __name__ == "__main__":
    main()
