from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parent
INPUTS = WORK / "inputs"
RESULTS = WORK / "results"
PROTOCOL = json.loads((WORK / "configs" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
GENERATION = json.loads((INPUTS / "generation_protocol_v1.json").read_text(encoding="utf-8"))
CFG = GENERATION["cases"]["industrial_pp_100"]
HISTORICAL = PROJECT / "PES_constraint_synthesis_no_anchor_20261010_v1" / "results_raw" / "formal_v1" / "industrial_pp_100"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Problem:
    def __init__(self) -> None:
        with (INPUTS / "industrial_pp_100_recipes.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.parameter_names = list(rows[0])[1:91]
        self.property_names = list(CFG["properties"])
        self.x_obs = np.asarray(
            [[float(row[name] or 0.0) / 100.0 for name in self.parameter_names] for row in rows],
            dtype=np.float64,
        )
        self.y_obs = np.asarray(
            [[float(row[name]) if row[name] else np.nan for name in self.property_names] for row in rows],
            dtype=np.float64,
        )
        self.windows = np.asarray(CFG["windows"], dtype=np.float64)
        self.bandwidth = float(CFG["rbf_bandwidth"])
        self.n = len(self.parameter_names)
        self.space = "simplex"
        self.lower = np.zeros(self.n, dtype=np.float64)
        self.upper = np.minimum(1.0, self.x_obs.max(axis=0) * float(CFG["ingredient_upper_margin"]))
        self.step = np.full(self.n, float(CFG["precision"]), dtype=np.float64)
        self.scale = np.maximum(self.upper - self.lower, self.step)
        masks: list[np.ndarray] = []
        lookup: dict[tuple[int, ...], int] = {}
        for row in self.x_obs:
            key = tuple(np.flatnonzero(row > 1e-12))
            if key not in lookup:
                lookup[key] = len(masks)
                masks.append(row > 1e-12)
        self.structure_masks = np.asarray(masks, dtype=bool)
        self.required_masks = self.structure_masks.copy()
        self.structure_count = len(masks)

    def _bounded_simplex(self, values: np.ndarray, structure: int) -> np.ndarray:
        mask = self.structure_masks[int(structure)]
        required = self.required_masks[int(structure)]
        lo = np.where(required, self.step, 0.0)
        hi = np.where(mask, self.upper, 0.0)
        residual = 1.0 - lo.sum()
        caps = hi - lo
        if residual < -1e-10 or caps.sum() < residual - 1e-10:
            raise ValueError((structure, "cannot close"))
        weights = np.maximum(np.asarray(values, dtype=float), 1e-12) * mask
        low, high = 0.0, 1.0
        while np.minimum(caps, weights * high).sum() < residual:
            high *= 2.0
        for _ in range(70):
            mid = (low + high) / 2.0
            if np.minimum(caps, weights * mid).sum() < residual:
                low = mid
            else:
                high = mid
        return lo + np.minimum(caps, weights * high)

    def _lattice_simplex(self, x: np.ndarray, structure: int) -> np.ndarray:
        step = float(self.step[0])
        mask = self.structure_masks[int(structure)]
        required = self.required_masks[int(structure)]
        low = np.where(required, 1, 0)
        high = np.floor(np.where(mask, self.upper, 0.0) / step + 1e-9).astype(int)
        scaled = x / step
        units = np.clip(np.floor(scaled + 1e-12).astype(int), low, high)
        diff = round(1.0 / step) - int(units.sum())
        fraction = scaled - np.floor(scaled)
        while diff:
            ids = np.flatnonzero(mask & ((units < high) if diff > 0 else (units > low)))
            order = ids[np.argsort(-fraction[ids] if diff > 0 else fraction[ids])]
            if not len(order):
                raise RuntimeError("lattice closure failed")
            for index in order:
                if not diff:
                    break
                units[index] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
        return units * step

    def project(self, values: np.ndarray, structure: int) -> np.ndarray:
        return self._lattice_simplex(self._bounded_simplex(values, structure), structure)

    def structural(self, x: np.ndarray, structure: int) -> bool:
        mask = self.structure_masks[int(structure)]
        required = self.required_masks[int(structure)]
        return bool(
            abs(float(x.sum()) - 1.0) <= 1e-8
            and np.all(x[~mask] <= 1e-10)
            and np.all(x >= self.lower - 1e-9)
            and np.all(x <= self.upper + 1e-9)
            and np.all(x[required] >= self.step[required] - 1e-9)
            and np.max(np.abs(x / self.step - np.rint(x / self.step))) <= 1e-6
        )

    def quality32(self, predicted: np.ndarray, design: np.ndarray) -> np.ndarray:
        predicted = np.asarray(predicted, dtype=np.float32)
        design = np.asarray(design, dtype=np.float32)
        windows = self.windows.astype(np.float32)
        quality = np.zeros(len(design), dtype=np.float32)
        for term in CFG["objective_terms"]:
            index = int(term["index"])
            width = windows[index, 1] - windows[index, 0]
            if term["kind"] == "property_max":
                score = (predicted[:, index] - windows[index, 0]) / width
            elif term["kind"] == "property_min":
                score = (windows[index, 1] - predicted[:, index]) / width
            else:
                score = design[:, term["indices"]].sum(axis=1) / np.float32(term["normalizer"])
            quality += np.float32(term["weight"]) * np.clip(score, np.float32(0), np.float32(1))
        return quality

    def audit32(self, design: np.ndarray, structures: np.ndarray, chunk: int = 1024) -> dict[str, np.ndarray]:
        design = np.asarray(design, dtype=np.float64)
        structures = np.asarray(structures, dtype=int)
        if not len(design):
            count = len(self.property_names)
            return {
                "predicted_properties": np.empty((0, count), dtype=np.float32),
                "task_quality": np.empty(0, dtype=np.float32),
                "nearest_database_distance": np.empty(0, dtype=np.float32),
                "structural_ok": np.empty(0, dtype=bool),
                "window_ok": np.empty(0, dtype=bool),
                "feasible": np.empty(0, dtype=bool),
            }
        xo = self.x_obs.astype(np.float32)
        scale = self.scale.astype(np.float32)
        y = self.y_obs.astype(np.float32)
        denominator = np.float32(2.0 * self.bandwidth**2 + 1e-12)
        predictions: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        for start in range(0, len(design), chunk):
            x = design[start : start + chunk].astype(np.float32)
            delta = (x[:, None, :] - xo[None, :, :]) / scale[None, None, :]
            d2 = np.sum(delta * delta, axis=2, dtype=np.float32)
            block: list[np.ndarray] = []
            for column in range(y.shape[1]):
                valid = np.isfinite(y[:, column])
                logits = -d2[:, valid] / denominator
                logits -= logits.max(axis=1, keepdims=True)
                weights = np.exp(logits, dtype=np.float32)
                weights /= weights.sum(axis=1, keepdims=True, dtype=np.float32)
                block.append(weights @ y[valid, column])
            predictions.append(np.stack(block, axis=1).astype(np.float32, copy=False))
            distances.append(np.sqrt(np.maximum(d2.min(axis=1), np.float32(0)) + np.float32(1e-12), dtype=np.float32))
        predicted = np.vstack(predictions)
        nearest = np.concatenate(distances)
        quality = self.quality32(predicted, design)
        structural = np.fromiter((self.structural(x, s) for x, s in zip(design, structures)), bool, len(design))
        windows = self.windows.astype(np.float32)
        window = np.all((predicted >= windows[:, 0]) & (predicted <= windows[:, 1]), axis=1)
        return {
            "predicted_properties": predicted,
            "task_quality": quality,
            "nearest_database_distance": nearest,
            "structural_ok": structural,
            "window_ok": window,
            "feasible": structural & window,
        }

    def predict64(self, design: np.ndarray, chunk: int = 512) -> np.ndarray:
        design = np.asarray(design, dtype=np.float64)
        out = np.empty((len(design), len(self.property_names)), dtype=np.float64)
        for start in range(0, len(design), chunk):
            x = design[start : start + chunk]
            d2 = np.sum(((x[:, None, :] - self.x_obs[None, :, :]) / self.scale) ** 2, axis=2)
            for column in range(self.y_obs.shape[1]):
                valid = np.isfinite(self.y_obs[:, column])
                logits = -d2[:, valid] / (2.0 * self.bandwidth**2 + 1e-12)
                logits -= logits.max(axis=1, keepdims=True)
                weights = np.exp(logits)
                weights /= weights.sum(axis=1, keepdims=True)
                out[start : start + len(x), column] = weights @ self.y_obs[valid, column]
        return out

    def quality64(self, predicted: np.ndarray, design: np.ndarray) -> np.ndarray:
        quality = np.zeros(len(design), dtype=np.float64)
        for term in CFG["objective_terms"]:
            index = int(term["index"])
            low, high = self.windows[index]
            if term["kind"] == "property_max":
                score = (predicted[:, index] - low) / (high - low)
            elif term["kind"] == "property_min":
                score = (high - predicted[:, index]) / (high - low)
            else:
                score = design[:, term["indices"]].sum(axis=1) / float(term["normalizer"])
            quality += float(term["weight"]) * np.clip(score, 0.0, 1.0)
        return quality


def normalized_distance(problem: Problem, first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm((first - second) / problem.scale))


def select_seeds(problem: Problem, pool: dict[str, np.ndarray]) -> np.ndarray:
    per_support = int(PROTOCOL["seed_selection"]["per_successful_support"])
    floor = float(PROTOCOL["seed_selection"]["quality_floor_quantile"])
    chosen: list[int] = []
    for structure in np.unique(pool["structure"]):
        ids = np.flatnonzero(pool["feasible"] & (pool["structure"] == structure))
        if not len(ids):
            continue
        quality = pool["task_quality"][ids]
        eligible = ids[quality >= np.quantile(quality, floor)]
        selected = [int(ids[np.argmax(quality)])]
        while len(selected) < per_support:
            remaining = [int(index) for index in eligible if index not in selected]
            if not remaining:
                break
            selected.append(
                max(
                    remaining,
                    key=lambda index: (
                        min(normalized_distance(problem, pool["design"][index], pool["design"][old]) for old in selected),
                        float(pool["task_quality"][index]),
                    ),
                )
            )
        chosen.extend(selected)
    return np.asarray(chosen, dtype=int)


def mutate(problem: Problem, x: np.ndarray, structure: int, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    active = np.flatnonzero(problem.structure_masks[structure])
    moves = int(rng.integers(PROTOCOL["ga"]["mutation_moves_min"], PROTOCOL["ga"]["mutation_moves_max"] + 1))
    step = float(problem.step[0])
    for _ in range(moves):
        first, second = rng.choice(active, 2, replace=False)
        amount = int(rng.integers(1, PROTOCOL["ga"]["mutation_quanta_max"] + 1)) * step
        y[first] += amount
        y[second] -= amount
    return problem.project(y, structure)


def run_ga(problem: Problem, pool: dict[str, np.ndarray], selected: np.ndarray, replicate: int) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    if not len(selected):
        raise RuntimeError("no feasible MPG seeds")
    budget = int(PROTOCOL["ga"]["attempts_per_run"])
    seed = 3101000 + replicate * 100 + 3
    rng = np.random.default_rng(seed)
    quotas = np.full(len(selected), budget // len(selected), dtype=int)
    quotas[: budget % len(selected)] += 1
    accepted_x: list[np.ndarray] = []
    accepted_structure: list[int] = []
    accepted_source: list[int] = []
    line_rows: list[dict] = []
    total = 0
    started = time.time()
    for lineage, (index, quota) in enumerate(zip(selected, quotas)):
        structure = int(pool["structure"][index])
        source = int(pool["source"][index])
        population = [pool["design"][index].copy()]
        qualities = [float(pool["task_quality"][index])]
        accepted = 0
        used = 0
        while used < quota:
            size = min(int(PROTOCOL["ga"]["batch_size"]), int(quota - used))
            children: list[np.ndarray] = []
            for _ in range(size):
                child = population[int(rng.integers(len(population)))].copy()
                if len(population) > 1 and rng.random() < PROTOCOL["ga"]["crossover_probability"]:
                    weight = rng.uniform(*PROTOCOL["ga"]["crossover_weight_range"])
                    child = weight * child + (1.0 - weight) * population[int(rng.integers(len(population)))]
                if rng.random() < PROTOCOL["ga"]["mutation_probability"]:
                    child = mutate(problem, child, structure, rng)
                else:
                    child = problem.project(child, structure)
                children.append(child)
            batch = np.asarray(children, dtype=np.float64)
            audit = problem.audit32(batch, np.full(size, structure, dtype=int))
            used += size
            total += size
            for child_index in np.flatnonzero(audit["feasible"]):
                accepted_x.append(batch[child_index])
                accepted_structure.append(structure)
                accepted_source.append(source)
                population.append(batch[child_index])
                qualities.append(float(audit["task_quality"][child_index]))
                accepted += 1
            order = np.argsort(-np.asarray(qualities))[: int(PROTOCOL["ga"]["elite_population"])]
            population = [population[position] for position in order]
            qualities = [qualities[position] for position in order]
        initial = float(pool["task_quality"][index])
        line_rows.append(
            {
                "replicate": replicate,
                "lineage": lineage,
                "pool_index": int(index),
                "structure": structure,
                "global_start_id": int(pool["global_start_id"][index]),
                "representative_step": int(pool["accepted_step"][index]),
                "budget": int(quota),
                "accepted": accepted,
                "initial_quality": initial,
                "best_quality": float(max(qualities)),
                "quality_gain": float(max(qualities) - initial),
            }
        )
    design = np.asarray(accepted_x, dtype=np.float64).reshape((-1, problem.n))
    structures = np.asarray(accepted_structure, dtype=int)
    audit = problem.audit32(design, structures)
    search = {
        "design": design,
        "structure": structures,
        "source": np.asarray(accepted_source, dtype=np.uint8),
        **audit,
    }
    receipt = {
        "replicate": replicate,
        "seed": seed,
        "attempts": total,
        "accepted": len(design),
        "accepted_rate": len(design) / total,
        "lineages": len(selected),
        "improved_lineages": sum(row["quality_gain"] > 0 for row in line_rows),
        "seconds": time.time() - started,
        "numeric_database_anchor_used": False,
        "nearest_row_repair_used": False,
    }
    return search, line_rows, receipt


def build_library(problem: Problem, pool: dict[str, np.ndarray], search: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    table: dict[tuple, list] = {}
    feasible = {
        key: value[pool["feasible"]]
        for key, value in pool.items()
        if isinstance(value, np.ndarray) and len(value) == len(pool["feasible"])
    }
    for stage, part in enumerate([feasible, search]):
        for index, row in enumerate(part["design"]):
            key = (int(part["structure"][index]), *np.round(row, 8))
            quality = float(part["task_quality"][index])
            source = int(part["source"][index])
            if key not in table:
                table[key] = [row, int(part["structure"][index]), quality, source, stage]
            else:
                table[key][2] = max(table[key][2], quality)
                table[key][3] |= source
                table[key][4] = max(table[key][4], stage)
    values = list(table.values())
    design = np.asarray([value[0] for value in values], dtype=np.float64)
    structures = np.asarray([value[1] for value in values], dtype=int)
    audit = problem.audit32(design, structures)
    library = {
        "design": design,
        "structure": structures,
        "task_quality": audit["task_quality"],
        "predicted_properties": audit["predicted_properties"],
        "nearest_database_distance": audit["nearest_database_distance"],
        "source_bits": np.asarray([value[3] for value in values], dtype=np.uint8),
        "latest_stage": np.asarray([value[4] for value in values], dtype=np.uint8),
    }
    per_support = int(PROTOCOL["library"]["portfolio_per_support"])
    floor = float(PROTOCOL["library"]["portfolio_quality_floor_quantile"])
    selected: list[int] = []
    for structure in np.unique(structures):
        group = np.flatnonzero(structures == structure)
        quality = library["task_quality"][group]
        eligible = group[quality >= np.quantile(quality, floor)]
        local = [int(group[np.argmax(quality)])]
        while len(local) < per_support:
            remaining = [int(index) for index in eligible if index not in local]
            if not remaining:
                break
            local.append(
                max(
                    remaining,
                    key=lambda index: (
                        min(normalized_distance(problem, design[index], design[old]) for old in local),
                        float(library["task_quality"][index]),
                    ),
                )
            )
        selected.extend(local)
    return library, np.asarray(selected, dtype=int)
