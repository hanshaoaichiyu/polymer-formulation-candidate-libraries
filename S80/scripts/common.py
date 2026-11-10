from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parent
SOURCE = PROJECT / "PES_constraint_synthesis_no_anchor_20261010_v1"
INPUTS = WORK / "inputs"
RESULTS = WORK / "results"
P = json.loads((INPUTS / "generation_protocol_v1.json").read_text(encoding="utf-8"))
FROZEN = json.loads((WORK / "configs" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
CASES = tuple(FROZEN["cases"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


class Problem:
    def __init__(self, name: str):
        self.name = name
        self.cfg = P["cases"][name]
        self.space = self.cfg["space"]
        dataset_name = {
            "natural_fiber_pp_hdpe": "pp_hdpe_banana_recipes.csv",
            "pcm_polymer_composite": "pcm_material_designs.csv",
        }[name]
        with (INPUTS / "data" / dataset_name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.parameter_names = list(self.cfg["parameters"])
        self.property_names = list(self.cfg["properties"])
        self.x_obs = np.asarray([[float(row[key]) for key in self.parameter_names] for row in rows], dtype=np.float64)
        self.y_obs = np.asarray(
            [[float(row[key]) if row[key] else np.nan for key in self.property_names] for row in rows], dtype=np.float64
        )
        self.windows = np.asarray(self.cfg["windows"], dtype=np.float64)
        self.bandwidth = float(self.cfg["rbf_bandwidth"])
        self.n = len(self.parameter_names)
        ranges = np.asarray(self.cfg["ranges"], dtype=np.float64)
        self.lower = ranges[:, 0]
        self.upper = ranges[:, 1]
        if self.space == "simplex":
            self.step = np.full(self.n, float(self.cfg["precision"]), dtype=np.float64)
            masks, required, names = [], [], []
            for branch in self.cfg["branches"]:
                mask = np.zeros(self.n, dtype=bool); mask[branch["allowed"]] = True
                req = np.zeros(self.n, dtype=bool); req[branch["required"]] = True
                masks.append(mask); required.append(req); names.append(branch["name"])
            self.structure_masks = np.asarray(masks, dtype=bool)
            self.required_masks = np.asarray(required, dtype=bool)
            self.structure_names = names
        else:
            self.step = np.asarray(self.cfg["precision"], dtype=np.float64)
            self.structure_masks = np.ones((1, self.n), dtype=bool)
            self.required_masks = np.zeros((1, self.n), dtype=bool)
            self.structure_names = ["global_box"]
        self.scale = np.maximum(self.upper - self.lower, self.step)
        self.structure_count = len(self.structure_masks)

    def _bounded_simplex(self, values: np.ndarray, structure: int) -> np.ndarray:
        mask = self.structure_masks[structure]
        required = self.required_masks[structure]
        low = np.where(required, self.step, 0.0)
        high = np.where(mask, self.upper, 0.0)
        residual = 1.0 - low.sum()
        capacities = high - low
        weights = np.maximum(np.asarray(values, dtype=float), 1e-12) * mask
        left, right = 0.0, 1.0
        while np.minimum(capacities, weights * right).sum() < residual:
            right *= 2.0
        for _ in range(70):
            middle = (left + right) / 2.0
            if np.minimum(capacities, weights * middle).sum() < residual:
                left = middle
            else:
                right = middle
        return low + np.minimum(capacities, weights * right)

    def _lattice_simplex(self, x: np.ndarray, structure: int) -> np.ndarray:
        step = float(self.step[0])
        mask = self.structure_masks[structure]
        required = self.required_masks[structure]
        low = np.where(required, 1, 0)
        high = np.floor(np.where(mask, self.upper, 0.0) / step + 1e-9).astype(int)
        scaled = x / step
        units = np.clip(np.floor(scaled + 1e-12).astype(int), low, high)
        difference = round(1.0 / step) - int(units.sum())
        fraction = scaled - np.floor(scaled)
        while difference:
            ids = np.flatnonzero(mask & ((units < high) if difference > 0 else (units > low)))
            order = ids[np.argsort(-fraction[ids] if difference > 0 else fraction[ids])]
            if not len(order):
                raise RuntimeError("lattice closure failed")
            for index in order:
                if not difference:
                    break
                units[index] += 1 if difference > 0 else -1
                difference += -1 if difference > 0 else 1
        return units * step

    @staticmethod
    def _transfer(row: np.ndarray, donors: np.ndarray, receivers: np.ndarray, amount: float,
                  low: np.ndarray, high: np.ndarray) -> None:
        remaining = max(float(amount), 0.0)
        for _ in range(4):
            if remaining <= 1e-14:
                break
            donor_capacity = np.maximum(row[donors] - low[donors], 0.0)
            receiver_capacity = np.maximum(high[receivers] - row[receivers], 0.0)
            available = min(float(donor_capacity.sum()), float(receiver_capacity.sum()), remaining)
            if available <= 1e-14:
                break
            row[donors] -= available * donor_capacity / donor_capacity.sum()
            row[receivers] += available * receiver_capacity / receiver_capacity.sum()
            remaining -= available

    def _repair_ratios_continuous(self, row: np.ndarray, structure: int) -> np.ndarray:
        if not self.cfg.get("ratio_constraints"):
            return row
        output = row.copy()
        mask = self.structure_masks[structure]
        low = np.where(self.required_masks[structure], self.step, 0.0)
        high = np.where(mask, self.upper, 0.0)
        for constraint in self.cfg["ratio_constraints"]:
            numerator = np.asarray(constraint["numerator"], dtype=int)
            denominator = np.asarray(constraint["denominator"], dtype=int)
            denominator_value = max(float(output[denominator].sum()), 1e-12)
            numerator_value = float(output[numerator].sum())
            maximum = float(constraint["max"])
            if numerator_value / denominator_value > maximum + 1e-12:
                transfer = (numerator_value - maximum * denominator_value) / (1.0 + maximum)
                self._transfer(output, numerator, denominator, transfer, low, high)
            denominator_value = max(float(output[denominator].sum()), 1e-12)
            numerator_value = float(output[numerator].sum())
            minimum = float(constraint["min"])
            if numerator_value / denominator_value < minimum - 1e-12 and minimum > 0:
                transfer = (minimum * denominator_value - numerator_value) / (1.0 + minimum)
                donors = np.asarray([index for index in denominator if mask[index]], dtype=int)
                receivers = np.asarray([index for index in numerator if mask[index]], dtype=int)
                self._transfer(output, donors, receivers, transfer, low, high)
        return output

    def _repair_ratios_lattice(self, x: np.ndarray, structure: int) -> np.ndarray:
        if not self.cfg.get("ratio_constraints"):
            return x
        y = x.copy(); step = float(self.step[0]); mask = self.structure_masks[structure]
        required = self.required_masks[structure]; low = np.where(required, step, 0.0)
        for constraint in self.cfg["ratio_constraints"]:
            numerator = np.asarray(constraint["numerator"], dtype=int)
            denominator = np.asarray(constraint["denominator"], dtype=int)
            for _ in range(int(round(1.0 / step)) + 2):
                ratio = y[numerator].sum() / max(y[denominator].sum(), 1e-12)
                if ratio <= float(constraint["max"]) + 1e-12:
                    break
                donors = [i for i in numerator if y[i] >= low[i] + step - 1e-12]
                receivers = [i for i in np.flatnonzero(mask) if i not in numerator and y[i] + step <= self.upper[i] + 1e-12]
                if not donors or not receivers:
                    break
                donor = max(donors, key=lambda i: y[i] - low[i]); receiver = max(receivers, key=lambda i: self.upper[i] - y[i])
                y[donor] -= step; y[receiver] += step
            for _ in range(int(round(1.0 / step)) + 2):
                ratio = y[numerator].sum() / max(y[denominator].sum(), 1e-12)
                if ratio >= float(constraint["min"]) - 1e-12:
                    break
                donors = [i for i in np.flatnonzero(mask) if i not in numerator and y[i] >= low[i] + step - 1e-12]
                receivers = [i for i in numerator if mask[i] and y[i] + step <= self.upper[i] + 1e-12]
                if not donors or not receivers:
                    break
                donor = max(donors, key=lambda i: y[i] - low[i]); receiver = max(receivers, key=lambda i: self.upper[i] - y[i])
                y[donor] -= step; y[receiver] += step
        return y

    def project(self, values: np.ndarray, structure: int) -> np.ndarray:
        if self.space == "simplex":
            return self._repair_ratios_lattice(self._lattice_simplex(self._bounded_simplex(values, structure), structure), structure)
        return np.clip(np.rint((np.asarray(values) - self.lower) / self.step) * self.step + self.lower, self.lower, self.upper)

    def project_continuous_batch(self, values: np.ndarray, structure: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.space == "box":
            return np.clip(values, self.lower, self.upper)
        mask = self.structure_masks[structure]
        required = self.required_masks[structure]
        low = np.where(required, self.step, 0.0)
        high = np.where(mask, self.upper, 0.0)
        residual = 1.0 - low.sum()
        capacities = high - low
        weights = np.maximum(values, 1e-12) * mask
        left = np.zeros(len(values)); right = np.ones(len(values))
        while True:
            sums = np.minimum(capacities[None, :], weights * right[:, None]).sum(axis=1)
            need = sums < residual
            if not np.any(need):
                break
            right[need] *= 2.0
        for _ in range(70):
            middle = (left + right) / 2.0
            sums = np.minimum(capacities[None, :], weights * middle[:, None]).sum(axis=1)
            need = sums < residual
            left[need] = middle[need]; right[~need] = middle[~need]
        output = low[None, :] + np.minimum(capacities[None, :], weights * right[:, None])
        if self.cfg.get("ratio_constraints"):
            output = np.asarray([self._repair_ratios_continuous(row, structure) for row in output])
        return output

    def structural(self, x: np.ndarray, structure: int) -> bool:
        if self.space == "box":
            return bool(
                np.all(x >= self.lower - 1e-8)
                and np.all(x <= self.upper + 1e-8)
                and np.max(np.abs((x - self.lower) / self.step - np.rint((x - self.lower) / self.step))) <= 1e-6
            )
        mask = self.structure_masks[structure]; required = self.required_masks[structure]
        ok = (
            abs(float(x.sum()) - 1.0) <= 1e-8
            and np.all(x[~mask] <= 1e-10)
            and np.all(x >= self.lower - 1e-9)
            and np.all(x <= self.upper + 1e-9)
            and np.all(x[required] >= self.step[required] - 1e-9)
            and np.max(np.abs(x / self.step - np.rint(x / self.step))) <= 1e-6
        )
        for constraint in self.cfg.get("ratio_constraints", []):
            ratio = x[constraint["numerator"]].sum() / max(x[constraint["denominator"]].sum(), 1e-12)
            ok &= ratio >= float(constraint["min"]) - 1e-9 and ratio <= float(constraint["max"]) + 1e-9
        return bool(ok)

    def predict(self, design: np.ndarray, dtype=np.float32, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        design = np.asarray(design, dtype=dtype)
        centers = self.x_obs.astype(dtype); scale = self.scale.astype(dtype); y = self.y_obs.astype(dtype)
        predicted = []; nearest = []
        denominator = dtype(2.0 * self.bandwidth**2 + 1e-12)
        for start in range(0, len(design), chunk):
            x = design[start : start + chunk]
            delta = (x[:, None, :] - centers[None, :, :]) / scale[None, None, :]
            d2 = np.sum(delta * delta, axis=2, dtype=dtype)
            columns = []
            for response in range(y.shape[1]):
                valid = np.isfinite(y[:, response])
                logits = -d2[:, valid] / denominator
                logits -= logits.max(axis=1, keepdims=True)
                weights = np.exp(logits).astype(dtype, copy=False)
                weights /= weights.sum(axis=1, keepdims=True, dtype=dtype)
                columns.append(weights @ y[valid, response])
            predicted.append(np.stack(columns, axis=1).astype(dtype, copy=False))
            nearest.append(np.sqrt(np.maximum(d2.min(axis=1), dtype(0)) + dtype(1e-12)).astype(dtype, copy=False))
        return np.vstack(predicted), np.concatenate(nearest)

    def quality(self, predicted: np.ndarray, design: np.ndarray, dtype=np.float32) -> np.ndarray:
        predicted = np.asarray(predicted, dtype=dtype); design = np.asarray(design, dtype=dtype)
        windows = self.windows.astype(dtype); result = np.zeros(len(design), dtype=dtype)
        for term in self.cfg["objective_terms"]:
            if term["kind"] == "property_max":
                index = int(term["index"]); score = (predicted[:, index] - windows[index, 0]) / (windows[index, 1] - windows[index, 0])
            elif term["kind"] == "property_min":
                index = int(term["index"]); score = (windows[index, 1] - predicted[:, index]) / (windows[index, 1] - windows[index, 0])
            else:
                score = design[:, term["indices"]].sum(axis=1, dtype=dtype) / dtype(term["normalizer"])
            result += dtype(term["weight"]) * np.clip(score, dtype(0), dtype(1))
        return result

    def audit(self, design: np.ndarray, structures: np.ndarray, dtype=np.float32) -> dict[str, np.ndarray]:
        design = np.asarray(design, dtype=np.float64); structures = np.asarray(structures, dtype=int)
        if not len(design):
            p = len(self.property_names)
            return {"predicted_properties": np.empty((0, p), dtype=dtype), "task_quality": np.empty(0, dtype=dtype),
                    "nearest_database_distance": np.empty(0, dtype=dtype), "structural_ok": np.empty(0, bool),
                    "window_ok": np.empty(0, bool), "feasible": np.empty(0, bool)}
        prediction, nearest = self.predict(design, dtype=dtype)
        quality = self.quality(prediction, design, dtype=dtype)
        structural = np.fromiter((self.structural(row, int(s)) for row, s in zip(design, structures)), bool, len(design))
        windows = self.windows.astype(dtype)
        window = np.all((prediction >= windows[:, 0]) & (prediction <= windows[:, 1]), axis=1)
        return {"predicted_properties": prediction, "task_quality": quality, "nearest_database_distance": nearest,
                "structural_ok": structural, "window_ok": window, "feasible": structural & window}

    def margin(self, prediction: np.ndarray) -> np.ndarray:
        width = np.ptp(self.windows, axis=1)
        return np.minimum((prediction - self.windows[:, 0]) / width, (self.windows[:, 1] - prediction) / width).min(axis=1)

    def coordinate_key(self, design: np.ndarray, structures: np.ndarray) -> np.ndarray:
        if self.space == "simplex":
            coords = np.rint(np.asarray(design) / self.step).astype(np.int64)
        else:
            coords = np.rint((np.asarray(design) - self.lower) / self.step).astype(np.int64)
        return np.column_stack([np.asarray(structures, dtype=np.int64), coords])


def objective_gradient(problem: Problem, x: np.ndarray, active: np.ndarray, penalty_weight: float = 10.0):
    x = np.asarray(x, dtype=float); xa = x[:, active]; centers = problem.x_obs[:, active]; scale = problem.scale[active]
    inactive = np.ones(problem.n, dtype=bool); inactive[active] = False
    d2 = np.sum(((xa[:, None] - centers[None]) / scale[None, None]) ** 2, axis=2)
    if np.any(inactive):
        d2 += np.sum(((x[:, None, inactive] - problem.x_obs[None, :, inactive]) / problem.scale[inactive]) ** 2, axis=2)
    logits = -d2 / (2.0 * problem.bandwidth**2 + 1e-12); logits -= logits.max(axis=1, keepdims=True)
    kernel = np.exp(logits); prediction = np.empty((len(x), len(problem.property_names)))
    derivative = np.empty((len(x), len(problem.property_names), len(active)))
    gradient_log_kernel = -(xa[:, None] - centers[None]) / (problem.bandwidth**2 * scale[None, None] ** 2 + 1e-12)
    for response in range(len(problem.property_names)):
        valid = np.isfinite(problem.y_obs[:, response]); weights = kernel[:, valid]; normalized = weights / weights.sum(axis=1, keepdims=True)
        y = problem.y_obs[valid, response]; prediction[:, response] = normalized @ y
        derivative[:, response] = np.sum((normalized * (y[None] - prediction[:, response, None]))[:, :, None] * gradient_log_kernel[:, valid], axis=1)
    q = np.zeros(len(x)); dq = np.zeros((len(x), len(active))); lookup = {int(index): local for local, index in enumerate(active)}
    for term in problem.cfg["objective_terms"]:
        weight = float(term["weight"])
        if term["kind"] == "property_max":
            response = int(term["index"]); width = np.ptp(problem.windows[response]); score = (prediction[:, response] - problem.windows[response, 0]) / width; sign = 1.0
        elif term["kind"] == "property_min":
            response = int(term["index"]); width = np.ptp(problem.windows[response]); score = (problem.windows[response, 1] - prediction[:, response]) / width; sign = -1.0
        else:
            local_indices = [lookup[int(index)] for index in term["indices"] if int(index) in lookup]
            score = x[:, term["indices"]].sum(axis=1) / float(term["normalizer"]); q += weight * np.clip(score, 0, 1)
            middle = (score > 0) & (score < 1)
            if local_indices:
                dq[np.ix_(middle, local_indices)] += weight / float(term["normalizer"])
            continue
        q += weight * np.clip(score, 0, 1); middle = (score > 0) & (score < 1)
        dq[middle] += sign * weight * derivative[middle, response] / width
    widths = np.ptp(problem.windows, axis=1)
    low = np.maximum((problem.windows[:, 0] - prediction) / widths, 0); high = np.maximum((prediction - problem.windows[:, 1]) / widths, 0)
    penalty = np.sum(low * low + high * high, axis=1)
    penalty_gradient = np.sum(((-2 * low / widths + 2 * high / widths)[:, :, None]) * derivative, axis=1)
    return q - penalty_weight * penalty, dq - penalty_weight * penalty_gradient, prediction


def objective_only(problem: Problem, x: np.ndarray, penalty_weight: float = 10.0) -> np.ndarray:
    prediction, _ = problem.predict(x, dtype=np.float64, chunk=4096)
    q = problem.quality(prediction, x, dtype=np.float64)
    widths = np.ptp(problem.windows, axis=1)
    low = np.maximum((problem.windows[:, 0] - prediction) / widths, 0); high = np.maximum((prediction - problem.windows[:, 1]) / widths, 0)
    return q - penalty_weight * np.sum(low * low + high * high, axis=1)


class BestCollector:
    def __init__(self, problem: Problem, starts: int):
        self.problem = problem; self.count = starts
        self.feasible_visit_count = np.zeros(starts, dtype=np.int32)
        self.audited_visit_count = np.zeros(starts, dtype=np.int32)
        self.best_found = np.zeros(starts, dtype=bool)
        self.best_q = np.full(starts, -np.inf, dtype=np.float32)
        self.best_design = np.full((starts, problem.n), np.nan, dtype=np.float64)
        self.best_prediction = np.full((starts, len(problem.property_names)), np.nan, dtype=np.float32)
        self.best_margin = np.full(starts, np.nan, dtype=np.float32)
        self.best_step = np.full(starts, -1, dtype=np.int16)
        self.best_objective = np.full(starts, np.nan, dtype=np.float64)
        self.counters = defaultdict(int); self.step_rows: list[dict] = []

    def visit(self, local_ids: np.ndarray, continuous: np.ndarray, objective: np.ndarray, structure: int, step: int) -> None:
        projected = np.asarray([self.problem.project(row, structure) for row in continuous])
        structures = np.full(len(projected), structure, dtype=int)
        audit = self.problem.audit(projected, structures, dtype=np.float32)
        self.audited_visit_count[local_ids] += 1
        feasible_positions = np.flatnonzero(audit["feasible"])
        self.feasible_visit_count[local_ids[feasible_positions]] += 1
        self.counters["audited_visits"] += len(projected); self.counters["feasible_visits"] += len(feasible_positions)
        improved = 0
        for position in feasible_positions:
            local = int(local_ids[position]); quality = float(audit["task_quality"][position])
            if not self.best_found[local] or quality > float(self.best_q[local]) + 1e-12:
                self.best_found[local] = True; self.best_q[local] = quality; self.best_design[local] = projected[position]
                self.best_prediction[local] = audit["predicted_properties"][position]
                self.best_margin[local] = float(self.problem.margin(audit["predicted_properties"][[position]])[0])
                self.best_step[local] = step; self.best_objective[local] = float(objective[position]); improved += 1
        self.step_rows.append({"structure": structure, "accepted_step": step, "audited": len(projected),
                               "feasible": len(feasible_positions), "representative_updates": improved})


def optimize_structure(problem: Problem, starts: np.ndarray, structure: int, maximum_updates: int) -> dict:
    spec = FROZEN["mpg_ascent"]; n = len(starts); active = np.flatnonzero(problem.structure_masks[structure])
    x = starts.copy(); collector = BestCollector(problem, n)
    objective, gradient, _ = objective_gradient(problem, x, active)
    gradient_evaluations = np.ones(n, dtype=np.int32); backtracking = np.zeros(n, dtype=np.int32); updates = np.zeros(n, dtype=np.int16)
    stop = np.full(n, -1, dtype=np.int8); alive = np.ones(n, dtype=bool); monotonic_violations = np.zeros(n, dtype=np.int16)
    collector.visit(np.arange(n), x, objective, structure, 0)
    for update in range(1, maximum_updates + 1):
        ids = np.flatnonzero(alive)
        if not len(ids):
            break
        if problem.space == "simplex":
            z = x[ids][:, active] / problem.scale[active]
        else:
            z = (x[ids][:, active] - problem.lower[active]) / problem.scale[active]
        gz = gradient[ids] * problem.scale[active]
        proposal = x[ids].copy()
        if problem.space == "simplex":
            proposal[:, active] = (z + float(spec["initial_step_size"]) * gz) * problem.scale[active]
        else:
            proposal[:, active] = problem.lower[active] + (z + float(spec["initial_step_size"]) * gz) * problem.scale[active]
        initial = problem.project_continuous_batch(proposal, structure)
        new_z = initial[:, active] / problem.scale[active] if problem.space == "simplex" else (initial[:, active] - problem.lower[active]) / problem.scale[active]
        displacement = new_z - z; norm = np.linalg.norm(displacement, axis=1); directional = np.sum(gz * displacement, axis=1)
        stationary = (norm <= float(spec["projected_step_norm_tolerance"])) | (directional <= float(spec["directional_derivative_tolerance"]))
        if np.any(stationary):
            stop[ids[stationary]] = 1; alive[ids[stationary]] = False
        unresolved = ids[~stationary]; accepted: list[int] = []
        for backtrack_index in range(int(spec["maximum_backtracks_per_update"])):
            if not len(unresolved):
                break
            if problem.space == "simplex":
                current_z = x[unresolved][:, active] / problem.scale[active]
            else:
                current_z = (x[unresolved][:, active] - problem.lower[active]) / problem.scale[active]
            current_gradient = gradient[unresolved] * problem.scale[active]
            alpha = float(spec["initial_step_size"]) * float(spec["backtracking_factor"]) ** backtrack_index
            trial = x[unresolved].copy()
            if problem.space == "simplex":
                trial[:, active] = (current_z + alpha * current_gradient) * problem.scale[active]
            else:
                trial[:, active] = problem.lower[active] + (current_z + alpha * current_gradient) * problem.scale[active]
            trial = problem.project_continuous_batch(trial, structure)
            new_objective = objective_only(problem, trial); backtracking[unresolved] += 1
            trial_z = trial[:, active] / problem.scale[active] if problem.space == "simplex" else (trial[:, active] - problem.lower[active]) / problem.scale[active]
            direction = np.sum(current_gradient * (trial_z - current_z), axis=1)
            ok = new_objective >= objective[unresolved] + float(spec["armijo_c1"]) * direction - float(spec["objective_increase_tolerance"])
            if np.any(ok):
                take = unresolved[ok]; old = objective[take].copy(); x[take] = trial[ok]; objective[take] = new_objective[ok]
                updates[take] += 1; monotonic_violations[take] += objective[take] < old - float(spec["objective_increase_tolerance"])
                accepted.extend(take.tolist())
            unresolved = unresolved[~ok]
        if len(unresolved):
            stop[unresolved] = 2; alive[unresolved] = False
        accepted_array = np.asarray(accepted, dtype=int)
        if len(accepted_array):
            values, new_gradient, _ = objective_gradient(problem, x[accepted_array], active)
            gradient_evaluations[accepted_array] += 1; objective[accepted_array] = values; gradient[accepted_array] = new_gradient
            collector.visit(accepted_array, x[accepted_array], values, structure, update)
        if update == maximum_updates:
            stop[alive] = 0; alive[:] = False
    return {"collector": collector, "updates": updates, "gradient_evaluations": gradient_evaluations,
            "backtracking": backtracking, "stop": stop, "monotonic_violations": monotonic_violations}


def normalized_distance(problem: Problem, first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm((first - second) / problem.scale))


def select_seeds(problem: Problem, pool: dict[str, np.ndarray]) -> np.ndarray:
    chosen = []; count = int(FROZEN["seed_selection"]["per_successful_structure"])
    floor = float(FROZEN["seed_selection"]["quality_floor_quantile"])
    for structure in np.unique(pool["structure"]):
        ids = np.flatnonzero(pool["feasible"] & (pool["structure"] == structure))
        quality = pool["task_quality"][ids]; eligible = ids[quality >= np.quantile(quality, floor)]
        selected = [int(ids[np.argmax(quality)])]
        while len(selected) < count:
            remaining = [int(index) for index in eligible if index not in selected]
            if not remaining:
                break
            selected.append(max(remaining, key=lambda index: (
                min(normalized_distance(problem, pool["design"][index], pool["design"][old]) for old in selected),
                float(pool["task_quality"][index]),
            )))
        chosen.extend(selected)
    return np.asarray(chosen, dtype=int)


def mutate(problem: Problem, x: np.ndarray, structure: int, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    if problem.space == "simplex":
        active = np.flatnonzero(problem.structure_masks[structure])
        moves = int(rng.integers(FROZEN["ga"]["mutation_moves_min"], FROZEN["ga"]["mutation_moves_max"] + 1))
        step = float(problem.step[0])
        for _ in range(moves):
            first, second = rng.choice(active, 2, replace=False)
            amount = int(rng.integers(1, FROZEN["ga"]["mutation_quanta_max"] + 1)) * step
            y[first] += amount; y[second] -= amount
    else:
        for _ in range(int(rng.integers(1, 5))):
            index = int(rng.integers(problem.n))
            y[index] += rng.choice([-1, 1]) * int(rng.integers(1, FROZEN["ga"]["mutation_quanta_max"] + 1)) * problem.step[index]
    return problem.project(y, structure)


def run_ga(problem: Problem, pool: dict[str, np.ndarray], selected: np.ndarray, replicate: int, budget: int) -> tuple[dict, list[dict], dict]:
    seed = 3101000 + replicate * 100 + 3; rng = np.random.default_rng(seed)
    quotas = np.full(len(selected), budget // len(selected), dtype=int); quotas[: budget % len(selected)] += 1
    accepted_x, accepted_structure, accepted_source, lineages = [], [], [], []; total = 0; started = time.perf_counter()
    for lineage, (index, quota) in enumerate(zip(selected, quotas)):
        structure = int(pool["structure"][index]); population = [pool["design"][index].copy()]
        quality = [float(pool["task_quality"][index])]; accepted = 0; used = 0
        while used < quota:
            size = min(int(FROZEN["ga"]["batch_size"]), int(quota - used)); children = []
            for _ in range(size):
                child = population[int(rng.integers(len(population)))].copy()
                if len(population) > 1 and rng.random() < FROZEN["ga"]["crossover_probability"]:
                    weight = rng.uniform(*FROZEN["ga"]["crossover_weight_range"])
                    child = weight * child + (1.0 - weight) * population[int(rng.integers(len(population)))]
                child = mutate(problem, child, structure, rng) if rng.random() < FROZEN["ga"]["mutation_probability"] else problem.project(child, structure)
                children.append(child)
            batch = np.asarray(children); audit = problem.audit(batch, np.full(size, structure), dtype=np.float32)
            used += size; total += size
            for position in np.flatnonzero(audit["feasible"]):
                accepted_x.append(batch[position]); accepted_structure.append(structure); accepted_source.append(8)
                population.append(batch[position]); quality.append(float(audit["task_quality"][position])); accepted += 1
            order = np.argsort(-np.asarray(quality))[: int(FROZEN["ga"]["elite_population"])]
            population = [population[position] for position in order]; quality = [quality[position] for position in order]
        initial = float(pool["task_quality"][index])
        lineages.append({"lineage": lineage, "pool_index": int(index), "structure": structure,
                         "global_start_id": int(pool["global_start_id"][index]), "representative_step": int(pool["accepted_step"][index]),
                         "budget": int(quota), "accepted": accepted, "initial_quality": initial,
                         "best_quality": float(max(quality)), "quality_gain": float(max(quality) - initial)})
    design = np.asarray(accepted_x, dtype=np.float64).reshape((-1, problem.n)); structures = np.asarray(accepted_structure, dtype=int)
    audit = problem.audit(design, structures, dtype=np.float32)
    search = {"design": design, "structure": structures, "source": np.asarray(accepted_source, dtype=np.uint8), **audit}
    receipt = {"seed": seed, "attempts": total, "accepted": len(design), "accepted_rate": len(design) / total,
               "lineages": len(selected), "improved_lineages": sum(row["quality_gain"] > 0 for row in lineages),
               "seconds": time.perf_counter() - started}
    return search, lineages, receipt


def build_library(problem: Problem, pool: dict[str, np.ndarray], search: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    table = {}
    for stage, part in enumerate([pool, search]):
        ids = np.flatnonzero(part["feasible"]) if "feasible" in part else np.arange(len(part["design"]))
        for index in ids:
            row = part["design"][index]
            key = tuple(problem.coordinate_key(row[None], np.asarray([part["structure"][index]]))[0])
            value = [row, int(part["structure"][index]), int(part["source"][index]), stage]
            if key not in table:
                table[key] = value
            else:
                table[key][2] |= value[2]; table[key][3] = max(table[key][3], stage)
    values = list(table.values()); design = np.asarray([value[0] for value in values]); structures = np.asarray([value[1] for value in values], dtype=int)
    audit = problem.audit(design, structures, dtype=np.float32)
    library = {"design": design, "structure": structures, "task_quality": audit["task_quality"],
               "predicted_properties": audit["predicted_properties"], "nearest_database_distance": audit["nearest_database_distance"],
               "source_bits": np.asarray([value[2] for value in values], dtype=np.uint8),
               "latest_stage": np.asarray([value[3] for value in values], dtype=np.uint8)}
    cap = int(FROZEN["portfolio_per_structure"][problem.name]); selected = []
    for structure in np.unique(structures):
        group = np.flatnonzero(structures == structure); quality = library["task_quality"][group]
        eligible = group[quality >= np.quantile(quality, 0.5)]; local = [int(group[np.argmax(quality)])]
        while len(local) < cap:
            remaining = [int(index) for index in eligible if index not in local]
            if not remaining:
                break
            local.append(max(remaining, key=lambda index: (
                min(normalized_distance(problem, design[index], design[old]) for old in local), float(library["task_quality"][index])
            )))
        selected.extend(local)
    return library, np.asarray(selected, dtype=int)
