from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import MONOTONE, ROOT, V14, Problem, atomic_json, load_npz, margin, write_csv


OLD_PROTOCOL = json.loads((ROOT / "configs" / "mpga_protocol_v1.json").read_text(encoding="utf-8"))


def weighted_project_z(z, scale, lo, hi):
    zlo = lo / scale; zhi = hi / scale
    left = float(np.min((z - zhi) / scale)) - 1.0
    right = float(np.max((z - zlo) / scale)) + 1.0
    for _ in range(80):
        tau = (left + right) / 2.0
        trial = np.clip(z - tau * scale, zlo, zhi)
        if np.dot(scale, trial) > 1.0:
            left = tau
        else:
            right = tau
    output = np.clip(z - (left + right) / 2.0 * scale, zlo, zhi)
    free = np.flatnonzero((output > zlo + 1e-12) & (output < zhi - 1e-12))
    if len(free):
        output[free[-1]] += (1.0 - np.dot(scale, output)) / scale[free[-1]]
    return output


def objective_gradient(problem, x, active, penalty_weight=10.0):
    x = np.asarray(x, float); xa = x[:, active]; centers = problem.x_obs[:, active]; scale = problem.scale[active]
    inactive = np.ones(problem.n, bool); inactive[active] = False
    d2 = np.sum(((xa[:, None] - centers[None]) / scale[None, None]) ** 2, axis=2)
    if np.any(inactive):
        d2 += np.sum((problem.x_obs[:, inactive] / problem.scale[inactive]) ** 2, axis=1)[None]
    log_kernel = -d2 / (2 * problem.bandwidth**2 + 1e-12); log_kernel -= np.max(log_kernel, axis=1, keepdims=True)
    kernel = np.exp(log_kernel); prediction = np.empty((len(x), len(problem.property_names)))
    derivative = np.empty((len(x), len(problem.property_names), len(active)))
    gradient_log_kernel = -(xa[:, None] - centers[None]) / (problem.bandwidth**2 * scale[None, None] ** 2 + 1e-12)
    for response in range(len(problem.property_names)):
        valid = np.isfinite(problem.y_obs[:, response]); weight = kernel[:, valid]; normalized = weight / weight.sum(1, keepdims=True)
        y = problem.y_obs[valid, response]; prediction[:, response] = normalized @ y
        derivative[:, response] = np.sum((normalized * (y[None] - prediction[:, response, None]))[:, :, None] * gradient_log_kernel[:, valid], axis=1)
    q = np.zeros(len(x)); dq = np.zeros((len(x), len(active))); lookup = {int(index): local for local, index in enumerate(active)}
    for term in problem.cfg["objective_terms"]:
        weight = float(term["weight"])
        if term["kind"] == "property_max":
            response = term["index"]; width = np.ptp(problem.windows[response]); score = (prediction[:, response] - problem.windows[response, 0]) / width; sign = 1
        elif term["kind"] == "property_min":
            response = term["index"]; width = np.ptp(problem.windows[response]); score = (problem.windows[response, 1] - prediction[:, response]) / width; sign = -1
        else:
            indices = [lookup[int(index)] for index in term["indices"] if int(index) in lookup]
            score = np.sum(x[:, term["indices"]], axis=1) / float(term["normalizer"]); q += weight * np.clip(score, 0, 1)
            middle = (score > 0) & (score < 1)
            if indices:
                dq[np.ix_(middle, indices)] += weight / float(term["normalizer"])
            continue
        q += weight * np.clip(score, 0, 1); middle = (score > 0) & (score < 1); dq[middle] += sign * weight * derivative[middle, response] / width
    widths = problem.windows[:, 1] - problem.windows[:, 0]
    low = np.maximum((problem.windows[:, 0] - prediction) / widths, 0); high = np.maximum((prediction - problem.windows[:, 1]) / widths, 0)
    penalty = np.sum(low * low + high * high, axis=1)
    penalty_gradient = np.sum(((-2 * low / widths + 2 * high / widths)[:, :, None]) * derivative, axis=1)
    return q - penalty_weight * penalty, dq - penalty_weight * penalty_gradient, prediction, q, penalty


def objective_only(problem, x, active, penalty_weight=10.0):
    x = np.asarray(x, float); xa = x[:, active]; centers = problem.x_obs[:, active]; scale = problem.scale[active]
    inactive = np.ones(problem.n, bool); inactive[active] = False
    d2 = np.sum(((xa[:, None] - centers[None]) / scale[None, None]) ** 2, axis=2)
    if np.any(inactive):
        d2 += np.sum((problem.x_obs[:, inactive] / problem.scale[inactive]) ** 2, axis=1)[None]
    log_kernel = -d2 / (2 * problem.bandwidth**2 + 1e-12); log_kernel -= np.max(log_kernel, axis=1, keepdims=True)
    kernel = np.exp(log_kernel); prediction = np.empty((len(x), len(problem.property_names)))
    for response in range(len(problem.property_names)):
        valid = np.isfinite(problem.y_obs[:, response]); weights = kernel[:, valid]
        prediction[:, response] = (weights @ problem.y_obs[valid, response]) / weights.sum(1)
    q = np.zeros(len(x))
    for term in problem.cfg["objective_terms"]:
        if term["kind"] == "property_max":
            response = term["index"]; score = (prediction[:, response] - problem.windows[response, 0]) / np.ptp(problem.windows[response])
        elif term["kind"] == "property_min":
            response = term["index"]; score = (problem.windows[response, 1] - prediction[:, response]) / np.ptp(problem.windows[response])
        else:
            score = np.sum(x[:, term["indices"]], axis=1) / float(term["normalizer"])
        q += float(term["weight"]) * np.clip(score, 0, 1)
    widths = problem.windows[:, 1] - problem.windows[:, 0]
    low = np.maximum((problem.windows[:, 0] - prediction) / widths, 0); high = np.maximum((prediction - problem.windows[:, 1]) / widths, 0)
    return q - penalty_weight * np.sum(low * low + high * high, axis=1)


class TrajectoryCollector:
    def __init__(self, problem, global_start_ids, structure):
        self.problem = problem; self.global_start_ids = np.asarray(global_start_ids, int); self.structure = int(structure)
        self.seen = [set() for _ in global_start_ids]; self.counter = defaultdict(int); self.rows = []
        self.design = []; self.start_id = []; self.step = []; self.q = []; self.margin = []; self.objective = []; self.continuous_window = []
        self.all_chunks = []

    def visit(self, local_ids, continuous, predictions, objective, step):
        self.counter["accepted_or_step0_visits"] += len(local_ids)
        continuous_ok = np.all((predictions >= self.problem.windows[:, 0]) & (predictions <= self.problem.windows[:, 1]), axis=1)
        self.counter["continuous_window_pass"] += int(continuous_ok.sum())
        unique_design = []; unique_local = []; unique_continuous_ok = []; unique_objective = []
        for position, local in enumerate(local_ids):
            repaired = self.problem.project(continuous[position], self.structure)
            self.counter["repair_calls"] += 1
            key = np.rint(repaired / self.problem.step).astype(np.int32).tobytes()
            if key in self.seen[int(local)]:
                self.counter["within_start_duplicate_after_repair"] += 1
                continue
            self.seen[int(local)].add(key); unique_design.append(repaired); unique_local.append(int(local))
            unique_continuous_ok.append(bool(continuous_ok[position])); unique_objective.append(float(objective[position]))
        if not unique_design:
            return
        unique_design = np.asarray(unique_design); structures = np.full(len(unique_design), self.structure, int)
        audit = self.problem.audit(unique_design, structures); self.counter["unique_repaired_audit_calls"] += len(unique_design)
        self.counter["audited_structural_pass"] += int(audit["structural_ok"].sum())
        self.counter["audited_window_pass"] += int(audit["window_ok"].sum())
        self.counter["audited_complete_pass"] += int(audit["feasible"].sum())
        self.all_chunks.append({
            "design": unique_design.astype(np.float32),
            "structure": structures.astype(np.int16),
            "global_start_id": self.global_start_ids[np.asarray(unique_local, int)].astype(np.int32),
            "accepted_step": np.full(len(unique_design), int(step), np.int16),
            "continuous_objective": np.asarray(unique_objective, np.float32),
            "continuous_window_pass": np.asarray(unique_continuous_ok, bool),
            "predicted_properties": audit["predicted_properties"].astype(np.float32),
            "task_quality": audit["task_quality"].astype(np.float32),
            "margin": margin(self.problem, audit["predicted_properties"]).astype(np.float32),
            "structural_ok": audit["structural_ok"].astype(bool),
            "window_ok": audit["window_ok"].astype(bool),
            "feasible": audit["feasible"].astype(bool),
            "property_window_ok": ((audit["predicted_properties"] >= self.problem.windows[:, 0]) & (audit["predicted_properties"] <= self.problem.windows[:, 1])).astype(bool),
        })
        for position in range(len(unique_design)):
            if unique_continuous_ok[position] and not audit["window_ok"][position]: self.counter["discrete_window_failure_added"] += 1
            if not unique_continuous_ok[position] and audit["window_ok"][position]: self.counter["repair_window_recovered"] += 1
            if not audit["feasible"][position]: continue
            self.design.append(unique_design[position].astype(np.float32)); self.start_id.append(int(self.global_start_ids[unique_local[position]]))
            self.step.append(int(step)); self.q.append(float(audit["task_quality"][position])); self.margin.append(float(margin(self.problem, audit["predicted_properties"][[position]])[0]))
            self.objective.append(unique_objective[position]); self.continuous_window.append(unique_continuous_ok[position])


def optimize_structure(problem, starts, global_ids, structure, max_updates):
    spec = OLD_PROTOCOL["optimizer"]; mask = problem.structure_masks[structure]; required = problem.required_masks[structure]
    active = np.flatnonzero(mask); scale = problem.scale[active]; lo = np.where(required, problem.step, 0.0)[active]; hi = np.where(mask, problem.upper, 0.0)[active]
    x = starts.copy(); n = len(x); collector = TrajectoryCollector(problem, global_ids, structure)
    objective, gradient, prediction, _, _ = objective_gradient(problem, x, active)
    gradient_evaluations = np.ones(n, np.int32); backtracking_forwards = np.zeros(n, np.int32); updates = np.zeros(n, np.int16)
    stop = np.full(n, -1, np.int8); alive = np.ones(n, bool); best_x = x.copy(); best_objective = objective.copy(); diagnostic_forwards = 0
    collector.visit(np.arange(n), x, prediction, objective, 0)
    for update in range(1, max_updates + 1):
        ids = np.flatnonzero(alive)
        if not len(ids): break
        z = x[ids][:, active] / scale; gz = gradient[ids] * scale
        initial = np.asarray([weighted_project_z(z[i] + float(spec["initial_step_size"]) * gz[i], scale, lo, hi) for i in range(len(ids))])
        displacement = initial - z; norm = np.linalg.norm(displacement, axis=1); directional = np.sum(gz * displacement, axis=1)
        stationary = (norm <= float(spec["projected_step_norm_tolerance"])) | (directional <= float(spec["directional_derivative_tolerance"]))
        if np.any(stationary): stop[ids[stationary]] = 1; alive[ids[stationary]] = False
        unresolved = ids[~stationary]; accepted = []
        for backtrack in range(int(spec["maximum_backtracks_per_update"])):
            if not len(unresolved): break
            alpha = float(spec["initial_step_size"]) * float(spec["backtracking_factor"]) ** backtrack
            current_z = x[unresolved][:, active] / scale; current_gradient = gradient[unresolved] * scale
            new_z = np.asarray([weighted_project_z(current_z[i] + alpha * current_gradient[i], scale, lo, hi) for i in range(len(unresolved))])
            trial = x[unresolved].copy(); trial[:, active] = new_z * scale
            new_objective = objective_only(problem, trial, active); backtracking_forwards[unresolved] += 1
            direction = np.sum(current_gradient * (new_z - current_z), axis=1)
            ok = new_objective >= objective[unresolved] + float(spec["armijo_c1"]) * direction - float(spec["objective_increase_tolerance"])
            if np.any(ok):
                take = unresolved[ok]; x[take] = trial[ok]; objective[take] = new_objective[ok]; updates[take] += 1
                improved = objective[take] > best_objective[take] + float(spec["objective_increase_tolerance"])
                best_x[take[improved]] = x[take[improved]]; best_objective[take[improved]] = objective[take[improved]]; accepted.extend(take.tolist())
            unresolved = unresolved[~ok]
        if len(unresolved): stop[unresolved] = 2; alive[unresolved] = False
        accepted = np.asarray(accepted, int)
        if len(accepted):
            values, new_gradient, accepted_prediction, _, _ = objective_gradient(problem, x[accepted], active)
            gradient_evaluations[accepted] += 1; objective[accepted] = values; gradient[accepted] = new_gradient
            if update == max_updates: diagnostic_forwards += len(accepted)
            collector.visit(accepted, x[accepted], accepted_prediction, values, update)
        if update == max_updates:
            stop[alive] = 0; alive[:] = False
    return {"best_x": best_x, "best_objective": best_objective, "updates": updates,
            "gradient_evaluations": gradient_evaluations, "backtracking_forwards": backtracking_forwards,
            "stop": stop, "collector": collector, "terminal_diagnostic_gradient_evals": diagnostic_forwards}


def run(replicate, pilot=False, starts_file=None, output_label=None):
    problem = Problem("industrial_pp_100")
    starts_path = Path(starts_file) if starts_file else V14 / "results_raw" / "mspg_formal_v1" / f"replicate_{replicate}" / "starts.npz"
    starts_data = load_npz(starts_path); starts = starts_data["design"]; structures = starts_data["structure"]
    if pilot:
        requested = [s for s in [0, 28] if np.any(structures == s)]
        if not requested: requested = list(np.unique(structures)[:2])
        keep = np.flatnonzero(np.isin(structures, requested))
        keep = np.concatenate([keep[structures[keep] == structure][:12] for structure in requested])
        starts = starts[keep]; structures = structures[keep]; global_ids = keep; maximum_updates = 20; mode = "pilot"
    else:
        global_ids = np.arange(len(starts)); maximum_updates = int(OLD_PROTOCOL["optimizer"]["maximum_updates_per_start"]); mode = "formal_v1"
    out = ROOT / "results_raw" / (output_label or "mpga_t") / mode / f"replicate_{replicate}"
    if (out / "COMPLETED.json").exists():
        print((out / "COMPLETED.json").read_text(encoding="utf-8")); return
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); results = []
    for structure in np.unique(structures):
        ids = np.flatnonzero(structures == structure)
        results.append((ids, optimize_structure(problem, starts[ids], global_ids[ids], int(structure), maximum_updates)))
    feasible_design = []; feasible_structure = []; feasible_start = []; feasible_step = []; feasible_q = []; feasible_margin = []; feasible_objective = []; feasible_continuous_window = []
    counters = defaultdict(int); structure_summary = []; total_updates = total_gradients = total_backtracks = total_diag = 0
    best = np.empty_like(starts); best_continuous_objective = np.empty(len(starts), float)
    best_continuous_prediction = np.empty((len(starts), len(problem.property_names)), float)
    best_continuous_q = np.empty(len(starts), float); best_continuous_penalty = np.empty(len(starts), float)
    all_visit_chunks = []
    for ids, result in results:
        collector = result["collector"]; best[ids] = result["best_x"]
        active = np.flatnonzero(problem.structure_masks[collector.structure])
        endpoint_objective, _, endpoint_prediction, endpoint_q, endpoint_penalty = objective_gradient(problem, result["best_x"], active)
        best_continuous_objective[ids] = endpoint_objective
        best_continuous_prediction[ids] = endpoint_prediction
        best_continuous_q[ids] = endpoint_q
        best_continuous_penalty[ids] = endpoint_penalty
        all_visit_chunks.extend(collector.all_chunks)
        total_updates += int(result["updates"].sum()); total_gradients += int(result["gradient_evaluations"].sum()); total_backtracks += int(result["backtracking_forwards"].sum()); total_diag += int(result["terminal_diagnostic_gradient_evals"])
        for key, value in collector.counter.items(): counters[key] += int(value)
        feasible_design.extend(collector.design); feasible_structure.extend([collector.structure] * len(collector.design)); feasible_start.extend(collector.start_id); feasible_step.extend(collector.step); feasible_q.extend(collector.q); feasible_margin.extend(collector.margin); feasible_objective.extend(collector.objective); feasible_continuous_window.extend(collector.continuous_window)
        structure_summary.append({"replicate": replicate, "structure": collector.structure, "starts": len(ids), **dict(collector.counter), "retained_feasible_records": len(collector.design)})
    designs = np.asarray(feasible_design, np.float32)
    np.savez_compressed(out / "trajectory_feasible_candidates.npz", design=designs, structure=np.asarray(feasible_structure, np.int16),
                        global_start_id=np.asarray(feasible_start, np.int32), accepted_step=np.asarray(feasible_step, np.int16),
                        task_quality=np.asarray(feasible_q, np.float32), margin=np.asarray(feasible_margin, np.float32),
                        continuous_objective=np.asarray(feasible_objective, np.float32), continuous_window_pass=np.asarray(feasible_continuous_window, bool))
    all_visits = {key: np.concatenate([chunk[key] for chunk in all_visit_chunks]) for key in all_visit_chunks[0]}
    np.savez_compressed(out / "trajectory_all_unique_audited_visits.npz", **all_visits)
    write_csv(out / "structure_audit_summary.csv", structure_summary)
    terminal = np.asarray([problem.project(row, int(structure)) for row, structure in zip(best, structures)])
    terminal_audit = problem.audit(terminal, structures)
    endpoint_margin = margin(problem, terminal_audit["predicted_properties"])
    np.savez_compressed(out / "same_rerun_best_endpoint_candidates.npz", global_start_id=np.asarray(global_ids,np.int32),
                        best_continuous=best, best_continuous_objective=best_continuous_objective.astype(np.float32),
                        best_continuous_predicted_properties=best_continuous_prediction.astype(np.float32),
                        best_continuous_task_quality=best_continuous_q.astype(np.float32),
                        best_continuous_window_penalty=best_continuous_penalty.astype(np.float32),
                        best_continuous_property_window_ok=((best_continuous_prediction >= problem.windows[:, 0]) & (best_continuous_prediction <= problem.windows[:, 1])),
                        best_continuous_window_ok=np.all((best_continuous_prediction >= problem.windows[:, 0]) & (best_continuous_prediction <= problem.windows[:, 1]), axis=1),
                        design=terminal, structure=structures, margin=endpoint_margin.astype(np.float32),
                        property_window_ok=((terminal_audit["predicted_properties"] >= problem.windows[:, 0]) & (terminal_audit["predicted_properties"] <= problem.windows[:, 1])),
                        **terminal_audit)
    old_path = MONOTONE / "results_raw" / "mpga_formal_v1" / f"replicate_{replicate}" / "terminal_candidates.npz"
    old = load_npz(old_path) if old_path.exists() else None
    receipt = {"replicate": replicate, "pilot": pilot, "starts": len(starts), "maximum_updates": maximum_updates,
               "accepted_updates": total_updates, "gradient_evaluations": total_gradients, "backtracking_forward_evaluations": total_backtracks,
               "extra_terminal_diagnostic_gradient_evaluations": total_diag, **dict(counters),
               "retained_feasible_records": len(designs), "retained_unique_global_compositions": len({(int(structure), np.rint(design / problem.step).astype(np.int32).tobytes()) for design, structure in zip(designs, feasible_structure)}),
               "saved_all_unique_audited_visits": int(len(all_visits["design"])),
               "terminal_feasible_rate": float(terminal_audit["feasible"].mean()),
               "old_terminal_feasible_rate": None if old is None else float(old["feasible"].mean()),
               "terminal_design_max_abs_difference_from_old": None if old is None else float(np.max(np.abs(terminal - old["design"]))),
               "terminal_feasible_flags_equal_old": None if old is None else bool(np.array_equal(terminal_audit["feasible"], old["feasible"])),
               "elapsed_seconds": time.perf_counter() - started,
               "rejected_line_search_trials_not_retained": True, "terminal_result_preserved_not_replaced": True}
    atomic_json(out / "COMPLETED.json", receipt); print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--replicate", type=int, required=True); parser.add_argument("--pilot", action="store_true"); parser.add_argument("--starts-file"); parser.add_argument("--output-label")
    arguments = parser.parse_args(); run(arguments.replicate, arguments.pilot, arguments.starts_file, arguments.output_label)
