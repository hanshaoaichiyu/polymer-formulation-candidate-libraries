from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import frozen_core
import mpg_common
import run_traced_ga


WORK = Path(__file__).resolve().parents[1]
INPUTS = WORK / "inputs"
RESULTS = WORK / "results" / "marginal"
CONFIG = json.loads((WORK / "configs" / "MARGINAL_GAIN_PROTOCOL.json").read_text(encoding="utf-8"))
V16 = WORK.parent / "CMS_v16_focused_analysis_20261014_v1" / "open_source_package" / "results_raw" / "mpga_t_paired_v16_formal" / "formal_v1"
MPGA_SPEC = json.loads((WORK / "configs" / "mpga_protocol_v1.json").read_text(encoding="utf-8"))["optimizer"]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def load(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def reconstruction_cache_identity(replicate: int) -> dict:
    return {
        "replicate": replicate,
        "starts_sha256": mpg_common.sha256(INPUTS / "mpga_starts" / f"replicate_{replicate}.npz"),
        "optimizer_protocol_sha256": mpg_common.sha256(WORK / "configs" / "mpga_protocol_v1.json"),
        "frozen_reference_implementation_sha256": mpg_common.sha256(WORK / "scripts" / "frozen_run_mpga_t.py"),
        "maximum_updates": 120,
        "state_fields": ["x", "gradient", "objective", "best_x", "best_objective", "updates", "gradient_evaluations", "backtracking_forwards", "stop", "active", "lo", "hi"],
    }


def mark_cache_verified(replicate: int, verification: dict) -> None:
    root = RESULTS / "state_cache" / f"replicate_{replicate}"
    receipt = reconstruction_cache_identity(replicate)
    receipt.update({"verified": bool(verification["exact_projected_endpoint_identity"]), "verification": verification})
    atomic_json(root / "CACHE_RECEIPT.json", receipt)


def synchronize() -> None:
    if frozen_core.torch.cuda.is_available():
        frozen_core.torch.cuda.synchronize()


def weighted_project_z(z, scale, lo, hi):
    zlo = lo / scale; zhi = hi / scale
    left = float(np.min((z - zhi) / scale)) - 1.0
    right = float(np.max((z - zlo) / scale)) + 1.0
    for _ in range(80):
        tau = (left + right) / 2.0
        trial = np.clip(z - tau * scale, zlo, zhi)
        if np.dot(scale, trial) > 1.0: left = tau
        else: right = tau
    output = np.clip(z - (left + right) / 2.0 * scale, zlo, zhi)
    free = np.flatnonzero((output > zlo + 1e-12) & (output < zhi - 1e-12))
    if len(free): output[free[-1]] += (1.0 - np.dot(scale, output)) / scale[free[-1]]
    return output


def objective_gradient(problem, x, active, penalty_weight=10.0):
    x = np.asarray(x, float); xa = x[:, active]; centers = problem.x_obs[:, active]; scale = problem.scale[active]
    inactive = np.ones(problem.n, bool); inactive[active] = False
    d2 = np.sum(((xa[:, None] - centers[None]) / scale[None, None]) ** 2, axis=2)
    if np.any(inactive): d2 += np.sum((problem.x_obs[:, inactive] / problem.scale[inactive]) ** 2, axis=1)[None]
    log_kernel = -d2 / (2 * problem.bandwidth**2 + 1e-12); log_kernel -= np.max(log_kernel, axis=1, keepdims=True)
    kernel = np.exp(log_kernel); prediction = np.empty((len(x), len(problem.property_names)))
    derivative = np.empty((len(x), len(problem.property_names), len(active)))
    gradient_log_kernel = -(xa[:, None] - centers[None]) / (problem.bandwidth**2 * scale[None, None] ** 2 + 1e-12)
    for response in range(len(problem.property_names)):
        valid = np.isfinite(problem.y_obs[:, response]); weight = kernel[:, valid]; normalized = weight / weight.sum(1, keepdims=True)
        y = problem.y_obs[valid, response]; prediction[:, response] = normalized @ y
        derivative[:, response] = np.sum((normalized * (y[None] - prediction[:, response, None]))[:, :, None] * gradient_log_kernel[:, valid], axis=1)
    q = np.zeros(len(x)); dq = np.zeros((len(x), len(active))); lookup = {int(index): local for local, index in enumerate(active)}
    for term in mpg_common.CFG["objective_terms"]:
        weight = float(term["weight"])
        if term["kind"] == "property_max":
            response = term["index"]; width = np.ptp(problem.windows[response]); score = (prediction[:, response] - problem.windows[response, 0]) / width; sign = 1
        elif term["kind"] == "property_min":
            response = term["index"]; width = np.ptp(problem.windows[response]); score = (problem.windows[response, 1] - prediction[:, response]) / width; sign = -1
        else:
            indices = [lookup[int(index)] for index in term["indices"] if int(index) in lookup]
            score = np.sum(x[:, term["indices"]], axis=1) / float(term["normalizer"]); q += weight * np.clip(score, 0, 1)
            middle = (score > 0) & (score < 1)
            if indices: dq[np.ix_(middle, indices)] += weight / float(term["normalizer"])
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
    for term in mpg_common.CFG["objective_terms"]:
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


def optimize_block(problem, starts, structure, maximum_updates, collect_visits=False):
    spec = MPGA_SPEC; mask = problem.structure_masks[structure]; required = problem.required_masks[structure]
    active = np.flatnonzero(mask); scale = problem.scale[active]; lo = np.where(required, problem.step, 0.0)[active]; hi = np.where(mask, problem.upper, 0.0)[active]
    x = starts.copy(); n = len(x)
    objective, gradient, prediction, _, _ = objective_gradient(problem, x, active)
    gradient_evaluations = np.ones(n, np.int32); backtracking_forwards = np.zeros(n, np.int32); updates = np.zeros(n, np.int16)
    stop = np.full(n, -1, np.int8); alive = np.ones(n, bool); best_x = x.copy(); best_objective = objective.copy()
    visit_x, visit_local, visit_step = [], [], []
    if collect_visits:
        visit_x.append(x.copy()); visit_local.append(np.arange(n)); visit_step.append(np.zeros(n, np.int16))
    for update in range(1, maximum_updates + 1):
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
            values, new_gradient, _, _, _ = objective_gradient(problem, x[accepted], active)
            gradient_evaluations[accepted] += 1; objective[accepted] = values; gradient[accepted] = new_gradient
            if collect_visits:
                visit_x.append(x[accepted].copy()); visit_local.append(accepted.copy()); visit_step.append(np.full(len(accepted), update, np.int16))
        if update == maximum_updates: stop[alive] = 0; alive[:] = False
    return {"x": x, "gradient": gradient, "objective": objective, "best_x": best_x, "best_objective": best_objective,
            "updates": updates, "gradient_evaluations": gradient_evaluations, "backtracking_forwards": backtracking_forwards,
            "stop": stop, "active": active, "lo": lo, "hi": hi,
            "visit_x": visit_x, "visit_local": visit_local, "visit_step": visit_step}


def reconstruct_run(problem, replicate, collect_new=False):
    starts_data = load(INPUTS / "mpga_starts" / f"replicate_{replicate}.npz")
    starts, structures = starts_data["design"], starts_data["structure"]
    cache_root = RESULTS / "state_cache" / f"replicate_{replicate}"
    receipt_path = cache_root / "CACHE_RECEIPT.json"
    identity = reconstruction_cache_identity(replicate)
    if not collect_new and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("verified") and all(receipt.get(key) == value for key, value in identity.items()):
            states = []
            for structure in np.unique(structures):
                data = load(cache_root / f"structure_{int(structure):03d}.npz")
                ids = data.pop("global_ids")
                data.update({"visit_x": [], "visit_local": [], "visit_step": []})
                states.append((int(structure), ids, data))
            return starts, structures, states
    states = []
    for structure in np.unique(structures):
        ids = np.flatnonzero(structures == structure)
        states.append((int(structure), ids, optimize_block(problem, starts[ids], int(structure), 120, collect_visits=collect_new)))
    if not collect_new:
        cache_root.mkdir(parents=True, exist_ok=True)
        for structure, ids, state in states:
            save_npz(cache_root / f"structure_{structure:03d}.npz", global_ids=ids,
                     **{key: state[key] for key in identity["state_fields"]})
        pending = dict(identity); pending.update({"verified": False, "verification": None})
        atomic_json(receipt_path, pending)
    return starts, structures, states


def verify_reconstruction(problem, replicate, structures, states):
    best = np.empty((len(structures), problem.n), float)
    for structure, ids, state in states: best[ids] = state["best_x"]
    projected = np.asarray([problem.project(x, int(s)) for x, s in zip(best, structures)])
    historical = load(V16 / f"replicate_{replicate}" / "same_rerun_best_endpoint_candidates.npz")
    return {"max_abs_projected_endpoint_difference": float(np.max(np.abs(projected - historical["design"]))),
            "exact_projected_endpoint_identity": bool(np.array_equal(np.rint(projected / problem.step), np.rint(historical["design"] / problem.step)))}


def continue_states(problem, states, extra_passes, collect=True):
    visits_x, visits_structure, visits_start, visits_step = [], [], [], []
    counts = {"gradient_evaluations": 0, "line_search_forwards": 0, "eligible_starts": 0, "accepted_updates": 0}
    started = time.perf_counter()
    for structure, global_ids, state in states:
        eligible = np.flatnonzero(state["stop"] == 0)
        counts["eligible_starts"] += len(eligible)
        if not len(eligible): continue
        x = state["x"].copy(); objective = state["objective"].copy(); gradient = state["gradient"].copy(); active = state["active"]
        scale = problem.scale[active]; lo = state["lo"]; hi = state["hi"]
        alive = np.zeros(len(x), bool); alive[eligible] = True
        for extra in range(1, extra_passes + 1):
            ids = np.flatnonzero(alive)
            if not len(ids): break
            z = x[ids][:, active] / scale; gz = gradient[ids] * scale
            initial = np.asarray([weighted_project_z(z[i] + float(MPGA_SPEC["initial_step_size"]) * gz[i], scale, lo, hi) for i in range(len(ids))])
            displacement = initial - z; directional = np.sum(gz * displacement, axis=1)
            stationary = (np.linalg.norm(displacement, axis=1) <= float(MPGA_SPEC["projected_step_norm_tolerance"])) | (directional <= float(MPGA_SPEC["directional_derivative_tolerance"]))
            alive[ids[stationary]] = False; unresolved = ids[~stationary]; accepted = []
            for backtrack in range(int(MPGA_SPEC["maximum_backtracks_per_update"])):
                if not len(unresolved): break
                alpha = float(MPGA_SPEC["initial_step_size"]) * float(MPGA_SPEC["backtracking_factor"]) ** backtrack
                current_z = x[unresolved][:, active] / scale; current_gradient = gradient[unresolved] * scale
                new_z = np.asarray([weighted_project_z(current_z[i] + alpha * current_gradient[i], scale, lo, hi) for i in range(len(unresolved))])
                trial = x[unresolved].copy(); trial[:, active] = new_z * scale
                new_objective = objective_only(problem, trial, active); counts["line_search_forwards"] += len(unresolved)
                direction = np.sum(current_gradient * (new_z - current_z), axis=1)
                ok = new_objective >= objective[unresolved] + float(MPGA_SPEC["armijo_c1"]) * direction - float(MPGA_SPEC["objective_increase_tolerance"])
                if np.any(ok):
                    take = unresolved[ok]; x[take] = trial[ok]; objective[take] = new_objective[ok]; accepted.extend(take.tolist())
                unresolved = unresolved[~ok]
            alive[unresolved] = False
            accepted = np.asarray(accepted, int)
            if len(accepted):
                values, new_gradient, _, _, _ = objective_gradient(problem, x[accepted], active)
                counts["gradient_evaluations"] += len(accepted); counts["accepted_updates"] += len(accepted)
                objective[accepted] = values; gradient[accepted] = new_gradient
                if collect:
                    visits_x.append(x[accepted].copy()); visits_structure.append(np.full(len(accepted), structure, int))
                    visits_start.append(global_ids[accepted].copy()); visits_step.append(np.full(len(accepted), 120 + extra, int))
    if visits_x:
        design = np.vstack(visits_x); structure = np.concatenate(visits_structure); start_id = np.concatenate(visits_start); step = np.concatenate(visits_step)
        projected = np.asarray([problem.project(x, int(s)) for x, s in zip(design, structure)])
        audit = problem.audit32(projected, structure)
    else:
        design = np.empty((0, problem.n)); structure = np.empty(0, int); start_id = np.empty(0, int); step = np.empty(0, int)
        projected = design.copy(); audit = problem.audit32(projected, structure)
    counts["audit_candidates"] = len(projected)
    counts["projection_candidates"] = len(projected)
    counts["seconds"] = time.perf_counter() - started
    return {"continuous_design": design, "design": projected, "structure": structure, "global_start_id": start_id, "accepted_step": step, **audit}, counts


def hit_and_run_step(x, lo, hi, active, rng):
    if len(active) <= 1: return x.copy(), True
    direction = rng.normal(size=len(active)); direction -= direction.mean(); norm = np.linalg.norm(direction)
    if norm <= 1e-14: return x.copy(), False
    direction /= norm; xa = x[active]; low_t, high_t = -np.inf, np.inf
    for value, lower, upper, delta in zip(xa, lo[active], hi[active], direction):
        if delta > 0: a, b = (lower - value) / delta, (upper - value) / delta
        elif delta < 0: a, b = (upper - value) / delta, (lower - value) / delta
        else: continue
        low_t, high_t = max(low_t, a), min(high_t, b)
    if not np.isfinite(low_t + high_t) or high_t <= low_t + 1e-15: return x.copy(), False
    y = x.copy(); y[active] = xa + rng.uniform(low_t, high_t) * direction; y[active[-1]] += 1.0 - y.sum(); return y, True


def sample_new_mpga_starts(problem, replicate, atomic_unit=0):
    chain = atomic_unit % 4
    designs, structures, transitions = [], [], 0
    seed_base = int(CONFIG["A_MPGA_then_GA"]["new_start_seed_base"])
    for structure in range(problem.structure_count):
        mask = problem.structure_masks[structure]; req = problem.required_masks[structure]
        lo = np.where(req, problem.step, 0.0); hi = np.where(mask, problem.upper, 0.0); active = np.flatnonzero(mask)
        x0 = lo.copy(); residual = 1.0 - x0.sum(); caps = hi - lo
        if residual > 0: x0[active] += residual * caps[active] / caps[active].sum()
        rng = np.random.default_rng(seed_base + replicate * 1000000 + atomic_unit * 100000 + structure * 100 + chain)
        x = x0.copy()
        for _ in range(1200): x, _ = hit_and_run_step(x, lo, hi, active, rng); transitions += 1
        for _ in range(10): x, _ = hit_and_run_step(x, lo, hi, active, rng); transitions += 1
        designs.append(x.copy()); structures.append(structure)
    return np.asarray(designs), np.asarray(structures, int), transitions


def run_new_mpga_unit(problem, replicate, atomic_unit=0, collect=True):
    started = time.perf_counter(); starts, structures, transitions = sample_new_mpga_starts(problem, replicate, atomic_unit); sample_seconds = time.perf_counter() - started
    visits_x, visits_s, visits_start, visits_step = [], [], [], []
    gradients = backtracks = 0
    for structure in np.unique(structures):
        ids = np.flatnonzero(structures == structure); state = optimize_block(problem, starts[ids], int(structure), 120, collect_visits=collect)
        gradients += int(state["gradient_evaluations"].sum()); backtracks += int(state["backtracking_forwards"].sum())
        if collect:
            for x, local, step in zip(state["visit_x"], state["visit_local"], state["visit_step"]):
                visits_x.append(x); visits_s.append(np.full(len(x), structure, int)); visits_start.append(ids[local] + atomic_unit * problem.structure_count); visits_step.append(step)
    compute_seconds = time.perf_counter() - started - sample_seconds
    continuous = np.vstack(visits_x); s = np.concatenate(visits_s); start_id = np.concatenate(visits_start); step = np.concatenate(visits_step)
    design = np.asarray([problem.project(x, int(structure)) for x, structure in zip(continuous, s)])
    audit = problem.audit32(design, s)
    return {"continuous_design": continuous, "design": design, "structure": s, "global_start_id": start_id,
            "accepted_step": step, **audit}, {"seconds": time.perf_counter() - started, "start_sampling_seconds": sample_seconds,
            "optimization_seconds": compute_seconds, "hit_and_run_transitions": transitions, "starts": len(starts),
            "gradient_evaluations": gradients, "line_search_forwards": backtracks, "audit_candidates": len(design),
            "atomic_unit": atomic_unit}


def run_generator_batch(replicate, batch=0, count=8000, collect=True):
    device = frozen_core.torch.device("cuda" if frozen_core.torch.cuda.is_available() else "cpu")
    problem = frozen_core.Problem("industrial_pp_100")
    model = frozen_core.Generator(problem.n, frozen_core.P["training"]["latent_dim"], frozen_core.P["training"]["hidden_dims"]).to(device)
    model.load_state_dict(frozen_core.torch.load(INPUTS / "generator_models" / f"replicate_{replicate}.pt", map_location=device, weights_only=True)); model.eval()
    seed = int(CONFIG["B_Generator_then_GA"]["new_sampling_seed_base"]) + replicate * 100000 + batch
    frozen_core.torch.manual_seed(seed); structures = np.arange(count, dtype=int) % problem.structure_count
    raw_parts, proposal_parts = [], []
    synchronize(); started = time.perf_counter()
    with frozen_core.torch.no_grad():
        for start in range(0, count, 2048):
            s = structures[start:start + 2048]; features = problem.structure_features(s, device)
            logits = model(frozen_core.torch.randn(len(s), model.latent_dim, device=device), features)
            raw, x, _, _ = problem.relaxed(logits, s); raw_parts.append(raw.cpu().numpy()); proposal_parts.append(x.cpu().numpy())
    raw = np.vstack(raw_parts); proposal = np.vstack(proposal_parts)
    design = np.asarray([problem.project(x, int(s)) for x, s in zip(proposal, structures)])
    audit = problem.audit(design, structures, device); synchronize(); compute_seconds = time.perf_counter() - started
    result = {"raw_output": raw, "pre_projection": proposal, "design": design, "structure": structures,
              "source": np.full(count, 1, np.uint8), **audit}
    return result, {"seconds": compute_seconds, "generator_inference_candidates": count, "audit_candidates": count,
                    "forward_predictor_candidates": count, "seed": seed, "device": str(device), "batch_size": 2048,
                    "sampling_batch": batch}


def calibrate():
    output = RESULTS / "calibration"; output.mkdir(parents=True, exist_ok=True)
    device = frozen_core.torch.device("cuda" if frozen_core.torch.cuda.is_available() else "cpu")
    timings = {}
    # Excluded warmups.
    for route in ("MPG_GA", "Generator_GA"):
        problem, pool, selected, audit_fn, mutate_fn, protocol, _ = run_traced_ga.load_route(1, route, device)
        _ = run_traced_ga.baseline_short(problem, pool, selected[:4], 1, route, audit_fn, mutate_fn, protocol, 512)
        synchronize(); started = time.perf_counter()
        baseline = run_traced_ga.baseline_short(problem, pool, selected, 1, route, audit_fn, mutate_fn, protocol, 40000)
        synchronize(); timings[f"{route}_GA_40000_seconds"] = time.perf_counter() - started
        timings[f"{route}_accepted"] = len(baseline[0])
    problem = mpg_common.Problem()
    started = time.perf_counter(); starts, structures, states = reconstruct_run(problem, 1); timings["MPGA_state_reconstruction_seconds"] = time.perf_counter() - started
    timings.update({f"reconstruction_{key}": value for key, value in verify_reconstruction(problem, 1, structures, states).items()})
    _, continuation_cost = continue_states(problem, states, 1, collect=True)
    timings["MPGA_continue_one_pass_seconds"] = continuation_cost["seconds"]
    _, new_mpga_cost = run_new_mpga_unit(problem, 1, atomic_unit=0, collect=True)
    timings["MPGA_new_start_atomic_seconds"] = new_mpga_cost["seconds"]
    _, generator_cost = run_generator_batch(1, batch=0, count=8000, collect=False)
    timings["Generator_new_8000_seconds"] = generator_cost["seconds"]
    anchor_a = timings["MPG_GA_GA_40000_seconds"]; anchor_b = timings["Generator_GA_GA_40000_seconds"]
    a2_units = int(anchor_a // timings["MPGA_continue_one_pass_seconds"])
    a3_units = int(anchor_a // timings["MPGA_new_start_atomic_seconds"])
    b2_units = int(anchor_b // timings["Generator_new_8000_seconds"])
    budget = {"status": "FROZEN_AFTER_TIME_ONLY_CALIBRATION_BEFORE_FORMAL_CONTRAST_RESULTS",
              "anchor_A_seconds": anchor_a, "anchor_B_seconds": anchor_b,
              "A2_primary_extra_passes": a2_units, "A2_execute_secondary_one_if_zero": a2_units == 0,
              "A3_primary_atomic_units": a3_units, "A3_execute_secondary_one_if_zero": a3_units == 0,
              "B2_primary_8000_batches": b2_units, "B2_execute_secondary_one_if_zero": b2_units == 0,
              "calibration": timings, "budget_selected_only_from_time": True,
              "calibration_outcomes_not_used": True}
    atomic_json(output / "CALIBRATION.json", timings); atomic_json(WORK / "configs" / "CALIBRATED_BUDGET.json", budget)
    print(json.dumps(budget, ensure_ascii=False, indent=2))


def formal(replicate):
    budget = json.loads((WORK / "configs" / "CALIBRATED_BUDGET.json").read_text(encoding="utf-8"))
    root = RESULTS / "formal" / f"replicate_{replicate}"
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    problem = mpg_common.Problem()
    started = time.perf_counter(); _, structures, states = reconstruct_run(problem, replicate); reconstruction_seconds = time.perf_counter() - started
    reconstruction = verify_reconstruction(problem, replicate, structures, states)
    if not reconstruction["exact_projected_endpoint_identity"]: raise RuntimeError(f"MPGA reconstruction mismatch run {replicate}")
    mark_cache_verified(replicate, reconstruction)
    a2_passes = int(budget["A2_primary_extra_passes"] or 1); a2_status = "primary_equal_time" if budget["A2_primary_extra_passes"] else "secondary_over_budget_minimum_atomic_unit"
    a2, a2_cost = continue_states(problem, states, a2_passes, collect=True); save_npz(root / "A2_continued_MPGA_visits.npz", **a2)
    a3_units = int(budget["A3_primary_atomic_units"] or 1); a3_parts = []; a3_costs = []
    for unit in range(a3_units):
        part, cost = run_new_mpga_unit(problem, replicate, atomic_unit=unit, collect=True); a3_parts.append(part); a3_costs.append(cost)
    a3 = {key: np.concatenate([part[key] for part in a3_parts]) for key in a3_parts[0]}
    save_npz(root / "A3_new_start_MPGA_visits.npz", **a3)
    b2_units = int(budget["B2_primary_8000_batches"] or 1); b2_parts = []; b2_costs = []
    for unit in range(b2_units):
        part, cost = run_generator_batch(replicate, batch=unit, count=8000, collect=True); b2_parts.append(part); b2_costs.append(cost)
    b2 = {key: np.concatenate([part[key] for part in b2_parts]) for key in b2_parts[0]}
    save_npz(root / "B2_additional_generator_candidates.npz", **b2)
    receipt = {"replicate": replicate, "reconstruction_seconds_experimental_overhead": reconstruction_seconds,
               "reconstruction": reconstruction, "A2_budget_status": a2_status, "A2_extra_passes": a2_passes, "A2_cost": a2_cost,
               "A3_budget_status": "primary_equal_time" if budget["A3_primary_atomic_units"] else "secondary_over_budget_minimum_atomic_unit",
               "A3_atomic_units": a3_units, "A3_cost": a3_costs,
               "B2_budget_status": "primary_equal_time" if budget["B2_primary_8000_batches"] else "secondary_over_budget_minimum_atomic_unit",
               "B2_batches": b2_units, "B2_cost": b2_costs,
               "hardware": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
                            "torch": frozen_core.torch.__version__, "cuda": frozen_core.torch.cuda.get_device_name(0) if frozen_core.torch.cuda.is_available() else None},
               "protocol_unchanged": True, "manuscript_or_figure_modified": False}
    atomic_json(root / "COMPLETED.json", receipt); print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--calibrate", action="store_true"); parser.add_argument("--replicate", type=int, choices=range(1, 6))
    args = parser.parse_args()
    if args.calibrate: calibrate()
    elif args.replicate: formal(args.replicate)
    else: parser.error("choose --calibrate or --replicate")


if __name__ == "__main__":
    main()
