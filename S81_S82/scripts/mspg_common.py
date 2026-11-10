from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / "PES_constraint_synthesis_no_anchor_20261010_v1"
V11 = PROJECT / "PES_v11_review_supplement_20261012_v1"
V13 = PROJECT / "PES_v13_targeted_analysis_20261012_v1"
BHR = PROJECT / "PES_v13_strong_bounded_random_20261012_v1"
PROTOCOL_PATH = ROOT / "configs" / "protocol_v1.json"
P = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v13 = _load(V13 / "scripts" / "common.py", "cms_v14_v13_common")
bhr = _load(BHR / "scripts" / "common.py", "cms_v14_bhr_common")
Problem = bhr.Problem


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_rep(rep: int) -> Path:
    return SOURCE / "results_raw" / "formal_v1" / "industrial_pp_100" / f"replicate_{rep}"


def project_bounded_simplex(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Exact Euclidean projection: x=clip(v-tau, lo, hi), sum(x)=total."""
    v = np.asarray(v, float)
    if lo.sum() > total + 1e-12 or hi.sum() < total - 1e-12:
        raise ValueError("empty bounded simplex")
    left = float(np.min(v - hi)) - 1.0
    right = float(np.max(v - lo)) + 1.0
    for _ in range(80):
        mid = (left + right) / 2.0
        if np.clip(v - mid, lo, hi).sum() > total:
            left = mid
        else:
            right = mid
    x = np.clip(v - (left + right) / 2.0, lo, hi)
    active = np.flatnonzero((x > lo + 1e-12) & (x < hi - 1e-12))
    if len(active):
        x[active[-1]] += total - x.sum()
    return x


def predict_with_gradient(problem: Problem, x: np.ndarray):
    """Frozen RBF prediction and analytic derivative with respect to one design."""
    x = np.asarray(x, float)
    delta = (x[None, :] - problem.x_obs) / problem.scale[None, :]
    d2 = np.sum(delta * delta, axis=1)
    kernel = np.exp(-d2 / (2.0 * problem.bandwidth**2 + 1e-12))
    grad_log_k = -(x[None, :] - problem.x_obs) / (
        problem.bandwidth**2 * problem.scale[None, :] ** 2 + 1e-12
    )
    pred = np.empty(len(problem.property_names), float)
    grad = np.empty((len(problem.property_names), problem.n), float)
    for j in range(len(problem.property_names)):
        valid = np.isfinite(problem.y_obs[:, j])
        w = kernel[valid]
        a = w / max(w.sum(), 1e-300)
        y = problem.y_obs[valid, j]
        pred[j] = float(a @ y)
        grad[j] = np.sum((a * (y - pred[j]))[:, None] * grad_log_k[valid], axis=0)
    return pred, grad, d2, kernel


def objective_and_gradient(problem: Problem, x: np.ndarray, penalty_weight: float):
    pred, dpdx, d2, kernel = predict_with_gradient(problem, x)
    q = 0.0
    dq = np.zeros(problem.n, float)
    for term in problem.cfg["objective_terms"]:
        weight = float(term["weight"])
        if term["kind"] == "property_max":
            j = int(term["index"])
            width = float(np.ptp(problem.windows[j]))
            score = (pred[j] - problem.windows[j, 0]) / width
            q += weight * np.clip(score, 0, 1)
            if 0.0 < score < 1.0:
                dq += weight * dpdx[j] / width
        elif term["kind"] == "property_min":
            j = int(term["index"])
            width = float(np.ptp(problem.windows[j]))
            score = (problem.windows[j, 1] - pred[j]) / width
            q += weight * np.clip(score, 0, 1)
            if 0.0 < score < 1.0:
                dq -= weight * dpdx[j] / width
        else:
            ids = np.asarray(term["indices"], int)
            score = float(np.sum(x[ids]) / float(term["normalizer"]))
            q += weight * np.clip(score, 0, 1)
            if 0.0 < score < 1.0:
                dq[ids] += weight / float(term["normalizer"])
    widths = problem.windows[:, 1] - problem.windows[:, 0]
    low_v = np.maximum((problem.windows[:, 0] - pred) / widths, 0.0)
    high_v = np.maximum((pred - problem.windows[:, 1]) / widths, 0.0)
    penalty = float(np.sum(low_v**2 + high_v**2))
    dpen = np.zeros(problem.n, float)
    for j in range(len(pred)):
        if low_v[j] > 0:
            dpen += -2.0 * low_v[j] * dpdx[j] / widths[j]
        if high_v[j] > 0:
            dpen += 2.0 * high_v[j] * dpdx[j] / widths[j]
    return float(q - penalty_weight * penalty), dq - penalty_weight * dpen, pred, float(q), penalty, d2, kernel


def hit_and_run_step(x, lo, hi, active, rng):
    if len(active) <= 1:
        return x.copy(), True
    direction = rng.normal(size=len(active))
    direction -= direction.mean()
    norm = np.linalg.norm(direction)
    if norm <= 1e-14:
        return x.copy(), False
    direction /= norm
    xa = x[active]
    low_t, high_t = -np.inf, np.inf
    for value, lower, upper, d in zip(xa, lo[active], hi[active], direction):
        if d > 0:
            a, b = (lower - value) / d, (upper - value) / d
        elif d < 0:
            a, b = (upper - value) / d, (lower - value) / d
        else:
            continue
        low_t, high_t = max(low_t, a), min(high_t, b)
    if not np.isfinite(low_t + high_t) or high_t <= low_t + 1e-15:
        return x.copy(), False
    y = x.copy()
    y[active] = xa + rng.uniform(low_t, high_t) * direction
    y[active[-1]] += 1.0 - y.sum()
    return y, True


def sample_starts(problem: Problem, structures: list[int], per_structure: int, rep: int, spec: dict):
    chains_n = int(spec["start_chains_per_structure"])
    if per_structure % chains_n:
        raise ValueError("starts per structure must be divisible by chain count")
    saved = per_structure // chains_n
    designs, structure_ids, transition_rows = [], [], []
    for s in structures:
        mask = problem.structure_masks[s]
        req = problem.required_masks[s]
        lo = np.where(req, problem.step, 0.0)
        hi = np.where(mask, problem.upper, 0.0)
        active = np.flatnonzero(mask)
        x0 = lo.copy()
        residual = 1.0 - x0.sum()
        caps = hi - lo
        if residual > 0:
            x0[active] += residual * caps[active] / caps[active].sum()
        for chain in range(chains_n):
            seed = int(spec["start_seed_base"]) + rep * 100000 + s * 100 + chain
            rng = np.random.default_rng(seed)
            x = x0.copy()
            stalled = 0
            init = int(spec["start_diversification_steps"])
            burn = int(spec["start_burn_in_steps"])
            thin = int(spec["start_thinning_steps"])
            for _ in range(init + burn):
                x, moved = hit_and_run_step(x, lo, hi, active, rng)
                stalled += int(not moved)
            for _ in range(saved):
                for _ in range(thin):
                    x, moved = hit_and_run_step(x, lo, hi, active, rng)
                    stalled += int(not moved)
                designs.append(x.copy())
                structure_ids.append(s)
            transition_rows.append({
                "replicate": rep, "structure": s, "chain": chain, "seed": seed,
                "saved": saved, "transitions": init + burn + saved * thin, "stalled": stalled,
            })
    return np.asarray(designs), np.asarray(structure_ids, int), transition_rows


def standardized_margin(problem: Problem, pred: np.ndarray) -> np.ndarray:
    widths = problem.windows[:, 1] - problem.windows[:, 0]
    return np.min(np.minimum((pred - problem.windows[:, 0]) / widths,
                             (problem.windows[:, 1] - pred) / widths), axis=-1)

