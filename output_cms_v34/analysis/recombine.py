from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUTE = ROOT / "output_cms_v29/reproducibility/three_route"
SOURCE = ROUTE / "full_archive/PES_constraint_synthesis_no_anchor_20261010_v1"
CORE = ROOT / "output_cms_v29/reproducibility/core_diagnostics_v17"
AUDIT = ROOT / "output_cms_v31_endpoint_audit_20261028"
ARCHIVE = ROOT / "output_cms_v20_submission/trajectory_archive_20261024"
CASE = "industrial_pp_100"
ARMS = {"ALL3": 7, "MPGA": 8, "ALL4": 15}
GA = ["G_GA", "R_GA", "GR_GA"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The historical selection module's core import is needed only for type hints here.
sys.modules["core"] = types.ModuleType("core")
historical = load_module("historical_selection",
                         ROUTE / "full_archive_workflow/run_frozen_analysis_full_archive.py")
frozen = load_module("frozen_pp", AUDIT / "run_endpoint_diagnostics.py")
write_csv = historical.write_csv
write_json = historical.write_json


def native(path):
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and path.is_absolute() else path


def sha256(path):
    return historical.sha256(native(path))


def read_json(path):
    return json.loads(native(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def arrays(path):
    with np.load(native(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def check_input(path, manifest, root, identities):
    relative = path.relative_to(root).as_posix()
    observed = sha256(path)
    assert observed == manifest[relative], relative
    identities[str(path.relative_to(ROOT))] = observed


def keys(design, support, step):
    assert np.all(np.isfinite(design)) and np.all((0 <= support) & (support < 80))
    units = np.rint(np.asarray(design, dtype=np.float64) / step)
    tail = float(np.max(np.abs(design - units * step)))
    assert tail < step.min() * 1e-3 and np.all((0 <= units) & (units <= 10000))
    output = np.column_stack((support, units)).astype("<u2")
    return output, tail


def audit_union(unique, problem):
    design = unique[:, 1:] * problem.step
    structure = unique[:, 0]
    composition = problem.composition(design, structure, lattice=True)["all_ok"]
    # Same squared Euclidean kernel, evaluated by the Gram identity in float64.
    x = design / problem.scale
    observed = problem.x_obs / problem.scale
    observed_norm = (observed * observed).sum(1)
    predictions = np.empty((len(x), 5), dtype=np.float64)
    for start in range(0, len(x), 4096):
        z = x[start:start+4096]
        d2 = np.maximum((z*z).sum(1)[:, None] + observed_norm - 2*(z @ observed.T), 0.0)
        for j in range(5):
            valid = np.isfinite(problem.y_obs[:, j])
            logw = -d2[:, valid] / (2*problem.bandwidth**2 + 1e-12)
            weight = np.exp(logw - logw.max(1, keepdims=True))
            predictions[start:start+len(z), j] = weight @ problem.y_obs[valid, j] / weight.sum(1)
    window = problem.windows_state(predictions)["per"].all(1)
    return design, structure, composition, window, predictions, problem.score(predictions)


def quota_selection(union, problem, old_common, old_target):
    cap = historical.PROTOCOL["selection"]["case_template_caps"][CASE]
    orders = {}
    for arm, mask in ARMS.items():
        ids = np.flatnonzero(union["membership_bits"] & mask)
        orders[arm] = {
            int(s): historical.select_template_order(ids[union["structure"][ids] == s],
                                                    union, problem.scale, cap)
            for s in np.unique(union["structure"][ids])
        }
    capacity = {arm: sum(map(len, supports.values())) for arm, supports in orders.items()}
    common = sorted(set(old_common) & set.intersection(*(set(order) for order in orders.values())))
    common_n = {s: min(old_common[s], *(len(orders[arm][s]) for arm in ARMS)) for s in common}
    target = min(old_target, *capacity.values())
    common_selection, full_selection = {}, {}
    for arm, order in orders.items():
        common_selection[arm] = [i for s in common for i in order[s][:common_n[s]]]
        full_selection[arm] = [order[s][j] for j in range(cap) for s in sorted(order)
                               if j < len(order[s])][:target]
        assert len(full_selection[arm]) == target
    return {"common_support": common_selection, "full_domain": full_selection}, capacity, common_n, target


def compare_run(run, problem, identities, route_manifest, visit_manifest, counts,
                old_supply, old_common, old_target, output):
    print(f"run {run} archive audit", flush=True)
    archived = ARCHIVE / f"replicate_{run}"
    path = archived / "trajectory_all_unique_audited_visits.npz"
    check_input(path, visit_manifest, ARCHIVE, identities)
    endpoint_path = archived / "same_rerun_best_endpoint_candidates.npz"
    check_input(endpoint_path, visit_manifest, ARCHIVE, identities)
    feasible_path = archived / "trajectory_feasible_candidates.npz"
    check_input(feasible_path, visit_manifest, ARCHIVE, identities)
    receipt_path = archived / "COMPLETED.json"
    identities[str(receipt_path.relative_to(ROOT))] = sha256(receipt_path)
    receipt = read_json(receipt_path)

    # Preserve the three historical populations before applying the shared replay.
    ga_runs = []
    for arm in GA:
        folder = SOURCE / f"results_raw/formal_v1/{CASE}/replicate_{run}/methods/{arm}"
        library = folder / "library.npz"
        ga_receipt = folder / "search/receipt.json"
        for path0 in [library, ga_receipt]:
            check_input(path0, route_manifest, ROUTE, identities)
        cost = read_json(ga_receipt)
        assert cost["attempts"] == 40000, cost
        ga_runs.append(arrays(library))

    with np.load(path, allow_pickle=False) as raw:
        visit_rows = np.flatnonzero(raw["feasible"])
        assert np.array_equal(raw["feasible"], raw["structural_ok"] & raw["window_ok"])
        assert np.all(raw["property_window_ok"][visit_rows])
        archive_visits = len(raw["feasible"])
        visit = {k: raw[k][visit_rows] for k in
                 ("design", "structure", "global_start_id", "accepted_step",
                  "predicted_properties", "task_quality")}
    with np.load(feasible_path, allow_pickle=False) as saved_subset:
        for field in set(visit) & set(saved_subset.files):
            assert np.array_equal(visit[field], saved_subset[field]), field
    with np.load(endpoint_path, allow_pickle=False) as endpoint:
        assert len(endpoint["feasible"]) == 8000
        assert np.array_equal(np.unique(visit["global_start_id"]), np.flatnonzero(
            np.bincount(visit["global_start_id"], minlength=8000)))
        ever = np.zeros(8000, bool)
        ever[visit["global_start_id"]] = True
        assert not np.any(endpoint["feasible"] & ~ever)
        endpoint_count = int(endpoint["feasible"].sum())
    assert archive_visits == receipt["saved_all_unique_audited_visits"]
    assert len(visit_rows) == receipt["retained_feasible_records"]
    del raw

    inputs = ga_runs + [visit]
    key_parts = [keys(data["design"], data["structure"], problem.step) for data in inputs]
    all_keys = np.concatenate([k for k, _ in key_parts])
    packed = all_keys.view(np.dtype((np.void, all_keys.dtype.itemsize * all_keys.shape[1]))).ravel()
    _, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
    unique = all_keys[first]
    ix = np.split(inverse, np.cumsum([len(k) for k, _ in key_parts])[:-1])
    original = np.zeros(len(unique), dtype=np.uint8)
    for i, ids in enumerate(ix):
        if i < 3:
            assert len(np.unique(ids)) == len(ids)
        original[ids] |= 1 << i
    assert np.count_nonzero(original & 7) == old_supply
    assert np.count_nonzero(original & 8) == receipt["retained_unique_global_compositions"]

    design, structure, composition, window, pred, score = audit_union(unique, problem)
    assert composition.all(), "Unexpected composition discrepancy"
    # Direct difference formulation on a deterministic coordinate slice checks
    # the numerically equivalent, faster evaluator used for the complete union.
    probe = np.linspace(0, len(design) - 1, 128, dtype=int)
    assert np.allclose(pred[probe], problem.predict64(design[probe]), atol=1e-9, rtol=0)
    keep = composition & window
    membership = np.where(keep, original, 0).astype(np.uint8)
    ga_provenance = np.full((len(unique), 3), -1, dtype=np.int32)
    ga_ancestry = np.zeros((len(unique), 3), dtype=np.uint8)
    for i, ids in enumerate(ix[:3]):
        ga_provenance[ids, i] = np.arange(len(ids))
        ga_ancestry[ids, i] = ga_runs[i]["source_bits"]
    mpga_first = np.full(len(unique), archive_visits, dtype=np.int32)
    np.minimum.at(mpga_first, ix[3], visit_rows)
    mpga_first[(original & 8) == 0] = -1

    audit_rows = []
    for i, (name, ids, data) in enumerate(zip(GA+["MPGA"], ix, inputs)):
        residual = np.abs(pred[ids] - data["predicted_properties"]) / np.ptp(problem.windows, axis=1)
        q_residual = np.abs(score[ids] - data["task_quality"])
        assert residual.max() < 1e-5 and q_residual.max() < 1e-5, name
        audit_rows.append({
            "run": run, "source": name, "saved_originally_admissible_rows": len(ids),
            "original_unique": int(np.count_nonzero(original & (1 << i))),
            "replay_excluded_rows": int((~keep[ids]).sum()),
            "replay_excluded_unique": int(np.count_nonzero(((original & (1 << i)) != 0) & ~keep)),
            "max_normalized_prediction_difference": float(residual.max()),
            "max_Q_difference": float(q_residual.max()),
            "maximum_storage_tail_fraction": key_parts[i][1],
        })
    results = {"design": design, "integer_coords": unique[:, 1:], "structure": structure,
               "task_quality": score, "predicted_properties": pred, "membership_bits": membership}
    np.savez_compressed(output / f"run_{run}_union_ledger.npz",
                        structure=structure, integer_coords=unique[:, 1:],
                        predicted_properties=pred, task_quality=score,
                        original_membership=original, membership_bits=membership,
                        replay_structural_pass=composition, replay_window_pass=window,
                        GA_original_row=ga_provenance, GA_upstream_ancestry_bits=ga_ancestry,
                        MPGA_first_archive_row=mpga_first, MPGA_archive_rows=visit_rows,
                        MPGA_visit_to_union=ix[3], MPGA_start=visit["global_start_id"],
                        MPGA_accepted_step=visit["accepted_step"])
    supply = []
    for arm, mask in ARMS.items():
        yes = (membership & mask) != 0
        supply.append({"run": run, "arm": arm, "unique": int(yes.sum()),
                       "supports": int(len(np.unique(structure[yes]))),
                       "Q_ge_0_75": int(np.sum(yes & (score >= .75)))})
    np.savez_compressed(output / f"run_{run}_replay_exclusions.npz",
                        index=np.flatnonzero(~keep), support=structure[~keep],
                        original_membership=original[~keep], prediction=pred[~keep],
                        window_flags=problem.windows_state(pred[~keep])["per"])
    overlap = [{"run": run, "mask": m, "unique": int(np.sum(membership == m))}
               for m in range(1, 16)]
    work = {"run": run, "GA_attempts": 120000, "MPGA_starts": 8000,
            "MPGA_accepted_updates": receipt["accepted_updates"],
            "MPGA_gradient_evaluations": receipt["gradient_evaluations"],
            "MPGA_backtracking_forward_evaluations": receipt["backtracking_forward_evaluations"],
            "MPGA_extra_terminal_diagnostic_gradient_evaluations":
                receipt["extra_terminal_diagnostic_gradient_evaluations"],
            "MPGA_saved_audited_rows": archive_visits,
            "MPGA_originally_feasible_rows": len(visit_rows),
            "historical_endpoint_pass": endpoint_count,
            "historical_endpoint_failure_with_feasible_visit": int(ever.sum()) - endpoint_count,
            "historical_no_feasible_visit": 8000 - int(ever.sum())}

    selection, capacity, common_n, target = quota_selection(results, problem, old_common, old_target)
    flat = {f"{layer}_{arm}": np.asarray(ids, dtype=int)
            for layer, arms in selection.items() for arm, ids in arms.items()}
    selected = np.unique(np.concatenate(list(flat.values())))
    frequencies = historical.bootstrap_frequency(problem, design[selected], counts)
    boot = dict(zip(map(int, selected), frequencies))
    np.savez_compressed(output / f"run_{run}_selection.npz",
                        **flat, selected_union_ids=selected, bootstrap_pass_frequency=frequencies)
    metrics, templates = [], []
    for layer, arms in selection.items():
        for arm, ids in arms.items():
            row, detailed = historical.metrics_for_selection(CASE, run, layer, arm, ids,
                                                             results, problem, boot)
            row.update(full_capacity=capacity[arm], full_target=target, old_full_target=old_target,
                       common_support_n=len(common_n), old_common_support_n=len(old_common))
            metrics.append(row)
            templates.extend(detailed)
    print(f"run {run} complete", flush=True)
    return supply, overlap, [work], audit_rows, metrics, templates


def main():
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
    output.mkdir(parents=True, exist_ok=False)
    assert sha256(HERE / "FROZEN_PROTOCOL.json") == (HERE / "FROZEN_PROTOCOL.sha256").read_text().strip()
    problem = frozen.FrozenProblem()
    ga_protocol = SOURCE / "configs/protocol_v1.json"
    mpga_protocol = CORE / "research_code/configs/generation_protocol_v1.json"
    ga_cfg, mpga_cfg = [read_json(path)["cases"][CASE] for path in (ga_protocol, mpga_protocol)]
    def scientific_config(config):
        return {key: value for key, value in config.items() if key != "dataset"}
    assert scientific_config(ga_cfg) == scientific_config(mpga_cfg) == scientific_config(problem.cfg)
    ga_csv = SOURCE / ga_cfg["dataset"]
    mpga_csv = AUDIT / "inputs/industrial_pp_100_recipes.csv"
    packaged_mpga_csv = CORE / "research_code" / mpga_cfg["dataset"]
    assert sha256(ga_csv) == sha256(mpga_csv) == sha256(packaged_mpga_csv)
    assert len(problem.structure_masks) == 80 and problem.n == 90 and not problem.cfg.get("ratio_constraints")
    evaluate = ROUTE / f"inputs/evaluator/{CASE}.npz"
    evaluator = arrays(evaluate)
    for key in ["x_obs", "y_obs", "windows", "scale", "bandwidth", "property_names"]:
        np.testing.assert_array_equal(evaluator[key].squeeze(), np.asarray(getattr(problem, key)))
    route_manifest = {r["package_relative_path"]: r["sha256"] for r in
                      read_json(ROUTE / "full_archive_workflow/FULL_ARCHIVE_MANIFEST.json")["files"]}
    visit_manifest = {r["path"]: r["sha256"] for r in read_json(ARCHIVE / "FILE_MANIFEST_SHA256.json")}
    bootstrap = next((ROUTE / "full_archive").glob(f"*/results_summary/case_modules/{CASE}/B_bootstrap_indices_{CASE}.npz"))
    counts = arrays(bootstrap)["counts"]
    np.testing.assert_array_equal(counts, evaluator["bootstrap_counts"])
    old_supply_path = ROOT / "output_cms_v29/figures/drawing_source/inputs/figure5_inputs/full_supply_arm_metrics.csv"
    old_capacity_path = ROUTE / "inputs/selection_capacity_and_counts.csv"
    old_common_path = ROUTE / "inputs/per_support_selection_quota.csv"
    identities = {str(p.relative_to(ROOT)): sha256(p) for p in
                  [ga_protocol, mpga_protocol, ga_csv, mpga_csv, packaged_mpga_csv, evaluate,
                   old_supply_path, old_capacity_path, old_common_path,
                   ROUTE / "configs/FROZEN_PROTOCOL.json", Path(historical.__file__), Path(frozen.__file__)]}
    check_input(bootstrap, route_manifest, ROUTE, identities)
    supply, overlap, work, replay, metrics, detail = [], [], [], [], [], []
    for run in range(1, 6):
        match = lambda r: r["case"] == CASE and int(r["replicate"]) == run and r["arm"] == "ALL"
        old_supply = int(next(r["unique_candidate_n"] for r in read_csv(old_supply_path) if match(r)))
        old_target = int(next(r["full_domain_target_n"] for r in read_csv(old_capacity_path) if match(r)))
        old_common = {int(r["structure"]): int(r["selected_n"]) for r in read_csv(old_common_path)
                      if match(r) and r["layer"] == "common_support_equal_per_template"}
        additions = compare_run(run, problem, identities, route_manifest, visit_manifest, counts,
                                old_supply, old_common, old_target, output)
        for total, new in zip((supply, overlap, work, replay, metrics, detail), additions):
            total.extend(new)
    for k, expected in [("historical_endpoint_pass", 10388),
                        ("historical_endpoint_failure_with_feasible_visit", 12842),
                        ("historical_no_feasible_visit", 16770)]:
        assert sum(row[k] for row in work) == expected
    for name, data in [("supply_by_run", supply), ("membership_by_run", overlap),
                       ("work_and_trajectory_by_run", work), ("audit_replay_by_source", replay),
                       ("selection_by_run", metrics), ("selection_by_support", detail)]:
        write_csv(output / f"{name}.csv", data)
    assert all(sha256(ROOT / path) == digest for path, digest in identities.items())
    write_json(output / "INPUTS_AND_EXECUTION.json", {
        "input_sha256": identities, "unchanged_input_hashes_verified": True,
        "configuration_agreement": ["industrial source bytes and case configuration",
                                    "80 supports, 90 coordinates, frozen windows and evaluator",
                                    "200 frozen bootstrap count arrays"],
        "run_alignment": "archival run index, not shared GA and MPGA starts",
        "python": sys.version, "numpy": np.__version__,
        "protocol_sha256": sha256(HERE / "FROZEN_PROTOCOL.json"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "retrained_or_reoptimized": False})


if __name__ == "__main__":
    main()

