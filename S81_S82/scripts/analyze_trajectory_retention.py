from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import mpg_common


WORK = Path(__file__).resolve().parents[1]
RESULTS = WORK / "results" / "formal"
ANALYSIS = WORK / "analysis"
ROOT_PROJECT = WORK.parent
MPGA_DIRECT = ROOT_PROJECT / "CMS_v16_focused_analysis_20261014_v1" / "open_source_package" / "results_raw" / "mpga_t_paired_v16_formal" / "formal_v1"
ROUTES = ("Generator_GA", "R_GA", "MPG_GA")
TOLERANCE = 1e-6
HIGH_Q = 0.75


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(structure: int, units: np.ndarray) -> tuple[int, bytes]:
    return int(structure), np.asarray(units, dtype=np.int32).tobytes()


def audit_common(problem, design, structure):
    return problem.audit32(np.asarray(design, np.float64), np.asarray(structure, int))


def dedup(problem, design, structure, lineage, candidate_id):
    units = np.rint(np.asarray(design) / problem.step).astype(np.int32)
    table = {}
    for index, (s, u) in enumerate(zip(structure, units)):
        key = identity(int(s), u)
        if key not in table:
            table[key] = index
    ids = np.asarray(list(table.values()), dtype=int)
    audit = audit_common(problem, np.asarray(design)[ids], np.asarray(structure)[ids])
    return {
        "design": np.asarray(design)[ids], "structure": np.asarray(structure)[ids], "lineage": np.asarray(lineage)[ids],
        "candidate_id": np.asarray(candidate_id)[ids], "coordinate_units": units[ids], **audit,
    }


def representation_metrics(replicate, route, name, data):
    feasible = np.asarray(data["feasible"], bool)
    return {
        "replicate": replicate, "route": route, "representation": name,
        "rows": len(data["design"]), "feasible_rows": int(feasible.sum()),
        "unique_feasible_coordinates": int(feasible.sum()),
        "Q_ge_0_75": int(np.sum(feasible & (data["task_quality"] >= HIGH_Q))),
        "supports": int(np.unique(data["structure"][feasible]).size) if np.any(feasible) else 0,
        "lineages": int(np.unique(data["lineage"][feasible]).size) if np.any(feasible) else 0,
        "mean_Q_feasible": float(np.mean(data["task_quality"][feasible])) if np.any(feasible) else None,
        "max_Q_feasible": float(np.max(data["task_quality"][feasible])) if np.any(feasible) else None,
    }


def best_by_lineage(problem, design, structure, lineage, candidate_id):
    audit = audit_common(problem, design, structure)
    selected = []
    for line in np.unique(lineage):
        ids = np.flatnonzero((lineage == line) & audit["feasible"])
        if not len(ids):
            continue
        q = audit["task_quality"][ids]
        maximum = float(q.max())
        tied = ids[q >= maximum - TOLERANCE]
        # Initial seed (negative ID) precedes proposals; otherwise earliest proposal identity.
        chosen = min(tied, key=lambda index: (0 if int(candidate_id[index]) < 0 else 1, int(candidate_id[index])))
        selected.append(int(chosen))
    selected = np.asarray(selected, int)
    return {
        "design": np.asarray(design)[selected], "structure": np.asarray(structure)[selected],
        "lineage": np.asarray(lineage)[selected], "candidate_id": np.asarray(candidate_id)[selected],
        "coordinate_units": np.rint(np.asarray(design)[selected] / problem.step).astype(np.int32),
        **{key: value[selected] for key, value in audit.items()},
    }


def save_representations(path: Path, representations: dict[str, dict]) -> None:
    arrays = {}
    for name, data in representations.items():
        for key, value in data.items():
            arrays[f"{name}__{key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def analyze_ga(problem):
    metrics, retention, consistency = [], [], []
    for replicate in range(1, 6):
        for route in ROUTES:
            root = RESULTS / f"replicate_{replicate}" / route
            initial = load(root / "initial_population.npz")
            proposals = load(root / "proposals.npz")
            final = load(root / "final_elite_population.npz")
            common_consistency = json.loads((root / "historical_consistency.json").read_text(encoding="utf-8"))
            consistency.append({"replicate": replicate, "route": route, **common_consistency})

            feasible_proposals = proposals["feasible"]
            all_design = np.concatenate([initial["design"], proposals["design"][feasible_proposals]])
            all_structure = np.concatenate([initial["structure"], proposals["structure"][feasible_proposals]])
            all_lineage = np.concatenate([initial["lineage"], proposals["lineage"][feasible_proposals]])
            all_candidate = np.concatenate([initial["candidate_id"], proposals["candidate_id"][feasible_proposals]])
            all_unique = dedup(problem, all_design, all_structure, all_lineage, all_candidate)
            final_unique = dedup(problem, final["design"], final["structure"], final["lineage"], final["candidate_id"])
            best_trajectory = best_by_lineage(problem, all_design, all_structure, all_lineage, all_candidate)
            best_final = best_by_lineage(problem, final["design"], final["structure"], final["lineage"], final["candidate_id"])
            representations = {"all_visit_unique": all_unique, "final_elite_unique": final_unique,
                               "trajectory_best_per_lineage": best_trajectory, "final_best_per_lineage": best_final}
            save_representations(root / "common_representations.npz", representations)
            for name, data in representations.items():
                metrics.append(representation_metrics(replicate, route, name, data))

            all_keys = {identity(s, u) for s, u, ok in zip(all_unique["structure"], all_unique["coordinate_units"], all_unique["feasible"]) if ok}
            final_keys = {identity(s, u) for s, u, ok in zip(final_unique["structure"], final_unique["coordinate_units"], final_unique["feasible"]) if ok}
            high_all = {identity(s, u) for s, u, q, ok in zip(all_unique["structure"], all_unique["coordinate_units"], all_unique["task_quality"], all_unique["feasible"]) if ok and q >= HIGH_Q}
            high_final = {identity(s, u) for s, u, q, ok in zip(final_unique["structure"], final_unique["coordinate_units"], final_unique["task_quality"], final_unique["feasible"]) if ok and q >= HIGH_Q}
            best_all_by_line = {int(line): float(q) for line, q in zip(best_trajectory["lineage"], best_trajectory["task_quality"])}
            best_final_by_line = {int(line): float(q) for line, q in zip(best_final["lineage"], best_final["task_quality"])}
            lost_best = [line for line, q in best_all_by_line.items() if q - best_final_by_line.get(line, -np.inf) > TOLERANCE]
            tied_not_same_identity = 0
            final_best_keys = {int(line): identity(s, u) for line, s, u in zip(best_final["lineage"], best_final["structure"], best_final["coordinate_units"])}
            for line, s, u, q in zip(best_trajectory["lineage"], best_trajectory["structure"], best_trajectory["coordinate_units"], best_trajectory["task_quality"]):
                if abs(float(q) - best_final_by_line.get(int(line), -np.inf)) <= TOLERANCE and identity(s, u) != final_best_keys.get(int(line)):
                    tied_not_same_identity += 1
            retention.append({
                "replicate": replicate, "route": route, "lineages": len(best_all_by_line),
                "highest_Q_lost_lineages": len(lost_best), "highest_Q_retention_rate": 1.0 - len(lost_best) / len(best_all_by_line),
                "tied_best_different_identity_lineages": tied_not_same_identity,
                "all_unique_feasible": len(all_keys), "final_unique_feasible": len(final_keys),
                "unique_identity_omitted_from_final": len(all_keys - final_keys),
                "coordinate_retention_rate": len(final_keys) / len(all_keys),
                "all_high_Q_unique": len(high_all), "final_high_Q_unique": len(high_final),
                "high_Q_identity_omitted_from_final": len(high_all - high_final),
                "high_Q_identity_retention_rate": len(high_final) / len(high_all) if high_all else None,
                "trajectory_to_final_compression": len(final_keys) / len(all_keys),
                "trajectory_best_representatives": len(best_trajectory["design"]),
                "final_best_representatives": len(best_final["design"]),
                "trajectory_supports": int(np.unique(all_unique["structure"][all_unique["feasible"]]).size),
                "final_supports": int(np.unique(final_unique["structure"][final_unique["feasible"]]).size),
            })
    write_csv(ANALYSIS / "GA_representation_metrics_by_run.csv", metrics)
    write_csv(ANALYSIS / "GA_retention_and_omission_by_run.csv", retention)
    write_csv(ANALYSIS / "GA_historical_consistency_by_run.csv", consistency)
    return metrics, retention, consistency


def analyze_mpga_direct(problem):
    rows = []
    input_rows = []
    for replicate in range(1, 6):
        root = MPGA_DIRECT / f"replicate_{replicate}"
        endpoint_path = root / "same_rerun_best_endpoint_candidates.npz"
        visits_path = root / "trajectory_all_unique_audited_visits.npz"
        endpoint = load(endpoint_path); visits = load(visits_path)
        # V16 trajectory arrays were compacted as float32. Canonicalize storage tails
        # back to the frozen native lattice before the shared replay audit.
        endpoint_design = np.rint(np.asarray(endpoint["design"], np.float64) / problem.step) * problem.step
        visit_design = np.rint(np.asarray(visits["design"], np.float64) / problem.step) * problem.step
        endpoint_audit = audit_common(problem, endpoint_design, endpoint["structure"])
        visit_audit = audit_common(problem, visit_design, visits["structure"])
        endpoint_ok = endpoint_audit["feasible"]
        visit_ok = visit_audit["feasible"]
        best_visit = {}
        for index in np.flatnonzero(visit_ok):
            start = int(visits["global_start_id"][index])
            rank = (-float(visit_audit["task_quality"][index]), int(visits["accepted_step"][index]))
            if start not in best_visit or rank < best_visit[start][0]:
                best_visit[start] = (rank, int(index))
        selected = np.asarray([item[1] for item in best_visit.values()], int)
        endpoint_keys = {identity(s, np.rint(x / problem.step).astype(np.int32)) for x, s, ok in zip(endpoint_design, endpoint["structure"], endpoint_ok) if ok}
        visit_keys = {identity(s, np.rint(x / problem.step).astype(np.int32)) for x, s, ok in zip(visit_design, visits["structure"], visit_ok) if ok}
        high_endpoint = {identity(s, np.rint(x / problem.step).astype(np.int32)) for x, s, q, ok in zip(endpoint_design, endpoint["structure"], endpoint_audit["task_quality"], endpoint_ok) if ok and q >= HIGH_Q}
        high_visit = {identity(s, np.rint(x / problem.step).astype(np.int32)) for x, s, q, ok in zip(visit_design, visits["structure"], visit_audit["task_quality"], visit_ok) if ok and q >= HIGH_Q}
        endpoint_by_start = {int(start): float(q) for start, q, ok in zip(endpoint["global_start_id"], endpoint_audit["task_quality"], endpoint_ok) if ok}
        lost_best = 0
        for start, (_, index) in best_visit.items():
            if float(visit_audit["task_quality"][index]) - endpoint_by_start.get(start, -np.inf) > TOLERANCE:
                lost_best += 1
        rows.append({
            "replicate": replicate, "starts": len(endpoint["design"]),
            "endpoint_feasible": int(endpoint_ok.sum()), "endpoint_unique_feasible": len(endpoint_keys),
            "all_visit_rows": len(visits["design"]), "all_visit_feasible_rows": int(visit_ok.sum()),
            "all_visit_unique_feasible": len(visit_keys), "best_feasible_visit_representatives": len(selected),
            "starts_where_best_feasible_visit_exceeds_endpoint_Q": lost_best,
            "visit_identity_omitted_from_endpoint": len(visit_keys - endpoint_keys),
            "endpoint_identity_not_in_visit_pool": len(endpoint_keys - visit_keys),
            "all_visit_high_Q_unique": len(high_visit), "endpoint_high_Q_unique": len(high_endpoint),
            "high_Q_visit_identity_omitted_from_endpoint": len(high_visit - high_endpoint),
            "endpoint_definition": "highest continuous penalized objective over step0 and accepted Armijo iterates; not final step",
            "comparison_warning": "MPGA endpoint is one point per start; GA final elite is a set per lineage",
        })
        input_rows.extend([
            {"replicate": replicate, "artifact": "endpoint", "path": str(endpoint_path), "sha256": sha256(endpoint_path), "rows": len(endpoint["design"])},
            {"replicate": replicate, "artifact": "all_unique_audited_visits", "path": str(visits_path), "sha256": sha256(visits_path), "rows": len(visits["design"])},
        ])
    write_csv(ANALYSIS / "MPGA_endpoint_trajectory_common_audit_by_run.csv", rows)
    write_csv(ANALYSIS / "MPGA_direct_input_identity.csv", input_rows)
    return rows


def summarize(retention, mpga_rows, consistency):
    route_summary = []
    for route in ROUTES:
        subset = [row for row in retention if row["route"] == route]
        for metric in ["highest_Q_lost_lineages", "unique_identity_omitted_from_final", "high_Q_identity_omitted_from_final",
                       "coordinate_retention_rate", "high_Q_identity_retention_rate", "trajectory_supports", "final_supports"]:
            values = np.asarray([row[metric] for row in subset if row[metric] is not None], float)
            route_summary.append({"route": route, "metric": metric, "mean": float(values.mean()),
                                  "sample_sd": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max()), "runs": len(values)})
    write_csv(ANALYSIS / "GA_five_run_summary.csv", route_summary)
    strict = all(row["strict_identity_reproduction"] for row in consistency)
    total_lost = sum(int(row["highest_Q_lost_lineages"]) for row in retention)
    total_high_omitted = sum(int(row["high_Q_identity_omitted_from_final"]) for row in retention)
    report = f"""# 工业PP GA完整轨迹重放、候选保留与MPGA统一口径分析

## 实验身份

三条路线各5次正式GA均复用原正式上游池、选种索引、随机种子、40,000次尝试预算、冻结预测器、Q、窗口、投影、审核与24成员精英策略。新增内容仅为完整轨迹日志。短程非干扰闸门逐坐标通过；15次正式重放的历史接受候选身份严格复现状态为 `{strict}`。原正式文件和论文未覆盖。

## 主要结果

- 15次运行合计发生最高Q严格遗漏的谱系数：{total_lost}（同分容差1e-6）。该指标区分“最高Q丢失”与其他身份淘汰。
- Q≥0.75但未进入最终精英集合的中间唯一坐标合计：{total_high_omitted}。这表示记录策略会压缩其他优质身份，不等于每条谱系的最高Q被丢失。
- 完整访问库、最终24成员精英集合、每谱系轨迹最高Q代表和每谱系最终最高Q代表均已用共同NumPy float32冻结审核重算并保存。逐运行计数、支持覆盖、压缩率与高Q保留率见CSV。
- MPGA直接优化采用同次重跑的端点和完整接受轨迹：端点定义为连续惩罚目标历史最优点，并非最后一步；GA最终精英是集合。二者共同审核结果仅用于结构性保留比较，不能解释为等起点、等预算或算法因果优越。

## 边界

预测Q不是实测性能。候选或坐标不是独立统计重复，五次完整运行才是重复单位。不同算法的起点、更新和成本不同；本报告不声称等算力比较。后续边际收益对照将另行冻结成本协议。
"""
    (WORK / "report" / "工业PP_GA完整轨迹与候选保留分析_20261106.md").write_text(report, encoding="utf-8")
    judgment = {"historical_strict_identity_all_15": strict, "highest_Q_lost_lineages_total": total_lost,
                "high_Q_identity_omitted_total": total_high_omitted, "quality_tie_tolerance": TOLERANCE,
                "high_quality_threshold": HIGH_Q, "mpga_runs": len(mpga_rows)}
    atomic_json(WORK / "status" / "TRAJECTORY_ANALYSIS_COMPLETED.json", judgment)
    print(json.dumps(judgment, ensure_ascii=False, indent=2))


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    problem = mpg_common.Problem()
    _, retention, consistency = analyze_ga(problem)
    mpga_rows = analyze_mpga_direct(problem)
    summarize(retention, mpga_rows, consistency)


if __name__ == "__main__":
    main()
