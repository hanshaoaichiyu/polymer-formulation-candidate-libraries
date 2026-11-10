from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import run_marginal_contrasts as marginal
import run_traced_ga


WORK = Path(__file__).resolve().parents[1]


def run(replicate: int) -> None:
    root = WORK / "results" / "marginal" / "cost_supplement" / f"replicate_{replicate}"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    device = marginal.frozen_core.torch.device("cuda" if marginal.frozen_core.torch.cuda.is_available() else "cpu")
    ga_cost = {}
    for route in ("MPG_GA", "Generator_GA"):
        problem, pool, selected, audit_fn, mutate_fn, protocol, _ = run_traced_ga.load_route(replicate, route, device)
        run_traced_ga.baseline_short(problem, pool, selected[:4], replicate, route, audit_fn, mutate_fn, protocol, 512)
        marginal.synchronize(); started = time.perf_counter()
        accepted = run_traced_ga.baseline_short(problem, pool, selected, replicate, route, audit_fn, mutate_fn, protocol, 40000)
        marginal.synchronize()
        ga_cost[route] = {"seconds": time.perf_counter() - started, "attempts": 40000, "accepted": len(accepted[0])}
    sixth, sixth_cost = marginal.run_generator_batch(replicate, batch=5, count=8000, collect=True)
    marginal.save_npz(root / "B2_sixth_generator_batch.npz", **sixth)
    receipt = {
        "replicate": replicate,
        "GA_no_logging_cost": ga_cost,
        "B2_sixth_batch_cost": sixth_cost,
        "cost_only_amendment": "configs/COST_ONLY_AMENDMENT_BEFORE_OUTCOME_ANALYSIS.json",
        "performance_outcomes_not_read_for_budget_choice": True,
        "serialization_excluded_from_seconds": True,
    }
    marginal.atomic_json(root / "COMPLETED.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, required=True, choices=range(1, 6))
    args = parser.parse_args()
    run(args.replicate)


if __name__ == "__main__":
    main()
