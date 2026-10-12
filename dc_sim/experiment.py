import os
import json
import traceback
import dataclasses
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from .config import FullConfig, DynamicWorkloadConfig
from .workload import generate_infrastructure, generate_workload, generate_dynamic_workload, make_rng_streams
from .simulator import DiscreteEventSimulator, provision_vms_on_pms
from .kbff import KBFuzzyFirefly
from .baselines import make_baseline
from .checkpoint import CheckpointManager
from .validation import validate_result, ValidationOutcome


KBFF_VARIANTS = {
    "KBFF": "full",
    "KBFF_no_fuzzy": "no_fuzzy",
    "KBFF_no_dtkb": "no_dtkb",
    "KBFF_no_energy_fitness": "no_energy_fitness",
    "KBFF_no_adaptive_gate": "no_adaptive_gate",
    "KBFF_w_energy_0": "KBFF_w_energy_0",
}


def _config_to_dict(cfg: FullConfig) -> Dict[str, Any]:
    def _convert(obj):
        if dataclasses.is_dataclass(obj):
            return {f.name: _convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        elif isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    return _convert(cfg)


def build_algorithm(name: str, cfg: FullConfig, rng: np.random.Generator):
    if name in KBFF_VARIANTS:
        return KBFuzzyFirefly(cfg, rng, variant=KBFF_VARIANTS[name])
    return make_baseline(name, cfg, rng)


def _run_one_unit(algorithm_name: str, workload_size: int, seed: int,
                  cfg: FullConfig, dynamic: bool = False) -> Dict[str, Any]:
    import copy
    if algorithm_name == "KBFF_w_energy_0":
        cfg = copy.deepcopy(cfg)
        cfg.firefly.w_energy = 0.0


    rng_infra, rng_work, rng_algo = make_rng_streams(seed)

    pms, vms = generate_infrastructure(cfg.infra, cfg.energy, rng_infra, cfg.scheduling)
    provision_vms_on_pms(pms, vms, rng_infra, policy=cfg.infra.provisioning_policy)

    if dynamic:
        tasks = generate_dynamic_workload(cfg.dynamic_workload, rng_work)
    else:
        tasks = generate_workload(cfg.workload, workload_size, rng_work)

    algo = build_algorithm(algorithm_name, cfg, rng_algo)
    hosted_vms = [v for v in vms if v.host_pm is not None]


    if (dynamic and cfg.dynamic_workload.dtkb_train_on_phase0_only
            and hasattr(algo, 'fit_knowledge_base')):
        algo.fit_knowledge_base(hosted_vms, pms, cfg, dynamic=True, seed=seed)
    elif hasattr(algo, "fit_knowledge_base"):
        algo.fit_knowledge_base(hosted_vms, pms, cfg, dynamic=False, seed=seed)
    elif hasattr(algo, "prepare"):
        algo.prepare(hosted_vms)

    sim = DiscreteEventSimulator(cfg, pms, vms, tasks, algo, algorithm_name, seed)
    result = sim.run()

    result_dict = result.to_dict()


    if hasattr(algo, "dtkb") and algo.dtkb.model is not None:
        dtkb_out = algo.dtkb.get_validation_outputs()
        result_dict.update({k: v for k, v in dtkb_out.items()
                            if k not in result_dict or not result_dict.get(k)})


    if hasattr(algo, "get_candidate_diagnostics"):
        result_dict.update(algo.get_candidate_diagnostics())


    if dynamic and tasks:
        for ph_name in set(getattr(t, 'phase', 'default') for t in tasks):
            ph_tasks = [t for t in tasks if getattr(t, 'phase', 'default') == ph_name]
            ph_viols = sum(1 for t in ph_tasks if t.sla_violated())
            result_dict[f"sla_violations_{ph_name}"] = ph_viols
            result_dict[f"n_tasks_{ph_name}"] = len(ph_tasks)

    return result_dict


def _worker(args: Tuple) -> Tuple[Dict[str, Any], "ValidationOutcome", Tuple]:
    algorithm_name, workload_size, seed, cfg, unit, dynamic = args
    try:
        r = _run_one_unit(algorithm_name, workload_size, seed, cfg, dynamic=dynamic)
        outcome = validate_result(r, workload_size)
        return r, outcome, unit
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        r = {
            "algorithm": algorithm_name, "n_tasks": workload_size, "seed": seed,
            "error": str(e), "traceback": tb,
        }
        outcome = ValidationOutcome(ok=False, reasons=[f"EXCEPTION: {e}"])
        return r, outcome, unit


class EarlyStopError(RuntimeError):
    def __init__(self, message: str, partial_results: List[Dict[str, Any]]):
        super().__init__(message)
        self.partial_results = partial_results


class Experiment:
    def __init__(self, cfg: FullConfig, algorithms: List[str] = None,
                 checkpoint_dir: str = None,
                 max_consecutive_failures: int = 3,
                 max_total_failure_rate: float = 0.5,
                 min_units_before_rate_check: int = 8,
                 dynamic: bool = False):
        self.cfg = cfg
        self.dynamic = dynamic
        self.algorithms = algorithms or list(cfg.experiment.algorithms) + \
            [a for a in KBFF_VARIANTS if a != "KBFF"]
        self.ckpt = CheckpointManager(checkpoint_dir or cfg.experiment.checkpoint_dir)
        self.max_consecutive_failures = max_consecutive_failures
        self.max_total_failure_rate = max_total_failure_rate
        self.min_units_before_rate_check = min_units_before_rate_check

    def _run_one(self, algorithm_name: str, workload_size: int, seed: int) -> Dict[str, Any]:
        return _run_one_unit(algorithm_name, workload_size, seed, self.cfg, dynamic=self.dynamic)

    def _run_one_safe(self, algorithm_name: str, workload_size: int, seed: int
                       ) -> Tuple[Dict[str, Any], ValidationOutcome]:
        try:
            r = self._run_one(algorithm_name, workload_size, seed)
            outcome = validate_result(r, workload_size)
            return r, outcome
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            r = {
                "algorithm": algorithm_name, "n_tasks": workload_size, "seed": seed,
                "error": str(e), "traceback": tb,
            }
            outcome = ValidationOutcome(ok=False, reasons=[f"EXCEPTION: {e}"])
            return r, outcome

    def save_config_json(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        cfg_dict = _config_to_dict(self.cfg)
        cfg_path = os.path.join(results_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        return cfg_path

    def pilot_check(self, verbose: bool = True) -> pd.DataFrame:
        seeds = list(self.cfg.experiment.seeds)
        workload_sizes = list(self.cfg.experiment.workload_sizes)
        pilot_seed = seeds[0] if seeds else 1
        pilot_workload = min(workload_sizes) if workload_sizes else 100

        rows = []
        pbar = tqdm(self.algorithms, desc="Pilot check (1 seed, smallest workload)",
                    disable=not verbose)
        for algo in pbar:
            pbar.set_postfix(algo=algo)
            import time
            t0 = time.time()
            r, outcome = self._run_one_safe(algo, pilot_workload, pilot_seed)
            elapsed = time.time() - t0
            row = dict(r)
            row["algorithm"] = algo
            row["n_tasks"] = pilot_workload
            row["seed"] = pilot_seed
            row["ok"] = outcome.ok
            row["validation_reasons"] = "; ".join(outcome.reasons)
            row["elapsed_s"] = elapsed
            rows.append(row)
        report = pd.DataFrame(rows)
        n_fail = int((~report["ok"]).sum())
        if verbose:
            if n_fail == 0:
                print(f"[pilot] All {len(report)} algorithms passed the pilot check.")
            else:
                print(f"[pilot] {n_fail}/{len(report)} algorithm(s) FAILED the pilot check:")
                for _, row in report[~report["ok"]].iterrows():
                    print(f"  - {row['algorithm']}: {row['validation_reasons']}")
        return report

    def run_all(self, seeds: List[int] = None, workload_sizes: List[int] = None,
                verbose: bool = True, fail_fast: bool = True) -> pd.DataFrame:
        seeds = seeds or list(self.cfg.experiment.seeds)
        workload_sizes = workload_sizes or list(self.cfg.experiment.workload_sizes)
        policy = f"{self.cfg.scheduling.policy}__{self.cfg.infra.provisioning_policy}"
        dyn_suffix = "_dynamic" if self.dynamic else ""

        import hashlib
        cfg_dict = _config_to_dict(self.cfg)
        if "experiment" in cfg_dict:
            exp_dict = dict(cfg_dict["experiment"])
            exp_dict.pop("seeds", None)
            exp_dict.pop("checkpoint_dir", None)
            exp_dict.pop("results_dir", None)
            cfg_dict["experiment"] = exp_dict
        cfg_hash = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode('utf-8')).hexdigest()[:8]
        base_ckpt_key = f"{policy}{dyn_suffix}_{cfg_hash}"

        all_units = [(alg, w, s) for w in workload_sizes for alg in self.algorithms for s in seeds]
        total_units_count = len(all_units)

        todo_units = []
        results = []

        if verbose:
            print(f"Checking checkpoints for {total_units_count} total experiment units...")

        for unit in all_units:
            alg, w, s = unit
            if self.ckpt.is_done(alg, w, s, base_ckpt_key):
                r = self.ckpt.load_result(alg, w, s, base_ckpt_key)
                results.append(r)
            else:
                todo_units.append(unit)

        already_done_count = len(results)
        need_to_run_count = len(todo_units)

        if verbose:
            print(f"Skipping {already_done_count} completed units. "
                  f"Running {need_to_run_count} remaining units.")

        if not todo_units:
            return pd.DataFrame(results)


        num_workers = min(10, need_to_run_count)

        if verbose:
            print(f"Starting parallel execution using {num_workers} worker process(es).")

        worker_args = [
            (alg, w, s, self.cfg, (alg, w, s), self.dynamic)
            for alg, w, s in todo_units
        ]

        total_failures = 0
        stopped_early = False

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker, args): args for args in worker_args}
            pbar = tqdm(as_completed(futures), total=need_to_run_count,
                        desc="KBFF Parallel Grid", disable=not verbose)

            for future in pbar:
                try:
                    result_dict, outcome, unit = future.result()
                except Exception as exc:
                    tqdm.write(f"[WORKER CRASH] {exc}")
                    total_failures += 1
                    continue

                alg, w, s = unit
                pbar.set_postfix(algo=alg, w=w, seed=s)

                if outcome.ok:
                    self.ckpt.save_result(alg, w, s, result_dict, base_ckpt_key)
                    results.append(result_dict)
                else:
                    total_failures += 1
                    if verbose:
                        tqdm.write(f"[VALIDATION FAILED] {alg} (w={w}, s={s}): "
                                   f"{'; '.join(outcome.reasons)}")

                current_total = len(results) + total_failures
                if fail_fast and current_total >= self.min_units_before_rate_check:
                    if (total_failures / current_total) > self.max_total_failure_rate:
                        tqdm.write(f"\n[EARLY STOP] High failure rate detected. "
                                   f"Cancelling remaining futures.")
                        for f in futures:
                            f.cancel()
                        stopped_early = True
                        break

        df = pd.DataFrame(results)
        df.attrs["stopped_early"] = stopped_early
        return df

    def load_existing(self) -> pd.DataFrame:
        return pd.DataFrame(self.ckpt.all_results())
