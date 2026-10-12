import os
import sys
import json
import time
import argparse
import dataclasses
import numpy as np
import pandas as pd

from dc_sim.config import (
    FullConfig, InfraConfig, WorkloadConfig, EnergyModelConfig,
    ExperimentConfig, SchedulingConfig
)
from dc_sim.experiment import Experiment, _run_one_unit
from dc_sim.stats import (
    summarise, ablation_table, convergence_curve_table,
    sla_by_priority_table, energy_summary_table, dynamic_phase_table,
    paired_significance
)

SEEDS_30 = tuple(range(1, 31))


MAIN_ALGORITHMS = [
    "KBFF", "KBFF_no_fuzzy", "KBFF_no_dtkb", "KBFF_no_energy_fitness",
    "KBFF_no_adaptive_gate", "KBFF_w_energy_0",
    "KB_FPA", "Fuzzy_SLW",
    "BFD", "MBF", "UtilBand", "Greedy_Fitness",
    "PSO", "ACO", "GA", "DE",
    "OriginalPSO", "OriginalGA", "OriginalDE", "OriginalACO", "OriginalFA",
    "Pure_FA", "Fuzzy_Only", "KB_Only",
]

ABLATION_VARIANTS = [
    "KBFF_no_fuzzy", "KBFF_no_dtkb", "KBFF_no_energy_fitness", "KBFF_no_adaptive_gate"
]


def make_base_config(seeds=SEEDS_30) -> FullConfig:
    cfg = FullConfig()
    cfg.experiment.seeds = seeds
    cfg.experiment.workload_sizes = (1000, 3000, 5000)
    cfg.experiment.algorithms = tuple(MAIN_ALGORITHMS)
    return cfg


def make_stress_config(seeds=SEEDS_30) -> FullConfig:
    cfg = make_base_config(seeds)
    cfg.infra.n_pms = 50
    cfg.infra.n_vms = 200
    cfg.workload.mean_interarrival_s = 0.038

    cfg.workload.deadline_slack_ranges = (
        (1.2, 2.0),
        (1.05, 1.5),
        (1.02, 1.2),
    )
    cfg.experiment.workload_sizes = (10000,)
    return cfg


def make_sensitivity_cfg(idle_fraction: float, sleep_fraction: float,
                          seeds=SEEDS_30) -> FullConfig:
    cfg = make_base_config(seeds)
    cfg.energy.idle_fraction = idle_fraction
    cfg.energy.sleep_fraction = sleep_fraction
    cfg.experiment.workload_sizes = (1000,)
    return cfg


def make_dynamic_config(seeds=SEEDS_30) -> FullConfig:
    cfg = make_base_config(seeds)
    cfg.infra.n_pms = 50
    cfg.infra.n_vms = 200
    cfg.experiment.workload_sizes = (10000,)
    cfg.dynamic_workload.dtkb_train_on_phase0_only = True
    return cfg


def run_timing_pass(cfg: FullConfig, results_dir: str = "results_final"):
    print("\n--- Running Single-Process Timing Pass (comment #5.3) ---")
    timing_cfg = make_base_config(seeds=(1, 2, 3))
    timing_cfg.experiment.workload_sizes = (1000,)
    timing_cfg.experiment.algorithms = tuple(MAIN_ALGORITHMS)

    exp = Experiment(timing_cfg, algorithms=MAIN_ALGORITHMS)


    print("Pre-warming caches for consistent timing...")
    for s in (1, 2, 3):
        try:
            exp._run_one("KBFF", 100, s)
        except Exception:
            pass

    times = []
    for algo in MAIN_ALGORITHMS:
        for seed in (1, 2, 3):
            t0 = time.perf_counter()
            try:
                r = exp._run_one(algo, 1000, seed)
                t1 = time.perf_counter()
                total_wall = t1 - t0
                sim_wall = r.get("wall_clock_runtime_s", total_wall)
                tasks_alloc = max(r.get("tasks_allocated", 1), 1)
                times.append({
                    "algorithm": algo, "seed": seed,
                    "wall_clock_runtime_s": sim_wall,
                    "total_overhead_s": total_wall,
                    "per_task_ms": (sim_wall / tasks_alloc) * 1000,
                })
            except Exception as e:
                print(f"  [TIMING ERROR] {algo} seed={seed}: {e}")

    if not times:
        return pd.DataFrame()

    df_times = pd.DataFrame(times)

    timing_summary = df_times.groupby("algorithm").agg(
        median_wall_s=("wall_clock_runtime_s", "median"),
        iqr_wall_s=("wall_clock_runtime_s", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        median_per_task_ms=("per_task_ms", "median"),
    ).reset_index()

    os.makedirs(results_dir, exist_ok=True)
    timing_summary.to_csv(os.path.join(results_dir, "timing_summary.csv"), index=False)
    print("Timing summary:")
    print(timing_summary.to_string(index=False))
    return df_times, timing_summary


def build_final_excel(all_sheets: dict, output_path: str = "Final.xlsx"):
    print(f"\nWriting Final.xlsx to {output_path}")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in all_sheets.items():
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue

            sn = sheet_name[:31]
            df.to_excel(writer, sheet_name=sn, index=False)
            print(f"  Written sheet: '{sn}' ({len(df)} rows)")
    print(f"Final.xlsx saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="KBFF experiment runner")
    parser.add_argument("--main", action="store_true", help="Main experiment only")
    parser.add_argument("--timing", action="store_true", help="Timing pass only")
    parser.add_argument("--stress", action="store_true", help="Stress config only")
    parser.add_argument("--sensitivity", action="store_true", help="Sensitivity sweep only")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic workload only")
    parser.add_argument("--pilot", action="store_true", help="Pilot check only")
    parser.add_argument("--seed-start", type=int, default=1, help="Starting seed (inclusive)")
    parser.add_argument("--seed-end", type=int, default=30, help="Ending seed (inclusive)")
    parser.add_argument("--output", default="Final.xlsx", help="Output Excel file")
    parser.add_argument("--ckpt-dir", default="checkpoints_final", help="Checkpoint directory")
    args = parser.parse_args()

    seeds = tuple(range(args.seed_start, args.seed_end + 1))
    all_sheets = {}
    results_dir = "results_final"
    os.makedirs(results_dir, exist_ok=True)


    if not any([args.main, args.timing, args.stress, args.sensitivity, args.dynamic, args.pilot]):
        run_all = True
    else:
        run_all = False

    if run_all or args.main or not any([args.stress, args.sensitivity, args.dynamic, args.pilot, args.timing]):
        print("\n" + "="*60)
        print("PART 1: MAIN EXPERIMENT")
        print("="*60)
        cfg_main = make_base_config(seeds)
        cfg_main.experiment.checkpoint_dir = args.ckpt_dir

        exp_main = Experiment(cfg_main, algorithms=MAIN_ALGORITHMS)
        cfg_path = exp_main.save_config_json(results_dir)
        print(f"Config saved to: {cfg_path}")

        df_main = exp_main.run_all()
        df_main.to_csv(os.path.join(results_dir, "raw_main.csv"), index=False)
        df_main.to_csv("raw_main.csv", index=False)


        sum_main = summarise(df_main)
        all_sheets["Main_Summary"] = sum_main


        all_sheets["Main_Raw"] = df_main


        try:
            from dc_sim.legacy_names import build_legacy_outputs
            legacy_dict = build_legacy_outputs(df_main, sum_main)
            if "table4_vm_utilization" in legacy_dict:
                all_sheets["Table4_VM_Util"] = legacy_dict["table4_vm_utilization"]
            if "table5_pm_utilization" in legacy_dict:
                all_sheets["Table5_PM_Util"] = legacy_dict["table5_pm_utilization"]
            if "table4c_vm_utilization_fleet" in legacy_dict:
                all_sheets["Table4c_VM_Fleet"] = legacy_dict["table4c_vm_utilization_fleet"]
            if "table5c_pm_utilization_fleet" in legacy_dict:
                all_sheets["Table5c_PM_Fleet"] = legacy_dict["table5c_pm_utilization_fleet"]
            if "fig14_energy_kw" in legacy_dict:
                all_sheets["Fig14_Energy_kW"] = legacy_dict["fig14_energy_kw"]
            if "summary_legacy_named" in legacy_dict:
                all_sheets["Legacy_Summary"] = legacy_dict["summary_legacy_named"]
        except Exception as e:
            print(f"Warning: could not generate legacy sheets: {e}")


        if not df_main.empty:
            abl = ablation_table(df_main, ABLATION_VARIANTS)
            all_sheets["Ablation"] = abl


        sla_pri = sla_by_priority_table(df_main)
        all_sheets["SLA_by_Priority"] = sla_pri


        eng_sum = energy_summary_table(df_main)
        all_sheets["Energy_kWh"] = eng_sum


        for metric in ["total_energy_kwh", "sla_violations", "vm_utilisation_active_pct"]:
            sig = paired_significance(df_main, proposed="KBFF", metric=metric)
            all_sheets[f"Sig_{metric[:20]}"] = sig

        print(f"Main experiment: {len(df_main)} runs completed.")


    if run_all or args.timing:
        print("\n" + "="*60)
        print("TIMING PASS (comment #5.3)")
        print("="*60)
        timing_result = run_timing_pass(make_base_config(seeds=(1, 2, 3)), results_dir)
        if timing_result and isinstance(timing_result, tuple):
            df_times, timing_summary = timing_result
            all_sheets["Timing_Raw"] = df_times
            all_sheets["Timing_Summary"] = timing_summary


    if run_all or args.stress:
        print("\n" + "="*60)
        print("STRESS TEST (comment #3)")
        print("="*60)
        cfg_stress = make_stress_config(seeds)
        cfg_stress.experiment.checkpoint_dir = args.ckpt_dir + "_stress"

        exp_stress = Experiment(cfg_stress, algorithms=MAIN_ALGORITHMS)
        df_stress = exp_stress.run_all()
        df_stress.to_csv(os.path.join(results_dir, "raw_stress.csv"), index=False)

        all_sheets["Stress_Summary"] = summarise(df_stress)
        all_sheets["Stress_Raw"] = df_stress
        all_sheets["Stress_SLA_Priority"] = sla_by_priority_table(df_stress)
        print(f"Stress experiment: {len(df_stress)} runs completed.")


    if run_all or args.sensitivity:
        print("\n" + "="*60)
        print("SENSITIVITY SWEEP (comment #2.4)")
        print("="*60)
        idle_vals = [0.0, 0.3, 0.55, 0.7]
        sleep_vals = [0.0, 0.05]
        sensitivity_rows = []

        for idle_f in idle_vals:
            for sleep_f in sleep_vals:
                print(f"  idle={idle_f}, sleep={sleep_f}")
                cfg_sens = make_sensitivity_cfg(idle_f, sleep_f, seeds=seeds[:5])
                cfg_sens.experiment.checkpoint_dir = (
                    args.ckpt_dir + f"_sens_idle{idle_f}_sleep{sleep_f}"
                )
                exp_s = Experiment(cfg_sens, algorithms=["KBFF", "BFD", "Pure_FA"])
                df_s = exp_s.run_all()
                df_s["idle_fraction"] = idle_f
                df_s["sleep_fraction"] = sleep_f
                sensitivity_rows.append(df_s)

        if sensitivity_rows:
            df_sens = pd.concat(sensitivity_rows, ignore_index=True)
            df_sens.to_csv(os.path.join(results_dir, "raw_sensitivity.csv"), index=False)
            all_sheets["Sensitivity_Raw"] = df_sens
            sum_sens = summarise(df_sens, group_cols=("algorithm", "idle_fraction", "sleep_fraction"))
            all_sheets["Sensitivity_Summary"] = sum_sens
            print(f"Sensitivity sweep: {len(df_sens)} runs completed.")


    if run_all or args.dynamic:
        print("\n" + "="*60)
        print("PART 2: DYNAMIC PHASE WORKLOAD")
        print("="*60)
        cfg_dyn = make_dynamic_config(seeds)
        cfg_dyn.experiment.checkpoint_dir = args.ckpt_dir + "_dynamic"

        exp_dyn = Experiment(cfg_dyn, algorithms=MAIN_ALGORITHMS, dynamic=True)
        cfg_path_dyn = exp_dyn.save_config_json(os.path.join(results_dir, "dynamic"))
        print(f"Dynamic config saved to: {cfg_path_dyn}")


        df_dyn = exp_dyn.run_all(workload_sizes=[10000])
        df_dyn.to_csv(os.path.join(results_dir, "raw_dynamic.csv"), index=False)

        all_sheets["Dynamic_Summary"] = summarise(df_dyn)
        all_sheets["Dynamic_Raw"] = df_dyn
        all_sheets["Dynamic_Phase_SLA"] = dynamic_phase_table(df_dyn)


        if "mean_lambda" in df_dyn.columns:
            lam_sum = summarise(df_dyn[["algorithm", "n_tasks", "mean_lambda", "seed"]]
                                .rename(columns={"mean_lambda": "dummy"}), group_cols=("algorithm",))

            lam_table = df_dyn.groupby("algorithm")["mean_lambda"].agg(
                ["mean", "std", "min", "max"]
            ).reset_index()
            all_sheets["Dynamic_Lambda"] = lam_table

        print(f"Dynamic experiment: {len(df_dyn)} runs completed.")


    if args.pilot:
        print("\n" + "="*60)
        print("PILOT CHECK")
        print("="*60)
        cfg_pilot = make_base_config(seeds=(1,))
        exp_pilot = Experiment(cfg_pilot, algorithms=MAIN_ALGORITHMS)
        pilot_report = exp_pilot.pilot_check()


        pilot_report.to_csv("pilot_raw.csv", index=False)
        os.makedirs(results_dir, exist_ok=True)
        pilot_report.to_csv(os.path.join(results_dir, "pilot_raw.csv"), index=False)
        all_sheets["Pilot_Check"] = pilot_report

        print(f"\nPilot results saved to 'pilot_raw.csv' and '{os.path.join(results_dir, 'pilot_raw.csv')}'.")


        mean_runtime_per_unit = float(pilot_report["elapsed_s"].mean()) if "elapsed_s" in pilot_report.columns else 2.0
        n_algos = len(MAIN_ALGORITHMS)
        n_seeds = 30
        n_sizes = 3
        total_main_units = n_algos * n_sizes * n_seeds
        n_workers = 10
        size_multiplier = (1000 + 3000 + 5000) / (3 * 1000)
        est_total_seconds = (total_main_units * mean_runtime_per_unit * size_multiplier) / n_workers
        est_hours = est_total_seconds / 3600.0
        print("\n" + "="*60)
        print("FULL 30-SEED GRID RUN ESTIMATED TIME OF ARRIVAL (ETA):")
        print(f"  - Total units: {total_main_units} ({n_algos} algos x {n_sizes} sizes x {n_seeds} seeds)")
        print(f"  - Parallel processes: {n_workers} workers")
        print(f"  - Measured baseline unit time: {mean_runtime_per_unit:.2f}s")
        print(f"  - Estimated execution time: {est_hours:.2f} hours ({est_total_seconds/60.0:.1f} minutes)")
        print("="*60 + "\n")


    build_final_excel(all_sheets, output_path=args.output)
    print(f"\nAll experiments complete! Results saved to {args.output}")


    if "Main_Summary" in all_sheets and "Main_Raw" in all_sheets:
        try:
            from dc_sim.plots import generate_all_plots
            df_dyn_sheet = all_sheets.get("Dynamic_Raw", None)
            df_stress_sheet = all_sheets.get("Stress_Raw", None)
            generate_all_plots(
                all_sheets["Main_Summary"],
                all_sheets["Main_Raw"],
                out_dir="figures",
                df_dyn=df_dyn_sheet,
                df_stress=df_stress_sheet
            )
        except Exception as e:
            print(f"Warning: could not generate plots: {e}")


if __name__ == '__main__':
    main()
