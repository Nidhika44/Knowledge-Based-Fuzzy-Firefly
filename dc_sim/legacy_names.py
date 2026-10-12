import pandas as pd

ALGORITHM_NAME_MAP = {
    "KBFF": "KB-Fuzzy Firefly",
    "KB_FPA": "KB-FPA",
    "Fuzzy_SLW": "Fuzzy-SLW",
    "KBFF_no_fuzzy": "KB-Fuzzy Firefly (no fuzzy stage, ablation)",
    "KBFF_no_dtkb": "KB-Fuzzy Firefly (no DTKB stage, ablation)",
    "KBFF_no_energy_fitness": "KB-Fuzzy Firefly (original priority-only FA fitness, ablation)",
    "KBFF_no_adaptive_gate": "KB-Fuzzy Firefly (naive union gate, ablation)",
    "KBFF_w_energy_0": "KB-Fuzzy Firefly (w_energy=0, ablation)",
    "BFD": "Best-Fit-Decreasing",
    "MBF": "Modified Best Fit",
    "DVFS": "DVFS-Aware Consolidation",
    "UtilBand": "UtilBand (Consolidation Heuristic)",
    "PSO": "PSO",
    "ACO": "ACO",
    "GA": "GA",
    "DE": "DE",
    "Pure_FA": "Pure Firefly (unfiltered)",
    "OriginalPSO": "Original PSO",
    "OriginalGA": "Original GA",
    "OriginalDE": "Original DE",
    "OriginalACO": "Original ACO",
    "OriginalFA": "Original Firefly",
    "Capacity_Filter": "Hard Capacity Filter",
    "Greedy_Fitness": "Greedy Fitness (Argmax Baseline)",
    "Fuzzy_Only": "Fuzzy-Only",
    "KB_Only": "KB-Only",
}

MEATRIC_NOTE = ""

METRIC_NAME_MAP = {
    "vm_utilisation_active_pct_mean": "VM_Utilization_%",
    "pm_utilisation_active_pct_mean": "PM_Utilization_%",
    "vm_utilisation_active_pct_std": "VM_Utilization_%_StdDev",
    "pm_utilisation_active_pct_std": "PM_Utilization_%_StdDev",
    "vm_utilisation_active_pct_ci95": "VM_Utilization_%_CI95",
    "pm_utilisation_active_pct_ci95": "PM_Utilization_%_CI95",
    "vm_utilisation_pct_mean": "VM_Utilization_%_TimeIntegral",
    "pm_utilisation_pct_mean": "PM_Utilization_%_TimeIntegral",
    "vm_utilisation_pct_std": "VM_Utilization_%_TimeIntegral_StdDev",
    "pm_utilisation_pct_std": "PM_Utilization_%_TimeIntegral_StdDev",
    "vm_utilisation_pct_ci95": "VM_Utilization_%_TimeIntegral_CI95",
    "pm_utilisation_pct_ci95": "PM_Utilization_%_TimeIntegral_CI95",
    "vm_utilisation_fleet_pct_mean": "VM_Utilization_%_Fleet",
    "pm_utilisation_fleet_pct_mean": "PM_Utilization_%_Fleet",
    "vm_utilisation_fleet_pct_std": "VM_Utilization_%_Fleet_StdDev",
    "pm_utilisation_fleet_pct_std": "PM_Utilization_%_Fleet_StdDev",
    "vm_utilisation_fleet_pct_ci95": "VM_Utilization_%_Fleet_CI95",
    "pm_utilisation_fleet_pct_ci95": "PM_Utilization_%_Fleet_CI95",
    "sla_violations_mean": "SLA_Violations_Count",
    "migrations_mean": "Migration_Count",
    "makespan_s_mean": "Makespan_seconds",
    "resource_wastage_pct_mean": "Resource_Wastage_%",
    "active_pm_count_mean": "Active_PM_Count",
    "wall_clock_runtime_s_mean": "Algorithm_Runtime_seconds",
}


def to_legacy_wide_table(summary_df: pd.DataFrame, metric_mean_col: str,
                          value_name: str) -> pd.DataFrame:
    sub = summary_df[["algorithm", "n_tasks", metric_mean_col]].copy()
    sub["algorithm"] = sub["algorithm"].map(lambda a: ALGORITHM_NAME_MAP.get(a, a))
    sub["n_tasks_label"] = sub["n_tasks"].map(lambda n: f"{n:,} Tasks")
    wide = sub.pivot_table(index="algorithm", columns="n_tasks_label",
                            values=metric_mean_col, aggfunc="mean")

    order = sorted(sub["n_tasks"].unique())
    wide = wide[[f"{n:,} Tasks" for n in order]]
    wide.index.name = "Algorithm"
    wide.columns.name = None
    return wide.reset_index()


def add_legacy_energy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Energy_Consumption_kW"] = out["total_energy_joules"] / out["makespan_s"].replace(0, float("nan")) / 1000.0
    out["Energy_kWh"] = out["total_energy_joules"] / 3600.0 / 1000.0
    return out


def build_legacy_outputs(raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict:
    raw_legacy = add_legacy_energy_columns(raw_df)
    raw_legacy["algorithm"] = raw_legacy["algorithm"].map(lambda a: ALGORITHM_NAME_MAP.get(a, a))

    summary_legacy = summary_df.copy()
    for old, new in METRIC_NAME_MAP.items():
        if old in summary_legacy.columns:
            summary_legacy[new] = summary_legacy[old]

    has_active = "vm_utilisation_active_pct_mean" in summary_df.columns
    vm_col = "vm_utilisation_active_pct_mean" if has_active else "vm_utilisation_pct_mean"
    pm_col = "pm_utilisation_active_pct_mean" if has_active else "pm_utilisation_pct_mean"

    table4 = to_legacy_wide_table(summary_df, vm_col, "VM_Utilization_%")
    table5 = to_legacy_wide_table(summary_df, pm_col, "PM_Utilization_%")

    outputs = {
        "raw_legacy_named": raw_legacy,
        "summary_legacy_named": summary_legacy,
        "table4_vm_utilization": table4,
        "table5_pm_utilization": table5,
    }

    if has_active:
        outputs["table4b_vm_utilization_time_integral"] = to_legacy_wide_table(
            summary_df, "vm_utilisation_pct_mean", "VM_Utilization_%_TimeIntegral")
        outputs["table5b_pm_utilization_time_integral"] = to_legacy_wide_table(
            summary_df, "pm_utilisation_pct_mean", "PM_Utilization_%_TimeIntegral")

    has_fleet = "vm_utilisation_fleet_pct_mean" in summary_df.columns
    if has_fleet:
        outputs["table4c_vm_utilization_fleet"] = to_legacy_wide_table(
            summary_df, "vm_utilisation_fleet_pct_mean", "VM_Utilization_%_Fleet")
        outputs["table5c_pm_utilization_fleet"] = to_legacy_wide_table(
            summary_df, "pm_utilisation_fleet_pct_mean", "PM_Utilization_%_Fleet")

    energy_summary = raw_df.copy()
    energy_summary = add_legacy_energy_columns(energy_summary)
    energy_grouped = energy_summary.groupby(["algorithm", "n_tasks"], as_index=False)["Energy_Consumption_kW"].mean()
    fig14 = energy_grouped.copy()
    fig14["algorithm"] = fig14["algorithm"].map(lambda a: ALGORITHM_NAME_MAP.get(a, a))
    fig14_wide = fig14.pivot_table(index="algorithm", columns="n_tasks", values="Energy_Consumption_kW")
    fig14_wide.columns = [f"{int(c):,} Tasks" for c in fig14_wide.columns]
    fig14_wide = fig14_wide.reset_index().rename(columns={"algorithm": "Algorithm"})
    outputs["fig14_energy_kw"] = fig14_wide

    return outputs
