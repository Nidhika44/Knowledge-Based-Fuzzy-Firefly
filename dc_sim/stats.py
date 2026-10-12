import numpy as np
import pandas as pd
from scipy import stats as sstats
from typing import List


METRICS = [
    "vm_utilisation_pct", "pm_utilisation_pct",
    "vm_utilisation_active_pct", "pm_utilisation_active_pct",
    "total_energy_joules", "total_energy_kwh", "mean_power_kw",
    "sla_violations", "sla_violations_low", "sla_violations_med", "sla_violations_high",
    "migrations", "migration_energy_j", "migration_energy_kwh",
    "makespan_s", "resource_wastage_pct",
    "active_pm_count", "wall_clock_runtime_s", "avg_queue_wait_s",
    "queued_task_count",
    "mean_l1_frac", "mean_l2_frac", "mean_final_frac", "mean_lambda",
    "dtkb_test_accuracy", "dtkb_test_precision", "dtkb_test_recall",
    "dtkb_retrain_count",
]


def boot_ci(x, n: int = 10000, rng: np.random.Generator = None,
            level: float = 0.95) -> tuple:
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        mean = float(np.mean(x)) if len(x) > 0 else 0.0
        return mean, mean
    m = rng.choice(x, (n, len(x))).mean(1)
    alpha = (1.0 - level) / 2.0
    return tuple(np.percentile(m, [alpha * 100, (1 - alpha) * 100]))


def cliffs_delta(lst1, lst2):
    m, n = len(lst1), len(lst2)
    if m == 0 or n == 0:
        return 0.0
    dom = 0
    for x in lst1:
        for y in lst2:
            if x > y:
                dom += 1
            elif x < y:
                dom -= 1
    return dom / (m * n)


def summarise(df: pd.DataFrame, group_cols=("algorithm", "n_tasks")) -> pd.DataFrame:
    _rng = np.random.default_rng(0)
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        n = len(g)
        for m in METRICS:
            if m not in g.columns:
                continue
            vals = pd.to_numeric(g[m], errors='coerce').dropna().values
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_lo, ci_hi = boot_ci(vals, rng=_rng) if len(vals) > 1 else (mean, mean)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95_lo"] = ci_lo
            row[f"{m}_ci95_hi"] = ci_hi
            row[f"{m}_ci95"] = (ci_hi - ci_lo) / 2.0
        row["n_seeds"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def paired_significance(df: pd.DataFrame, proposed: str = "KBFF",
                         metric: str = "vm_utilisation_pct",
                         group_col: str = "n_tasks") -> pd.DataFrame:
    rows = []
    for w, g in df.groupby(group_col):
        prop = g[g["algorithm"] == proposed].sort_values("seed")
        if prop.empty:
            continue
        for algo in sorted(g["algorithm"].unique()):
            if algo == proposed:
                continue
            other = g[g["algorithm"] == algo].sort_values("seed")
            merged = prop.merge(other, on="seed", suffixes=("_prop", "_other"))
            if len(merged) < 2:
                continue
            a = merged[f"{metric}_prop"].astype(float).values
            b = merged[f"{metric}_other"].astype(float).values
            diff = a - b
            try:
                t_stat, t_p = sstats.ttest_rel(a, b)
            except Exception:
                t_stat, t_p = np.nan, np.nan
            try:
                w_stat, w_p = sstats.wilcoxon(a, b) if np.any(diff != 0) else (np.nan, 1.0)
            except Exception:
                w_stat, w_p = np.nan, np.nan

            c_delta = cliffs_delta(a, b)
            med_diff = np.median(diff)
            lower_better = {
                "total_energy_joules", "total_energy_kwh", "mean_power_kw",
                "sla_violations", "sla_violations_low", "sla_violations_med", "sla_violations_high",
                "makespan_s", "resource_wastage_pct", "migration_energy_j", "migration_energy_kwh",
                "avg_queue_wait_s", "queued_task_count"
            }
            n_better = np.sum(diff < 0) if metric in lower_better else np.sum(diff > 0)

            rows.append({
                group_col: w, "algorithm": algo, "metric": metric,
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(med_diff),
                "n_better": int(n_better),
                "t_stat": float(t_stat) if t_stat == t_stat else None,
                "t_pvalue": float(t_p) if t_p == t_p else None,
                "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
                "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
                "cliffs_delta": float(c_delta),
                "n_pairs": len(merged),
                "significant_at_0.05": bool(t_p is not None and t_p == t_p and t_p < 0.05),
            })
    if not rows:
        return pd.DataFrame(rows)
    result = pd.DataFrame(rows)
    for col_raw, col_bh in [("t_pvalue", "t_pvalue_bh"),
                              ("wilcoxon_pvalue", "wilcoxon_pvalue_bh")]:
        valid_mask = result[col_raw].notna()
        if valid_mask.sum() >= 2:
            raw_pvals = result.loc[valid_mask, col_raw].values
            result.loc[valid_mask, col_bh] = sstats.false_discovery_control(raw_pvals, method="bh")
        else:
            result[col_bh] = result[col_raw]
    result["significant_at_0.05_bh"] = result["t_pvalue_bh"].apply(
        lambda p: bool(p is not None and p == p and p < 0.05) if p is not None else False
    )
    result["wilcoxon_significant_at_0.05_bh"] = result["wilcoxon_pvalue_bh"].apply(
        lambda p: bool(p is not None and p == p and p < 0.05) if p is not None else False
    )
    return result


def ablation_table(df: pd.DataFrame, variants: List[str], base: str = "KBFF",
                    group_col: str = "n_tasks") -> pd.DataFrame:
    frames = []
    for m in ["vm_utilisation_pct", "pm_utilisation_pct", "pm_utilisation_active_pct",
              "total_energy_joules", "sla_violations", "total_energy_kwh", "mean_power_kw"]:
        sub = df[df["algorithm"].isin(variants + [base])]
        sig = paired_significance(sub, proposed=base, metric=m, group_col=group_col)
        sig["metric"] = m
        frames.append(sig[sig["algorithm"].isin(variants)])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def convergence_curve_table(raw_traces: dict) -> pd.DataFrame:
    rows = []
    for algo, traces in raw_traces.items():
        max_len = max((len(t) for t in traces), default=0)
        for gen in range(max_len):
            vals = [t[gen] for t in traces if len(t) > gen]
            if vals:
                rows.append({"algorithm": algo, "generation": gen,
                             "mean_best_fitness": float(np.mean(vals)),
                             "std_best_fitness": float(np.std(vals))})
    return pd.DataFrame(rows)


def sla_by_priority_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, n_tasks), g in df.groupby(["algorithm", "n_tasks"]):
        row = {"algorithm": algo, "n_tasks": n_tasks}
        for pri_col in ["sla_violations_low", "sla_violations_med", "sla_violations_high"]:
            if pri_col in g.columns:
                vals = pd.to_numeric(g[pri_col], errors='coerce').dropna().values
                row[f"{pri_col}_mean"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                row[f"{pri_col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def energy_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, n_tasks), g in df.groupby(["algorithm", "n_tasks"]):
        row = {"algorithm": algo, "n_tasks": n_tasks}
        for col in ["total_energy_kwh", "mean_power_kw", "migration_energy_kwh"]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors='coerce').dropna().values
                row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def dynamic_phase_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo, g in df.groupby("algorithm"):
        row = {"algorithm": algo}
        for ph in ["P1", "P2", "P3"]:
            viol_col = f"sla_violations_{ph}"
            n_col = f"n_tasks_{ph}"
            if viol_col in g.columns:
                vals = pd.to_numeric(g[viol_col], errors='coerce').dropna().values
                n_vals = pd.to_numeric(g[n_col], errors='coerce').dropna().values if n_col in g.columns else None
                row[f"{ph}_sla_violations_mean"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                row[f"{ph}_sla_rate_mean"] = (
                    float(np.mean(vals / np.maximum(n_vals, 1)))
                    if (n_vals is not None and len(n_vals) > 0) else 0.0
                )
        rows.append(row)
    return pd.DataFrame(rows)
