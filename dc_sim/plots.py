import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})


ALGO_STYLES = {
    "KBFF": {"color": "#D62728", "marker": "o", "lw": 2.2, "label": "KBFF (Proposed)"},
    "KB_FPA": {"color": "#FF7F0E", "marker": "s", "lw": 1.5, "label": "KB-FPA"},
    "Fuzzy_SLW": {"color": "#2CA02C", "marker": "^", "lw": 1.5, "label": "Fuzzy-SLW"},
    "Pure_FA": {"color": "#9467BD", "marker": "v", "lw": 1.5, "label": "Pure Firefly"},
    "GA": {"color": "#1F77B4", "marker": "D", "lw": 1.5, "label": "GA"},
    "PSO": {"color": "#8C564B", "marker": "p", "lw": 1.5, "label": "PSO"},
    "ACO": {"color": "#E377C2", "marker": "h", "lw": 1.5, "label": "ACO"},
    "DE": {"color": "#7F7F7F", "marker": "X", "lw": 1.5, "label": "DE"},
    "Greedy_Fitness": {"color": "#BCBD22", "marker": "<", "lw": 1.4, "label": "Greedy Fitness"},
    "UtilBand": {"color": "#17BECF", "marker": ">", "lw": 1.4, "label": "UtilBand (DVFS)"},
    "BFD": {"color": "#333333", "marker": "x", "lw": 1.4, "label": "BFD (Standard)"},
    "MBF": {"color": "#555555", "marker": "+", "lw": 1.4, "label": "MBF"},
    "KBFF_no_dtkb": {"color": "#8C564B", "marker": "s", "lw": 1.4, "label": "Ablation: No DTKB"},
    "KBFF_no_fuzzy": {"color": "#E377C2", "marker": "^", "lw": 1.4, "label": "Ablation: No Fuzzy"},
    "KBFF_no_adaptive_gate": {"color": "#7F7F7F", "marker": "v", "lw": 1.4, "label": "Ablation: Static Gate"},
}


def _save_fig(fig, out_base_path):
    os.makedirs(os.path.dirname(out_base_path), exist_ok=True)
    fig.tight_layout()

    fig.savefig(f"{out_base_path}.png", dpi=300, bbox_inches="tight")

    fig.savefig(f"{out_base_path}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_workload(summary_df: pd.DataFrame, metric: str, out_base: str,
                             ylabel: str, title: str, algorithms=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    algos = algorithms or [
        "KBFF", "GA", "PSO", "ACO", "DE", "Pure_FA", "KB_FPA", "Fuzzy_SLW", "UtilBand", "BFD"
    ]

    for algo in algos:
        sub = summary_df[summary_df["algorithm"] == algo].sort_values("n_tasks")
        if sub.empty:
            continue
        style = ALGO_STYLES.get(algo, {"color": None, "marker": "o", "lw": 1.2, "label": algo})
        mean_col = f"{metric}_mean" if f"{metric}_mean" in sub.columns else metric
        ci_col = f"{metric}_ci95" if f"{metric}_ci95" in sub.columns else None

        y_vals = sub[mean_col].values
        y_err = sub[ci_col].values if ci_col and ci_col in sub.columns else None
        x_vals = sub["n_tasks"].values

        if y_err is not None:
            ax.errorbar(x_vals, y_vals, yerr=y_err, marker=style.get("marker", "o"),
                        color=style.get("color"), linewidth=style.get("lw", 1.5),
                        capsize=4, capthick=1, label=style.get("label", algo))
        else:
            ax.plot(x_vals, y_vals, marker=style.get("marker", "o"),
                    color=style.get("color"), linewidth=style.get("lw", 1.5),
                    label=style.get("label", algo))

    ax.set_xlabel("Workload Scale (Number of Tasks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_xticks([1000, 3000, 5000])
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    _save_fig(fig, out_base)


def plot_pareto_frontier(summary_df: pd.DataFrame, out_base: str, n_tasks: int = 5000):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sub = summary_df[summary_df["n_tasks"] == n_tasks]
    if sub.empty:
        sub = summary_df

    eng_col = "total_energy_kwh_mean" if "total_energy_kwh_mean" in sub.columns else "total_energy_kwh"
    sla_col = "sla_violations_mean" if "sla_violations_mean" in sub.columns else "sla_violations"

    for _, row in sub.iterrows():
        algo = row["algorithm"]
        x = row[eng_col]
        y = row[sla_col]
        style = ALGO_STYLES.get(algo, {"color": "#555555", "marker": "o", "label": algo})
        ax.scatter(x, y, color=style["color"], marker=style["marker"], s=90, zorder=4,
                   label=style.get("label", algo))

        offset_x = 0.001
        offset_y = 1.0
        ax.annotate(algo, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Total Energy Consumption (kWh)")
    ax.set_ylabel("SLA Violations (Count)")
    ax.set_title(f"Multi-Objective Pareto Frontier at {n_tasks} Tasks", pad=10)
    ax.grid(True)
    _save_fig(fig, out_base)


def plot_ablation_comparison(ablation_df: pd.DataFrame, out_base: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))


    variants = ["KBFF", "KBFF_no_dtkb", "KBFF_no_fuzzy", "KBFF_no_adaptive_gate"]
    labels = ["Full KBFF", "w/o DTKB", "w/o Fuzzy", "w/o Adapt. Gate"]

    sub = ablation_df[ablation_df["algorithm"].isin(variants)]
    if sub.empty:
        return


    max_w = sub["n_tasks"].max()
    sub_w = sub[sub["n_tasks"] == max_w].set_index("algorithm").reindex(variants)

    sla_vals = sub_w.get("sla_violations_mean", sub_w.get("sla_violations", [0]*4)).values
    eng_vals = sub_w.get("total_energy_kwh_mean", sub_w.get("total_energy_kwh", [0]*4)).values

    x = np.arange(len(variants))
    colors = ["#D62728", "#8C564B", "#E377C2", "#7F7F7F"]


    bars1 = ax1.bar(x, sla_vals, color=colors, width=0.55, edgecolor="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("SLA Violations (Count)")
    ax1.set_title(f"Ablation: SLA Violations ({max_w} tasks)")
    ax1.grid(axis="y")
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2.0, yval + 1, f"{yval:.0f}", ha="center", va="bottom", fontsize=8.5)


    bars2 = ax2.bar(x, eng_vals, color=colors, width=0.55, edgecolor="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Total Energy (kWh)")
    ax2.set_title(f"Ablation: Energy Consumption ({max_w} tasks)")
    ax2.grid(axis="y")
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=8.5)

    _save_fig(fig, out_base)


def plot_dynamic_phases(df_dyn: pd.DataFrame, out_base: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    phases = ["P1 (Light)", "P2 (134% Burst)", "P3 (Mixed)"]
    algos_to_plot = ["KBFF", "KBFF_no_dtkb", "KBFF_no_adaptive_gate", "GA", "PSO", "BFD"]

    x = np.arange(len(phases))
    width = 0.13

    for i, algo in enumerate(algos_to_plot):
        sub = df_dyn[df_dyn["algorithm"] == algo]
        if sub.empty:
            continue
        row = sub.iloc[0]
        y_vals = [
            row.get("sla_P1", row.get("sla_violations_P1", 0)),
            row.get("sla_P2", row.get("sla_violations_P2", 0)),
            row.get("sla_P3", row.get("sla_violations_P3", 0)),
        ]
        style = ALGO_STYLES.get(algo, {"color": None, "label": algo})
        ax.bar(x + (i - len(algos_to_plot)/2) * width, y_vals, width,
               label=style.get("label", algo), color=style.get("color"), edgecolor="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("SLA Violations Count")
    ax.set_title("Dynamic Phase Shift: Resilience Under Non-Stationary Workload")
    ax.legend(frameon=True)
    ax.grid(axis="y")
    _save_fig(fig, out_base)


def generate_all_plots(summary_df: pd.DataFrame, raw_df: pd.DataFrame, out_dir: str = "figures",
                        df_dyn: pd.DataFrame = None, df_stress: pd.DataFrame = None):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating 300-DPI publication figures into '{out_dir}/'...")


    plot_metric_vs_workload(
        summary_df, "vm_utilisation_active_pct", os.path.join(out_dir, "fig06_vm_utilization"),
        ylabel="Active VM Resource Utilization (%)",
        title="Figure 6: VM Resource Utilization vs Workload Scale"
    )


    plot_metric_vs_workload(
        summary_df, "pm_utilisation_active_pct", os.path.join(out_dir, "fig07_pm_utilization"),
        ylabel="Active PM Resource Utilization (%)",
        title="Figure 7: PM Resource Utilization vs Workload Scale"
    )


    plot_metric_vs_workload(
        summary_df, "total_energy_kwh", os.path.join(out_dir, "fig08_energy_kwh"),
        ylabel="Total Energy Consumption (kWh)",
        title="Figure 8: Total Energy Consumption vs Workload Scale"
    )


    plot_metric_vs_workload(
        summary_df, "sla_violations", os.path.join(out_dir, "fig09_sla_violations"),
        ylabel="SLA Violations (Count)",
        title="Figure 9: SLA Violations vs Workload Scale"
    )


    plot_pareto_frontier(
        summary_df, os.path.join(out_dir, "fig10_pareto_energy_vs_sla")
    )


    plot_ablation_comparison(
        summary_df, os.path.join(out_dir, "fig12_ablation_comparison")
    )


    if df_dyn is not None and not df_dyn.empty:
        plot_dynamic_phases(df_dyn, os.path.join(out_dir, "fig13_dynamic_phases"))

    print(f"All publication figures successfully created in '{out_dir}/' (PNG + PDF).")
