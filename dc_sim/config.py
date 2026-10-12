from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class EnergyModelConfig:
    cpu_power_range: Tuple[float, float] = (73.0, 95.0)
    ram_power_range: Tuple[float, float] = (3.0, 4.5)
    storage_power_range: Tuple[float, float] = (0.6, 2.8)
    idle_fraction: float = 0.55
    sleep_fraction: float = 0.05
    idle_timeout_s: float = 60.0
    wake_energy_j: float = 500.0
    wake_latency_s: float = 5.0
    migration_energy_j: float = 200.0
    dvfs_enabled: bool = False
    dvfs_min_freq_scale: float = 0.6


@dataclass
class SchedulingConfig:


    policy: str = "space_shared"
    vm_num_pes_choices: Tuple[int, ...] = (2, 4)


@dataclass
class InfraConfig:
    n_pms: int = 50
    n_vms: int = 200
    ram_levels_gb: Tuple[float, ...] = (1024.0, 2048.0, 4096.0)
    storage_levels_gb: Tuple[float, ...] = (10000.0, 15000.0, 20000.0)
    vm_storage_levels_gb: Tuple[float, ...] = (250.0, 500.0, 1000.0)
    vm_ram_levels_gb: Tuple[float, ...] = (16.0, 32.0, 64.0)
    pm_mips_range: Tuple[float, float] = (9000.0, 16000.0)
    vm_mips: float = 367.0
    vm_bandwidth_gb: float = 1_000_000.0
    heterogeneous: bool = True
    provisioning_policy: str = "spread"


@dataclass
class WorkloadConfig:
    n_tasks: int = 1000

    task_length_lognorm_mu: float = 8.5
    task_length_lognorm_sigma: float = 0.8

    mean_interarrival_s: float = 0.085

    deadline_slack_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (1.5, 5.0),
        (1.2, 3.0),
        (1.01, 1.5)
    )
    priority_probs: Tuple[float, float, float] = (0.5, 0.35, 0.15)
    task_ram_gb_range: Tuple[float, float] = (0.5, 48.0)
    task_disk_gb_range: Tuple[float, float] = (1.0, 150.0)

    dynamic_phase_shift: bool = False


@dataclass
class PhaseSpec:
    name: str
    n_tasks: int
    mean_interarrival_s: float
    lognorm_mu: float
    lognorm_sigma: float
    ram_range: Tuple[float, float]
    disk_range: Tuple[float, float]
    priority_probs: Tuple[float, float, float]
    slack_range: Tuple[float, float]


@dataclass
class DynamicWorkloadConfig:
    phases: Tuple[PhaseSpec, ...] = field(default_factory=lambda: (
        PhaseSpec(
            name="P1",
            n_tasks=3000,
            mean_interarrival_s=0.10,
            lognorm_mu=8.5,
            lognorm_sigma=0.8,
            ram_range=(0.5, 8.0),
            disk_range=(1.0, 30.0),
            priority_probs=(0.7, 0.25, 0.05),
            slack_range=(2.0, 3.0),
        ),
        PhaseSpec(
            name="P2",
            n_tasks=4000,
            mean_interarrival_s=0.035,
            lognorm_mu=9.0,
            lognorm_sigma=0.7,
            ram_range=(24.0, 48.0),
            disk_range=(80.0, 150.0),
            priority_probs=(0.2, 0.4, 0.4),
            slack_range=(1.1, 1.6),
        ),
        PhaseSpec(
            name="P3",
            n_tasks=3000,
            mean_interarrival_s=0.06,
            lognorm_mu=8.5,
            lognorm_sigma=0.8,
            ram_range=(0.5, 48.0),
            disk_range=(1.0, 150.0),
            priority_probs=(0.5, 0.35, 0.15),
            slack_range=(1.2, 3.0),
        ),
    ))
    dtkb_train_on_phase0_only: bool = True


@dataclass
class FireflyConfig:
    beta0: float = 0.6
    alpha: float = 0.4
    gamma: float = 1.0
    max_generations: int = 10
    population_scale: float = 1.0
    max_population: int = 15
    plateau_patience: int = 3
    plateau_eps: float = 1e-4
    w_energy: float = 0.55
    w_sla: float = 0.20
    w_migration: float = 0.10
    w_balance: float = 0.05
    w_priority: float = 0.10


@dataclass
class DTKBConfig:
    branching_factor: int = 3
    max_depth: int = 2
    n_historical_records: int = 20000
    criterion: str = "gini"
    ccp_alpha: float = 0.0
    retrain_every_n_tasks: int = 500
    test_size: float = 0.2


@dataclass
class FuzzyConfig:
    dims: Tuple[str, ...] = ("ram", "storage", "cpu", "bandwidth",
                              "migration_cost", "deadline_slack", "priority")
    dim_weights: Tuple[float, ...] = (0.22, 0.18, 0.22, 0.08, 0.10, 0.12, 0.08)
    n_linguistic_terms: int = 3
    n_rules: int = 3 ** 3
    gate_tau: float = 0.35


@dataclass
class ExperimentConfig:
    seeds: Tuple[int, ...] = tuple(range(1, 31))
    workload_sizes: Tuple[int, ...] = (1000, 3000, 5000)
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    algorithms: Tuple[str, ...] = (
        "KBFF",
        "KBFF_no_fuzzy",
        "KBFF_no_dtkb",
        "KBFF_no_energy_fitness",
        "KB_FPA",
        "Fuzzy_SLW",
        "BFD",
        "MBF",
        "UtilBand",
        "PSO",
        "ACO",
        "GA",
        "DE",
        "Pure_FA",
        "Fuzzy_Only",
        "KB_Only",
    )


@dataclass
class FullConfig:
    infra: InfraConfig = field(default_factory=InfraConfig)
    energy: EnergyModelConfig = field(default_factory=EnergyModelConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    firefly: FireflyConfig = field(default_factory=FireflyConfig)
    dtkb: DTKBConfig = field(default_factory=DTKBConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    dynamic_workload: DynamicWorkloadConfig = field(default_factory=DynamicWorkloadConfig)
