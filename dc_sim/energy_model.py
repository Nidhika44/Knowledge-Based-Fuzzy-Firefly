from dataclasses import dataclass
from .entities import PhysicalMachine
from .config import EnergyModelConfig


def peak_power(pm: PhysicalMachine) -> float:
    return pm.cpu_power_max + pm.ram_power_max + pm.storage_power_max

def pm_instantaneous_power_watts(pm: PhysicalMachine, cfg: EnergyModelConfig) -> float:
    u_cpu = pm.util_mips()
    u_ram = pm.util_ram()
    u_disk = pm.util_storage()

    freq_scale = 1.0
    if cfg.dvfs_enabled:

        freq_scale = max(cfg.dvfs_min_freq_scale, u_cpu)

    cpu_idle = cfg.idle_fraction * pm.cpu_power_max
    cpu_dynamic = (pm.cpu_power_max - cpu_idle) * u_cpu * freq_scale
    cpu_power = cpu_idle * (freq_scale if cfg.dvfs_enabled else 1.0) + cpu_dynamic

    ram_idle = cfg.idle_fraction * pm.ram_power_max
    ram_power = ram_idle + (pm.ram_power_max - ram_idle) * u_ram

    disk_idle = cfg.idle_fraction * pm.storage_power_max
    disk_power = disk_idle + (pm.storage_power_max - disk_idle) * u_disk

    return cpu_power + ram_power + disk_power


def datacenter_power_watts(pms, cfg: EnergyModelConfig) -> float:
    return sum(pm_instantaneous_power_watts(p, cfg) for p in pms)


def energy_kwh(power_watts: float, duration_seconds: float) -> float:
    return power_watts * duration_seconds / 3600.0 / 1000.0


def energy_joules(power_watts: float, duration_seconds: float) -> float:
    return power_watts * duration_seconds
