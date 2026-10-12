import numpy as np
from typing import List, Tuple, Optional
from .entities import PhysicalMachine, VirtualMachine, Task
from .config import InfraConfig, WorkloadConfig, EnergyModelConfig, DynamicWorkloadConfig


def make_rng_streams(seed: int) -> Tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    ss = np.random.SeedSequence(seed)
    rng_infra, rng_work, rng_algo = [np.random.default_rng(s) for s in ss.spawn(3)]
    return rng_infra, rng_work, rng_algo


def generate_infrastructure(infra: InfraConfig, energy: EnergyModelConfig,
                             rng: np.random.Generator,
                             scheduling: "SchedulingConfig" = None
                             ) -> Tuple[List[PhysicalMachine], List[VirtualMachine]]:
    pms = []
    for i in range(infra.n_pms):
        ram = rng.choice(infra.ram_levels_gb)
        storage = rng.choice(infra.storage_levels_gb)
        mips = rng.uniform(*infra.pm_mips_range)
        cpu_p = rng.uniform(*energy.cpu_power_range)
        ram_p = rng.uniform(*energy.ram_power_range)
        disk_p = rng.uniform(*energy.storage_power_range)
        pms.append(PhysicalMachine(
            id=i, ram_gb=ram, storage_gb=storage, mips=mips,
            cpu_power_max=cpu_p, ram_power_max=ram_p, storage_power_max=disk_p,
            idle_fraction=energy.idle_fraction,
        ))

    vms = []
    for i in range(infra.n_vms):
        ram = rng.choice(infra.vm_ram_levels_gb)
        storage = rng.choice(infra.vm_storage_levels_gb)
        num_pes = int(rng.choice(scheduling.vm_num_pes_choices)) if scheduling is not None else 2


        bw = float(rng.uniform(0.5, 1.5)) * infra.vm_bandwidth_gb

        vms.append(VirtualMachine(
            id=i, ram_gb=ram, storage_gb=storage, mips=infra.vm_mips,
            bandwidth_gb=bw, num_pes=num_pes,
        ))
    return pms, vms


def generate_workload(wl: WorkloadConfig, n_tasks: int,
                       rng: np.random.Generator) -> List[Task]:
    tasks = []
    t = 0.0
    priorities = np.array([0, 1, 2])
    for i in range(n_tasks):
        interarrival = rng.exponential(wl.mean_interarrival_s)
        t += interarrival
        priority = int(rng.choice(priorities, p=wl.priority_probs))
        length_mi = float(rng.lognormal(wl.task_length_lognorm_mu, wl.task_length_lognorm_sigma))
        ram_req = float(rng.uniform(*wl.task_ram_gb_range))
        disk_req = float(rng.uniform(*wl.task_disk_gb_range))


        slack_ranges = getattr(wl, 'deadline_slack_ranges', None)
        if slack_ranges is not None:
            slack_range = slack_ranges[priority]
        else:
            slack_range = (1.2, 3.0)
        slack = rng.uniform(*slack_range)

        nominal_mips = 367.0
        nominal_exec = length_mi / nominal_mips
        deadline = t + slack * nominal_exec
        tasks.append(Task(
            id=i, arrival_time=t, length_mi=length_mi, ram_req_gb=ram_req,
            disk_req_gb=disk_req, priority=priority, deadline=deadline,
            exec_seconds_hint=nominal_exec,
            phase="default",
        ))
    return tasks


def generate_dynamic_workload(dyn_cfg: DynamicWorkloadConfig,
                               rng: np.random.Generator) -> List[Task]:
    tasks = []
    t = 0.0
    task_id = 0
    priorities = np.array([0, 1, 2])

    for ph in dyn_cfg.phases:
        for _ in range(ph.n_tasks):
            interarrival = rng.exponential(ph.mean_interarrival_s)
            t += interarrival
            priority = int(rng.choice(priorities, p=ph.priority_probs))
            length_mi = float(rng.lognormal(ph.lognorm_mu, ph.lognorm_sigma))
            ram_req = float(rng.uniform(*ph.ram_range))
            disk_req = float(rng.uniform(*ph.disk_range))
            slack = rng.uniform(*ph.slack_range)

            nominal_mips = 367.0
            nominal_exec = length_mi / nominal_mips
            deadline = t + slack * nominal_exec
            tasks.append(Task(
                id=task_id, arrival_time=t, length_mi=length_mi,
                ram_req_gb=ram_req, disk_req_gb=disk_req,
                priority=priority, deadline=deadline,
                exec_seconds_hint=nominal_exec,
                phase=ph.name,
            ))
            task_id += 1

    return tasks
