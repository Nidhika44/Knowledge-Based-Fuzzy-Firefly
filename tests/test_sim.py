import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
from dc_sim.config import FullConfig, EnergyModelConfig
from dc_sim.entities import PhysicalMachine, VirtualMachine, Task
from dc_sim.energy_model import pm_instantaneous_power_watts, peak_power
from dc_sim.workload import generate_infrastructure, generate_workload, make_rng_streams
from dc_sim.simulator import DiscreteEventSimulator, provision_vms_on_pms, SimResult
from dc_sim.experiment import Experiment, build_algorithm, _run_one_unit


def make_minimal_cfg():
    cfg = FullConfig()
    cfg.infra.n_pms = 1
    cfg.infra.n_vms = 1
    cfg.workload.n_tasks = 1
    cfg.infra.ram_levels_gb = (1024.0,)
    cfg.infra.storage_levels_gb = (10000.0,)
    cfg.infra.pm_mips_range = (1000.0, 1000.0)
    cfg.infra.vm_mips = 367.0
    cfg.infra.vm_ram_levels_gb = (16.0,)
    cfg.infra.vm_storage_levels_gb = (250.0,)
    cfg.scheduling.vm_num_pes_choices = (4,)
    cfg.experiment.seeds = (1,)
    cfg.experiment.workload_sizes = (1,)
    cfg.experiment.algorithms = ("BFD",)
    return cfg


def test_single_pm_energy_constant_utilization():
    cfg_e = EnergyModelConfig(
        cpu_power_range=(100.0, 100.0),
        ram_power_range=(10.0, 10.0),
        storage_power_range=(5.0, 5.0),
        idle_fraction=0.5,
    )
    pm = PhysicalMachine(
        id=0, ram_gb=1024.0, storage_gb=10000.0, mips=1000.0,
        cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0,
        idle_fraction=0.5,
    )
    pm.state = "ACTIVE"

    pm.used_mips = 500.0
    pm.reserved_mips = 1000.0

    T = 60.0
    power = pm_instantaneous_power_watts(pm, cfg_e)
    expected_energy = power * T


    assert abs(power - 82.5) < 0.01, f"Expected 82.5 W, got {power:.4f} W"
    assert abs(expected_energy - 82.5 * T) < 0.01
    print(f"[TEST 1 PASS] Single PM power={power:.2f}W, energy over {T}s = {expected_energy:.2f}J")


def test_time_shared_two_tasks_double_time():
    from dc_sim.scheduling import make_scheduler
    import heapq

    cfg = FullConfig()
    cfg.scheduling.policy = "time_shared"

    vm = VirtualMachine(id=0, ram_gb=64.0, storage_gb=1000.0, mips=367.0,
                         bandwidth_gb=1e6, num_pes=4)
    pm = PhysicalMachine(id=0, ram_gb=1024.0, storage_gb=10000.0, mips=4000.0,
                          cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0,
                          idle_fraction=0.5)
    pm.place(vm)

    sched = make_scheduler("time_shared")
    event_heap = []

    task1 = Task(id=0, arrival_time=0.0, length_mi=367.0, ram_req_gb=1.0, disk_req_gb=1.0,
                 priority=1, deadline=1e9, exec_seconds_hint=1.0)
    task2 = Task(id=1, arrival_time=0.0, length_mi=367.0, ram_req_gb=1.0, disk_req_gb=1.0,
                 priority=1, deadline=1e9, exec_seconds_hint=1.0)


    sched_single = make_scheduler("time_shared")
    vm_s = VirtualMachine(id=1, ram_gb=64.0, storage_gb=1000.0, mips=367.0,
                           bandwidth_gb=1e6, num_pes=4)
    pm_s = PhysicalMachine(id=1, ram_gb=1024.0, storage_gb=10000.0, mips=4000.0,
                            cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0,
                            idle_fraction=0.5)
    pm_s.place(vm_s)
    heap_s = []
    task_s = Task(id=2, arrival_time=0.0, length_mi=367.0, ram_req_gb=1.0, disk_req_gb=1.0,
                   priority=1, deadline=1e9, exec_seconds_hint=1.0)
    sched_single.submit(vm_s, pm_s, task_s, 0.0, heap_s)
    single_finish = heap_s[0][0] if heap_s else None
    assert single_finish is not None, "Single task should produce a finish event"


    sched.submit(vm, pm, task1, 0.0, event_heap)
    sched.submit(vm, pm, task2, 0.0, event_heap)


    task_map = {0: task1, 1: task2}
    while event_heap:
        t, vm_id, task_id = heapq.heappop(event_heap)
        genuine = sched.notify_finish(vm, pm, task_id, t, event_heap)
        if genuine and task_id in task_map:
            task_map[task_id].finish_time = t

    finish_times = [t.finish_time for t in [task1, task2] if t.finish_time is not None]
    assert len(finish_times) == 2, f"Both tasks should finish, got {finish_times}"
    max_finish = max(finish_times)
    expected = single_finish * 2.0


    assert abs(max_finish - expected) < expected * 0.25, (
        f"Expected ~{expected:.2f}s for two co-resident tasks, got {max_finish:.2f}s"
    )
    print(f"[TEST 2 PASS] Single={single_finish:.2f}s, Two-task max_finish={max_finish:.2f}s"
          f" (expected ~{expected:.2f}s)")


def test_energy_accounting_sum():
    cfg = make_minimal_cfg()
    cfg.energy.wake_energy_j = 500.0
    cfg.energy.migration_energy_j = 200.0

    rng_infra, rng_work, rng_algo = make_rng_streams(42)
    pms, vms = generate_infrastructure(cfg.infra, cfg.energy, rng_infra, cfg.scheduling)
    provision_vms_on_pms(pms, vms, rng_infra, policy="bin_packing")
    tasks = generate_workload(cfg.workload, 1, rng_work)

    algo = build_algorithm("BFD", cfg, rng_algo)
    hosted_vms = [v for v in vms if v.host_pm is not None]
    if hasattr(algo, "prepare"):
        algo.prepare(hosted_vms)

    sim = DiscreteEventSimulator(cfg, pms, vms, tasks, algo, "BFD", 42)
    result = sim.run()


    total_reported = result.total_energy_joules
    migration_energy = result.migration_energy_j
    assert total_reported >= 0, "Total energy should be non-negative"
    assert migration_energy >= 0, "Migration energy should be non-negative"

    assert migration_energy <= total_reported + 1.0, (
        f"Migration energy ({migration_energy:.2f}J) exceeds total ({total_reported:.2f}J)"
    )
    print(f"[TEST 3 PASS] total={total_reported:.2f}J, migration_energy={migration_energy:.2f}J")


def test_no_pm_capacity_violation():
    cfg = FullConfig()
    cfg.infra.n_pms = 50
    cfg.infra.n_vms = 200
    rng_infra, rng_work, rng_algo = make_rng_streams(7)
    pms, vms = generate_infrastructure(cfg.infra, cfg.energy, rng_infra, cfg.scheduling)
    provision_vms_on_pms(pms, vms, rng_infra, policy="bin_packing")

    violations = []
    for pm in pms:
        if pm.used_ram > pm.ram_gb + 1e-6:
            violations.append(f"PM {pm.id}: used_ram={pm.used_ram:.2f} > capacity={pm.ram_gb:.2f}")
        if pm.used_storage > pm.storage_gb + 1e-6:
            violations.append(f"PM {pm.id}: used_storage={pm.used_storage:.2f} > capacity={pm.storage_gb:.2f}")
        if pm.reserved_mips > pm.mips * 1.01:
            violations.append(f"PM {pm.id}: reserved_mips={pm.reserved_mips:.2f} > capacity={pm.mips:.2f}")

    assert len(violations) == 0, f"Capacity violations found:\n" + "\n".join(violations)
    print(f"[TEST 4 PASS] No capacity violations across {len(pms)} PMs, {len(vms)} VMs")


def test_reproducibility():
    cfg = make_minimal_cfg()
    cfg.infra.n_pms = 5
    cfg.infra.n_vms = 10
    cfg.workload.n_tasks = 5
    cfg.experiment.seeds = (42,)
    cfg.experiment.workload_sizes = (5,)
    cfg.experiment.algorithms = ("BFD",)

    r1 = _run_one_unit("BFD", 5, 42, cfg, dynamic=False)
    r2 = _run_one_unit("BFD", 5, 42, cfg, dynamic=False)

    keys_to_check = [
        "total_energy_joules", "sla_violations", "makespan_s",
        "tasks_allocated", "tasks_blocked", "migrations", "used_vm_count"
    ]
    mismatches = []
    for k in keys_to_check:
        v1, v2 = r1.get(k), r2.get(k)
        if v1 != v2:
            mismatches.append(f"  {k}: {v1} != {v2}")

    assert len(mismatches) == 0, (
        f"Reproducibility violated! Different results for same seed:\n" + "\n".join(mismatches)
    )
    print(f"[TEST 5 PASS] Identical results for seed=42:")
    for k in keys_to_check:
        print(f"  {k} = {r1.get(k)}")


def test_energy_unit_consistency():
    cfg = make_minimal_cfg()
    cfg.infra.n_pms = 5
    cfg.infra.n_vms = 10

    r = _run_one_unit("BFD", 5, 1, cfg, dynamic=False)

    total_j = r.get("total_energy_joules", 0.0)
    total_kwh = r.get("total_energy_kwh", 0.0)
    mean_kw = r.get("mean_power_kw", 0.0)
    makespan = r.get("makespan_s", 1.0)

    expected_kwh = total_j / 3_600_000.0
    assert abs(total_kwh - expected_kwh) < 1e-9, (
        f"Energy kWh mismatch: reported={total_kwh}, expected={expected_kwh}"
    )

    if makespan > 0:
        makespan_h = makespan / 3600.0
        expected_kw = total_kwh / makespan_h
        assert abs(mean_kw - expected_kw) < 1e-9, (
            f"Mean power kW mismatch: reported={mean_kw}, expected={expected_kw}"
        )
    print(f"[TEST 6 PASS] Energy units consistent: {total_kwh:.6f} kWh, {mean_kw:.6f} kW")


if __name__ == "__main__":
    test_single_pm_energy_constant_utilization()
    test_time_shared_two_tasks_double_time()
    test_energy_accounting_sum()
    test_no_pm_capacity_violation()
    test_reproducibility()
    test_energy_unit_consistency()
    print("\nAll tests passed!")
