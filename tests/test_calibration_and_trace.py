import sys
import os
import copy
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dc_sim.config import FullConfig, InfraConfig, WorkloadConfig, EnergyModelConfig
from dc_sim.entities import PhysicalMachine, VirtualMachine, Task
from dc_sim.workload import generate_infrastructure, generate_workload, make_rng_streams
from dc_sim.simulator import DiscreteEventSimulator, provision_vms_on_pms
from dc_sim.kbff import KBFuzzyFirefly


def test_hand_trace_state_machine():
    print("\n--- TEST: Hand-Trace State Machine (Small 2-PM, 4-VM Cluster) ---")
    cfg = FullConfig()
    cfg.energy.wake_latency_s = 4.0
    cfg.energy.wake_energy_j = 200.0
    cfg.energy.idle_timeout_s = 10.0
    cfg.scheduling.policy = "space_shared"

    pm0 = PhysicalMachine(id=0, ram_gb=1024.0, storage_gb=10000.0, mips=4000.0,
                          cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0, idle_fraction=0.5)
    pm0.state = "SLEEP"

    vm0 = VirtualMachine(id=0, ram_gb=32.0, storage_gb=500.0, mips=367.0, bandwidth_gb=1e6, num_pes=1)
    pm0.place(vm0)


    t1 = Task(id=1, arrival_time=1.0, length_mi=367.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=1, deadline=20.0)

    t2 = Task(id=2, arrival_time=2.0, length_mi=367.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=1, deadline=20.0)

    class StaticAlloc:
        def allocate(self, t, vb, pb, ids):
            return 0

    sim = DiscreteEventSimulator(cfg, [pm0], [vm0], [t1, t2], StaticAlloc(), "Static", 1)
    res = sim.run()

    print(f"Task 1: arrival={t1.arrival_time:.1f}s, start={t1.start_time:.1f}s, finish={t1.finish_time:.1f}s")
    print(f"Task 2: arrival={t2.arrival_time:.1f}s, start={t2.start_time:.1f}s, finish={t2.finish_time:.1f}s")
    print(f"Total energy: {res.total_energy_joules:.2f} J, Makespan: {res.makespan_s:.2f} s")


    assert t1.start_time == 5.0, f"Expected T1 start at 5.0s, got {t1.start_time}"

    assert t1.finish_time == 6.0, f"Expected T1 finish at 6.0s, got {t1.finish_time}"


    assert t2.start_time == 6.0, f"Expected T2 start at 6.0s, got {t2.start_time}"

    assert t2.finish_time == 7.0, f"Expected T2 finish at 7.0s, got {t2.finish_time}"
    print("[PASS] Hand-trace state machine verified: WAKING delay + SpaceShared core queueing exactly as expected!")


def test_load_calibration():
    print("\n--- TEST: Load Calibration & Real Queueing ---")
    cfg = FullConfig()
    cfg.infra.n_pms = 50
    cfg.infra.n_vms = 200
    cfg.scheduling.policy = "space_shared"
    cfg.scheduling.vm_num_pes_choices = (2, 4)


    cfg.workload.mean_interarrival_s = 0.037
    cfg.workload.task_length_lognorm_mu = 8.2
    cfg.workload.n_tasks = 1000

    rng_infra, rng_work, rng_algo = make_rng_streams(42)
    pms, vms = generate_infrastructure(cfg.infra, cfg.energy, rng_infra, cfg.scheduling)
    provision_vms_on_pms(pms, vms, rng_infra, policy="spread")
    tasks = generate_workload(cfg.workload, 1000, rng_work)

    total_cores = sum(v.num_pes for v in vms)
    print(f"Provisioned Cluster: {len(pms)} PMs, {len(vms)} VMs, Total Cores: {total_cores}")

    algo = KBFuzzyFirefly(cfg, rng_algo)
    algo.fit_knowledge_base([v for v in vms if v.host_pm is not None], pms, cfg, dynamic=False, seed=42)

    sim = DiscreteEventSimulator(cfg, pms, vms, tasks, algo, "KBFF", 42)
    res = sim.run()

    print(f"\nSimulation Results for N=1000 tasks:")
    print(f"  Allocated: {res.tasks_allocated}, Blocked: {res.tasks_blocked}")
    print(f"  SLA Violations: {res.sla_violations} (Low:{res.sla_violations_low}, Med:{res.sla_violations_med}, High:{res.sla_violations_high})")
    print(f"  Active PMs: {res.active_pm_count} / {len(pms)}")
    print(f"  Used VMs:   {res.used_vm_count} / {len(vms)}")
    print(f"  VM Util (Active-period): {res.vm_utilisation_active_pct:.2f}%")
    print(f"  PM Util (Active-period): {res.pm_utilisation_active_pct:.2f}%")
    print(f"  VM Util (Fleet-wide):    {res.vm_utilisation_fleet_pct:.2f}%")
    print(f"  PM Util (Fleet-wide):    {res.pm_utilisation_fleet_pct:.2f}%")
    print(f"  Queued Tasks Count:      {res.queued_task_count}")
    print(f"  Avg Queue Wait Time:     {res.avg_queue_wait_s:.3f} s")
    print(f"  Energy:                  {res.total_energy_kwh:.3f} kWh ({res.total_energy_joules:.1f} J)")


    assert res.tasks_allocated + res.tasks_blocked == 1000, "Task conservation violated!"
    assert res.queued_task_count > 0, "No queueing occurred! Workload is still too sparse."
    assert res.avg_queue_wait_s > 0.0, "Average queue wait time is zero!"
    assert 10.0 <= res.vm_utilisation_fleet_pct <= 95.0, f"Fleet util out of target range: {res.vm_utilisation_fleet_pct}%"
    assert res.total_energy_joules > 0.0, "Energy <= 0"
    print("\n[PASS] Load calibration verified: genuine queueing confirmed and metrics in healthy physical range!")


if __name__ == "__main__":
    test_hand_trace_state_machine()
    test_load_calibration()
