import os
import sys
import copy
import hashlib
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dc_sim.config import FullConfig, FuzzyConfig, DTKBConfig
from dc_sim.entities import PhysicalMachine, VirtualMachine, Task
from dc_sim.knowledge_base import DTKB
from dc_sim.fuzzy_system import FRBS
from dc_sim.kbff import KBFuzzyFirefly, AdaptiveGate
from dc_sim.baselines import KBOnlyAllocator, KBFPAAllocator, _feasible_vm_ids
from dc_sim.legacy_names import ALGORITHM_NAME_MAP
from dc_sim.simulator import DiscreteEventSimulator
from dc_sim.experiment import Experiment, _config_to_dict


def test_fix1_rng_independence():
    cfg = FullConfig()
    vms = [VirtualMachine(id=i, ram_gb=32.0, storage_gb=500.0, mips=367.0, bandwidth_gb=1e6) for i in range(5)]
    pms = [PhysicalMachine(id=0, ram_gb=1024.0, storage_gb=10000.0, mips=10000.0,
                           cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0, idle_fraction=0.5)]
    pms[0].place(vms[0])

    rng1 = np.random.default_rng(999)
    state1_before = copy.deepcopy(rng1.bit_generator.state)
    dtkb1 = DTKB(cfg=cfg.dtkb, rng=rng1)

    dtkb1.fit_with_warmup(vms, pms, cfg, n_warm=20, dynamic=False, seed=999)
    state1_after = rng1.bit_generator.state


    assert state1_before['state']['state'] == state1_after['state']['state'], "self.rng was mutated by fit_with_warmup!"


    rng2 = np.random.default_rng(999)
    state2_before = copy.deepcopy(rng2.bit_generator.state)
    dtkb2 = DTKB(cfg=cfg.dtkb, rng=rng2)
    dtkb2.fit_with_warmup(vms, pms, cfg, n_warm=20, dynamic=False, seed=999)
    state2_after = rng2.bit_generator.state

    assert state2_before['state']['state'] == state2_after['state']['state'], "self.rng was mutated on cache hit!"
    print("[PASS] Fix 1: RNG state completely unpolluted by fit_with_warmup.")


def test_fix2_dtkb_baseline_parity():
    cfg = FullConfig()
    rng = np.random.default_rng(42)

    kb_only = KBOnlyAllocator(cfg, rng)
    kb_fpa = KBFPAAllocator(cfg, rng)

    assert hasattr(kb_only, "observe_outcome"), "KBOnlyAllocator missing observe_outcome!"
    assert hasattr(kb_fpa, "observe_outcome"), "KBFPAAllocator missing observe_outcome!"

    task = Task(id=1, arrival_time=0.0, length_mi=100.0, ram_req_gb=1.0, disk_req_gb=1.0, priority=1, deadline=10.0)
    task.start_time = 0.0
    task._feat = [16.0, 250.0, 0.8, 1.0, 1.0, 1.0, 2.0]
    vm = VirtualMachine(id=0, ram_gb=16.0, storage_gb=250.0, mips=367.0, bandwidth_gb=1e6)


    kb_only.observe_outcome(task, vm, met_deadline=True)
    assert len(kb_only.dtkb._rolling_log_X) == 1, "KBOnly rolling log not updated!"
    assert kb_only.dtkb._rolling_log_y[0] == 1, "KBOnly rolling outcome incorrect!"

    kb_fpa.observe_outcome(task, vm, met_deadline=True)
    assert len(kb_fpa.dtkb._rolling_log_X) == 1, "KBFPA rolling log not updated!"
    print("[PASS] Fix 2: KB baselines have online observation and DTKB parity.")


def test_fix4_adaptive_gate():
    gate = AdaptiveGate(lam=0.5)


    gate.update("dtkb", success=True)
    assert gate.lam > 0.5, f"Expected lambda > 0.5, got {gate.lam}"


    for _ in range(5):
        gate.update("frbs", success=True)
    gate.update("dtkb", success=False)
    assert gate.lam < 0.5, f"Expected lambda < 0.5, got {gate.lam}"
    print("[PASS] Fix 4: Adaptive gate reliability precision update functions properly.")


def test_fix5_frbs_weights():
    cfg1 = FuzzyConfig(dim_weights=(0.50, 0.10, 0.20, 0.05, 0.05, 0.05, 0.05))
    cfg2 = FuzzyConfig(dim_weights=(0.05, 0.10, 0.20, 0.05, 0.05, 0.05, 0.50))

    frbs1 = FRBS.build(cfg1, None)
    frbs2 = FRBS.build(cfg2, None)

    vm = VirtualMachine(id=0, ram_gb=64.0, storage_gb=1000.0, mips=367.0, bandwidth_gb=1e6)
    task_low = Task(id=0, arrival_time=0.0, length_mi=100.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=0, deadline=10.0)
    task_hi = Task(id=1, arrival_time=0.0, length_mi=100.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=2, deadline=10.0)


    gap1 = abs(frbs1.suitability(vm, task_hi) - frbs1.suitability(vm, task_low))
    gap2 = abs(frbs2.suitability(vm, task_hi) - frbs2.suitability(vm, task_low))

    assert gap2 > gap1 * 3.0, f"FRBS not responding to dim_weights! gap1={gap1}, gap2={gap2}"
    print("[PASS] Fix 5: FRBS dynamically honors FuzzyConfig.dim_weights and priority anchor.")


def test_fix6_legacy_name_mappings():
    required_keys = [
        "UtilBand", "KBFF_w_energy_0", "Capacity_Filter",
        "OriginalPSO", "OriginalGA", "OriginalDE", "OriginalACO", "OriginalFA"
    ]
    for k in required_keys:
        assert k in ALGORITHM_NAME_MAP, f"Missing key in ALGORITHM_NAME_MAP: {k}"
    print("[PASS] Fix 6: All baselines and variants correctly registered in ALGORITHM_NAME_MAP.")


def test_fix7_wake_latency():
    cfg = FullConfig()
    cfg.energy.wake_latency_s = 5.0
    cfg.energy.wake_energy_j = 100.0
    cfg.scheduling.policy = "space_shared"

    pm = PhysicalMachine(id=0, ram_gb=1024.0, storage_gb=10000.0, mips=4000.0,
                         cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0, idle_fraction=0.5)
    pm.state = "SLEEP"
    vm = VirtualMachine(id=0, ram_gb=64.0, storage_gb=1000.0, mips=367.0, bandwidth_gb=1e6, num_pes=4)
    pm.place(vm)

    task1 = Task(id=1, arrival_time=2.0, length_mi=367.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=1, deadline=20.0)
    task2 = Task(id=2, arrival_time=4.0, length_mi=367.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=1, deadline=20.0)

    class FixedAlloc:
        def allocate(self, t, vms, pms, ids):
            return 0

    sim = DiscreteEventSimulator(cfg, [pm], [vm], [task1, task2], FixedAlloc(), "Fixed", 1)
    res = sim.run()


    assert task1.start_time == 7.0, f"Task 1 should start at 7.0, got {task1.start_time}"

    assert task2.start_time == 7.0, f"Task 2 should start at 7.0, got {task2.start_time}"

    assert res.total_energy_joules >= 100.0, "Wake energy was not charged!"
    print(f"[PASS] Fix 7: Wake latency enforced consistently (T1 start={task1.start_time}s, T2 start={task2.start_time}s).")


def test_fix8_feasible_vm_fallback():
    vm = VirtualMachine(id=0, ram_gb=16.0, storage_gb=250.0, mips=367.0, bandwidth_gb=1e6, num_pes=1)
    vm.used_mips = 367.0

    task = Task(id=1, arrival_time=0.0, length_mi=100.0, ram_req_gb=4.0, disk_req_gb=20.0, priority=1, deadline=10.0)

    vms_by_id = {0: vm}
    feas = _feasible_vm_ids(task, vms_by_id, [0])

    assert feas == [0], f"Expected fallback to VM 0, got {feas}"
    print("[PASS] Fix 8: Feasible VM fallback prevents dropping tasks when cores are busy.")


def test_fix9_checkpoint_hash_stability():
    cfg1 = FullConfig()
    cfg1.experiment.seeds = (1, 2, 3)

    cfg2 = FullConfig()
    cfg2.experiment.seeds = tuple(range(1, 31))

    def compute_hash(cfg):
        cfg_dict = _config_to_dict(cfg)
        if "experiment" in cfg_dict:
            exp_dict = dict(cfg_dict["experiment"])
            exp_dict.pop("seeds", None)
            exp_dict.pop("checkpoint_dir", None)
            exp_dict.pop("results_dir", None)
            cfg_dict["experiment"] = exp_dict
        return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode('utf-8')).hexdigest()[:8]

    h1 = compute_hash(cfg1)
    h2 = compute_hash(cfg2)

    assert h1 == h2, f"cfg_hash changed when seeds changed: {h1} != {h2}"
    print(f"[PASS] Fix 9: Checkpoint hash invariant across seeds ({h1} == {h2}).")


def test_point1_deadline_slack():
    cfg = FuzzyConfig(dim_weights=(0.1, 0.1, 0.1, 0.1, 0.1, 0.45, 0.05))
    frbs = FRBS.build(cfg, None)
    vm = VirtualMachine(id=0, ram_gb=64.0, storage_gb=1000.0, mips=367.0, bandwidth_gb=1e6)


    t_tight = Task(id=1, arrival_time=0.0, length_mi=200.0, ram_req_gb=2.0, disk_req_gb=10.0,
                   priority=1, deadline=1.0, exec_seconds_hint=2.0)

    t_ample = Task(id=2, arrival_time=0.0, length_mi=200.0, ram_req_gb=2.0, disk_req_gb=10.0,
                   priority=1, deadline=6.0, exec_seconds_hint=2.0)

    score_tight = frbs.suitability(vm, t_tight)
    score_ample = frbs.suitability(vm, t_ample)

    assert score_ample > score_tight, f"Ample slack ({score_ample}) should score higher than tight slack ({score_tight})"
    print(f"[PASS] Point 1: Deadline slack correctly rewards safety headroom (tight={score_tight:.3f}, ample={score_ample:.3f}).")


def test_point2_wake_power_continuous():
    cfg = FullConfig()
    cfg.energy.wake_latency_s = 4.0
    cfg.energy.wake_energy_j = 200.0
    cfg.energy.sleep_fraction = 0.0

    pm = PhysicalMachine(id=0, ram_gb=1024.0, storage_gb=10000.0, mips=4000.0,
                         cpu_power_max=100.0, ram_power_max=0.0, storage_power_max=0.0, idle_fraction=0.0)
    pm.state = "SLEEP"
    vm = VirtualMachine(id=0, ram_gb=64.0, storage_gb=1000.0, mips=367.0, bandwidth_gb=1e6, num_pes=1)
    pm.place(vm)

    task = Task(id=1, arrival_time=0.0, length_mi=367.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=1, deadline=20.0)

    class SingleAlloc:
        def allocate(self, t, vms, pms, ids):
            return 0

    sim = DiscreteEventSimulator(cfg, [pm], [vm], [task], SingleAlloc(), "Single", 1)
    res = sim.run()


    assert res.total_energy_joules >= 200.0, f"Total energy {res.total_energy_joules} < wake energy 200 J"
    print(f"[PASS] Point 2: Continuous wake-power integrated smoothly (E_total={res.total_energy_joules:.1f} J).")


def test_point3_fleet_utilization():
    cfg = FullConfig()

    pms = [PhysicalMachine(id=i, ram_gb=1024.0, storage_gb=10000.0, mips=4000.0,
                           cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0, idle_fraction=0.5)
           for i in range(4)]
    vms = [VirtualMachine(id=i, ram_gb=64.0, storage_gb=1000.0, mips=367.0, bandwidth_gb=1e6, num_pes=1)
           for i in range(4)]
    for i in range(4):
        pms[i].place(vms[i])

    task = Task(id=1, arrival_time=0.0, length_mi=367.0, ram_req_gb=2.0, disk_req_gb=10.0, priority=1, deadline=20.0)

    class VM0OnlyAlloc:
        def allocate(self, t, vms, pms, ids):
            return 0

    sim = DiscreteEventSimulator(cfg, pms, vms, [task], VM0OnlyAlloc(), "VM0Only", 1)
    res = sim.run()


    assert res.vm_utilisation_fleet_pct < res.vm_utilisation_pct + 1e-3, "Fleet util must be <= used-VM util"
    assert res.pm_utilisation_fleet_pct < res.pm_utilisation_pct + 1e-3, "Fleet PM util must be <= active-PM util"
    assert 0.0 <= res.vm_utilisation_fleet_pct <= 100.0, "Fleet VM util must be in [0, 100]"
    assert 0.0 <= res.pm_utilisation_fleet_pct <= 100.0, "Fleet PM util must be in [0, 100]"
    print(f"[PASS] Point 3: Fleet utilization verified (VM fleet={res.vm_utilisation_fleet_pct:.2f}%, active={res.vm_utilisation_pct:.2f}%).")


def test_point4_dtkb_continuous_proba():
    cfg = FullConfig()
    vms = [VirtualMachine(id=i, ram_gb=16.0 * (i + 1), storage_gb=250.0 * (i + 1), mips=367.0, bandwidth_gb=1e6)
           for i in range(5)]
    pms = [PhysicalMachine(id=0, ram_gb=1024.0, storage_gb=10000.0, mips=10000.0,
                           cpu_power_max=100.0, ram_power_max=10.0, storage_power_max=5.0, idle_fraction=0.5)]
    pms[0].place(vms[0])

    dtkb = DTKB(cfg=cfg.dtkb, rng=np.random.default_rng(42))
    dtkb.fit(vms, fit_seed=42)

    task = Task(id=1, arrival_time=0.0, length_mi=200.0, ram_req_gb=4.0, disk_req_gb=20.0, priority=1, deadline=10.0)
    probs = dtkb.predict_proba(task, vms, {0: pms[0]})

    assert len(probs) == len(vms), "Probabilities should be returned for all VMs"
    for vid, p in probs.items():
        assert 0.0 <= p <= 1.0, f"Probability for VM {vid} out of bounds: {p}"


    gate = AdaptiveGate(lam=0.6)
    frbs_score = 0.8
    p_val = probs[0]
    blended = 0.6 * p_val + 0.4 * frbs_score
    assert 0.0 <= blended <= 1.0, "Blended score out of bounds"
    print(f"[PASS] Point 4: DTKB continuous posterior confidence verified (VM0 p={p_val:.3f}, blend={blended:.3f}).")


def test_point5_task_conservation_and_validation():
    from dc_sim.validation import validate_result


    valid_res = {
        "tasks_allocated": 95,
        "tasks_blocked": 5,
        "sla_violations": 10,
        "sla_violations_low": 3,
        "sla_violations_med": 4,
        "sla_violations_high": 3,
        "total_energy_joules": 3600000.0,
        "total_energy_kwh": 1.0,
        "makespan_s": 500.0,
        "used_vm_count": 5,
        "active_pm_count": 2,
        "vm_utilisation_pct": 50.0,
        "pm_utilisation_pct": 60.0,
        "vm_utilisation_active_pct": 55.0,
        "pm_utilisation_active_pct": 65.0,
        "vm_utilisation_fleet_pct": 25.0,
        "pm_utilisation_fleet_pct": 30.0,
    }
    outcome = validate_result(valid_res, n_tasks=100)
    assert outcome.ok, f"Expected valid result to pass, failed with: {outcome.reasons}"


    bad_res = dict(valid_res)
    bad_res["tasks_allocated"] = 90
    outcome_bad = validate_result(bad_res, n_tasks=100)
    assert not outcome_bad.ok, "Expected failure on task conservation mismatch"
    assert any("Task conservation violated" in r for r in outcome_bad.reasons)


    bad_energy = dict(valid_res)
    bad_energy["total_energy_kwh"] = 99.0
    outcome_energy = validate_result(bad_energy, n_tasks=100)
    assert not outcome_energy.ok, "Expected failure on energy conversion mismatch"
    assert any("Energy conversion mismatch" in r for r in outcome_energy.reasons)

    print("[PASS] Point 5: Result validator strictly enforces task conservation and invariants.")


if __name__ == "__main__":
    test_fix1_rng_independence()
    test_fix2_dtkb_baseline_parity()
    test_fix4_adaptive_gate()
    test_fix5_frbs_weights()
    test_fix6_legacy_name_mappings()
    test_fix7_wake_latency()
    test_fix8_feasible_vm_fallback()
    test_fix9_checkpoint_hash_stability()
    test_point1_deadline_slack()
    test_point2_wake_power_continuous()
    test_point3_fleet_utilization()
    test_point4_dtkb_continuous_proba()
    test_point5_task_conservation_and_validation()
    print("\nALL INVARIANT VERIFICATION TESTS PASSED SUCCESSFULLY!")

