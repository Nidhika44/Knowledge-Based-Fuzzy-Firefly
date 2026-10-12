import heapq
import time as _time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

from .entities import PhysicalMachine, VirtualMachine, Task
from .energy_model import pm_instantaneous_power_watts, energy_joules, peak_power
from .config import FullConfig
from .scheduling import make_scheduler


def provision_vms_on_pms(pms: List[PhysicalMachine], vms: List[VirtualMachine],
                          rng: np.random.Generator, policy: str = "bin_packing") -> None:
    if policy == "spread":
        order = list(vms)
        rng.shuffle(order)
        pm_cycle = list(pms)
        rng.shuffle(pm_cycle)
        i = 0
        n = len(pm_cycle)
        for vm in order:
            for _ in range(n):
                pm = pm_cycle[i % n]
                i += 1
                if pm.can_host(vm):
                    pm.place(vm)
                    break
    else:

        order = sorted(vms, key=lambda v: (v.ram_gb + v.storage_gb), reverse=True)
        for vm in order:
            best_pm = None
            best_residual = None
            for pm in pms:
                if pm.can_host(vm):
                    residual = (pm.ram_gb - pm.used_ram - vm.ram_gb) + \
                               (pm.storage_gb - pm.used_storage - vm.storage_gb)
                    if best_residual is None or residual < best_residual:
                        best_residual = residual
                        best_pm = pm
            if best_pm is not None:
                best_pm.place(vm)


    vms[:] = [v for v in vms if v.host_pm is not None]


@dataclass
class SimResult:
    algorithm: str
    n_tasks: int
    seed: int
    n_pms: int
    n_vms: int
    scheduling_policy: str = "time_shared"
    tasks_allocated: int = 0
    tasks_blocked: int = 0
    sla_violations: int = 0
    sla_violations_low: int = 0
    sla_violations_med: int = 0
    sla_violations_high: int = 0
    migrations: int = 0
    migration_energy_j: float = 0.0
    migration_energy_kwh: float = 0.0
    makespan_s: float = 0.0
    total_energy_joules: float = 0.0
    total_energy_kwh: float = 0.0
    mean_power_kw: float = 0.0
    active_pm_count: int = 0
    avg_active_pm_count: float = 0.0
    used_vm_count: int = 0
    vm_utilisation_pct: float = 0.0
    pm_utilisation_pct: float = 0.0
    vm_utilisation_active_pct: float = 0.0
    pm_utilisation_active_pct: float = 0.0
    vm_utilisation_fleet_pct: float = 0.0
    pm_utilisation_fleet_pct: float = 0.0
    resource_wastage_pct: float = 0.0
    wall_clock_runtime_s: float = 0.0
    avg_queue_wait_s: float = 0.0
    queued_task_count: int = 0
    convergence_trace_mean: List[float] = field(default_factory=list)

    mean_l1_frac: float = 0.0
    mean_l2_frac: float = 0.0
    mean_final_frac: float = 0.0
    mean_lambda: float = 0.5

    dtkb_test_accuracy: float = 0.0
    dtkb_test_precision: float = 0.0
    dtkb_test_recall: float = 0.0
    dtkb_n_training_records: int = 0
    dtkb_retrain_count: int = 0

    def to_dict(self):
        d = self.__dict__.copy()
        d["convergence_trace_mean"] = list(d["convergence_trace_mean"])
        return d


class DiscreteEventSimulator:
    def __init__(self, cfg: FullConfig, pms: List[PhysicalMachine], vms: List[VirtualMachine],
                 tasks: List[Task], algo, algorithm_name: str, seed: int,
                 migration_interval: int = 100, low_util_threshold: float = 0.10):
        self.cfg = cfg
        self.pms = pms
        self.vms = vms
        self.tasks = tasks
        self.algo = algo
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.migration_interval = migration_interval
        self.low_util_threshold = low_util_threshold

        self.pms_by_id: Dict[int, PhysicalMachine] = {p.id: p for p in pms}
        self.vms_by_id: Dict[int, VirtualMachine] = {v.id: v for v in vms}
        self.tasks_by_id: Dict[int, Task] = {t.id: t for t in tasks}
        self.all_vm_ids = [v.id for v in vms if v.host_pm is not None]

        self.scheduler = make_scheduler(cfg.scheduling.policy)

        self.clock = tasks[0].arrival_time if tasks else 0.0
        self.total_energy_j = 0.0
        self.migration_energy_j = 0.0
        self.vm_busy_integral: Dict[int, float] = {v.id: 0.0 for v in vms}
        self.pm_busy_integral: Dict[int, float] = {p.id: 0.0 for p in pms}
        self.vm_busy_duration: Dict[int, float] = {v.id: 0.0 for v in vms}
        self.pm_busy_duration: Dict[int, float] = {p.id: 0.0 for p in pms}
        self.used_vms_ever: Set[int] = set()
        self.active_pms_ever: Set[int] = set()
        self._active_pm_time_integral: float = 0.0
        self.migrations = 0
        self.sla_violations = 0
        self.sla_violations_by_priority: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self.tasks_allocated = 0
        self.tasks_blocked = 0

        self._busy_vm_ids: Set[int] = set()
        self._busy_pm_ids: Set[int] = set()

        self.event_heap: List = []
        self.pending_heap: List = []
        self.pm_waking_until: Dict[int, float] = {}
        self._seq_counter: int = 0


    def _mark_vm_busy(self, vm_id: int):
        self._busy_vm_ids.add(vm_id)
        vm = self.vms_by_id[vm_id]
        if vm.host_pm is not None:
            self._busy_pm_ids.add(vm.host_pm)

    def _refresh_vm_busy(self, vm_id: int):
        vm = self.vms_by_id.get(vm_id)
        if vm is None:
            self._busy_vm_ids.discard(vm_id)
            return
        if vm.used_mips <= 0:
            self._busy_vm_ids.discard(vm_id)
            if vm.host_pm is not None:
                pm = self.pms_by_id.get(vm.host_pm)
                if pm is not None and pm.used_mips <= 0:
                    self._busy_pm_ids.discard(vm.host_pm)

    def _advance_clock_to(self, t_target: float):
        dt = t_target - self.clock
        if dt <= 0:
            self.clock = max(self.clock, t_target)
            return

        self._active_pm_time_integral += len(self._busy_pm_ids) * dt


        for pm in self.pms:
            if pm.id in self._busy_pm_ids:

                pm.state = "ACTIVE"
                pm.last_idle_start = None
                power = pm_instantaneous_power_watts(pm, self.cfg.energy)
                self.total_energy_j += energy_joules(power, dt)
            elif pm.state == "WAKING":


                wake_pwr = self.cfg.energy.wake_energy_j / max(self.cfg.energy.wake_latency_s, 1e-6)
                self.total_energy_j += energy_joules(wake_pwr, dt)
                if t_target >= self.pm_waking_until.get(pm.id, 0.0):
                    pm.state = "ACTIVE"
                    pm.last_idle_start = None
            else:

                if pm.state == "ACTIVE":
                    pm.state = "IDLE"
                    pm.last_idle_start = self.clock

                if pm.state == "IDLE" and pm.last_idle_start is not None:
                    time_since_idle = self.clock - pm.last_idle_start
                    idle_remaining = max(0.0, self.cfg.energy.idle_timeout_s - time_since_idle)
                    idle_dt = min(dt, idle_remaining)
                    sleep_dt = dt - idle_dt

                    idle_pwr = self.cfg.energy.idle_fraction * peak_power(pm)
                    self.total_energy_j += idle_pwr * idle_dt

                    if sleep_dt > 0:
                        pm.state = "SLEEP"
                        sleep_pwr = self.cfg.energy.sleep_fraction * peak_power(pm)
                        self.total_energy_j += sleep_pwr * sleep_dt
                elif pm.state == "SLEEP":
                    sleep_pwr = self.cfg.energy.sleep_fraction * peak_power(pm)
                    self.total_energy_j += sleep_pwr * dt

        for pm_id in self._busy_pm_ids:
            pm = self.pms_by_id[pm_id]
            u = pm.util_mips()
            self.pm_busy_integral[pm_id] += u * dt
            self.pm_busy_duration[pm_id] += dt
        for vm_id in self._busy_vm_ids:
            vm = self.vms_by_id[vm_id]
            u = vm.util_mips()
            self.vm_busy_integral[vm_id] += u * dt
            self.vm_busy_duration[vm_id] += dt
        self.clock = t_target

    def _process_events_up_to(self, up_to_time: float):
        while True:
            has_event = bool(self.event_heap and self.event_heap[0][0] <= up_to_time)
            has_pending = bool(self.pending_heap and self.pending_heap[0][0] <= up_to_time)
            if not has_event and not has_pending:
                break

            if has_pending and (not has_event or self.pending_heap[0][0] <= self.event_heap[0][0]):
                start_time, _, task, vm_id = heapq.heappop(self.pending_heap)
                self._advance_clock_to(start_time)
                vm = self.vms_by_id[vm_id]
                pm = self.pms_by_id.get(vm.host_pm) if vm.host_pm is not None else None
                if pm is not None:
                    pm.state = "ACTIVE"
                    pm.last_idle_start = None
                    if pm.id in self.pm_waking_until and start_time >= self.pm_waking_until[pm.id]:
                        del self.pm_waking_until[pm.id]
                self.scheduler.submit(vm, pm, task, start_time, self.event_heap)
                self._mark_vm_busy(vm_id)
            else:
                finish_time, vm_id, task_id = heapq.heappop(self.event_heap)
                self._advance_clock_to(finish_time)
                vm = self.vms_by_id.get(vm_id)
                if vm is None:
                    continue
                pm = self.pms_by_id.get(vm.host_pm) if vm.host_pm is not None else None
                genuine = self.scheduler.notify_finish(vm, pm, task_id, finish_time, self.event_heap)
                if genuine:
                    self._refresh_vm_busy(vm_id)
                    task = self.tasks_by_id.get(task_id)
                    if task is not None:
                        task.finish_time = finish_time
                        if hasattr(self.algo, "observe_outcome") and task.assigned_vm is not None:
                            vm_obj = self.vms_by_id.get(task.assigned_vm)
                            self.algo.observe_outcome(task, vm_obj, not task.sla_violated())
                        if task.sla_violated():
                            self.sla_violations += 1
                            pri = getattr(task, 'priority', 0)
                            self.sla_violations_by_priority[pri] = \
                                self.sla_violations_by_priority.get(pri, 0) + 1

    def _maybe_migrate(self, task_index: int):
        if task_index == 0 or task_index % self.migration_interval != 0:
            return
        active = [p for p in self.pms_by_id.values() if p.active()]
        lightly_loaded = [p for p in active if p.util_overall() < self.low_util_threshold]
        heavier_candidates = sorted(
            [p for p in active if p.util_overall() >= self.low_util_threshold],
            key=lambda p: -p.util_overall())
        for src in lightly_loaded:
            for vm_id in list(src.hosted_vms):
                vm = self.vms_by_id[vm_id]
                if vm.used_mips > 0:
                    continue

                has_pending = any(task_item[3] == vm.id for task_item in self.pending_heap)
                if has_pending:
                    continue
                for tgt in heavier_candidates:
                    if tgt.id == src.id:
                        continue
                    if tgt.can_host(vm):
                        src.remove(vm)
                        tgt.place(vm)
                        vm.migrations += 1
                        self.migrations += 1

                        mig_e = self.cfg.energy.migration_energy_j * (vm.ram_gb / 32.0)
                        self.total_energy_j += mig_e
                        self.migration_energy_j += mig_e
                        break
            if len(src.hosted_vms) == 0 and src.used_mips <= 0:
                src.state = "SLEEP"
                src.last_idle_start = None

    def run(self) -> SimResult:
        t_start = _time.perf_counter()
        conv_means = []

        for i, task in enumerate(self.tasks):
            self._process_events_up_to(task.arrival_time)
            self._advance_clock_to(task.arrival_time)
            self._maybe_migrate(i)

            winner = self.algo.allocate(task, self.vms_by_id, self.pms_by_id, self.all_vm_ids)

            if hasattr(self.algo, "convergence_traces") and self.algo.convergence_traces:
                last_trace = self.algo.convergence_traces[-1]
                conv_means.append(last_trace[-1] if last_trace else 0.0)

            if winner is None:
                self.tasks_blocked += 1
                self.sla_violations += 1
                pri = getattr(task, 'priority', 0)
                self.sla_violations_by_priority[pri] = \
                    self.sla_violations_by_priority.get(pri, 0) + 1
                continue

            vm = self.vms_by_id[winner]
            pm = self.pms_by_id.get(vm.host_pm) if vm.host_pm is not None else None

            task.assigned_vm = winner
            if pm is not None:
                self.active_pms_ever.add(pm.id)
            self.used_vms_ever.add(vm.id)
            self.tasks_allocated += 1


            actual_start_time = task.arrival_time
            if pm is not None:
                if pm.state == "SLEEP":
                    wake_end = task.arrival_time + self.cfg.energy.wake_latency_s
                    self.pm_waking_until[pm.id] = wake_end
                    pm.state = "WAKING"
                    actual_start_time = wake_end
                elif pm.state == "WAKING" or (pm.id in self.pm_waking_until and task.arrival_time < self.pm_waking_until[pm.id]):
                    wake_end = self.pm_waking_until[pm.id]
                    actual_start_time = max(task.arrival_time, wake_end)

            if actual_start_time > task.arrival_time:
                self._seq_counter += 1
                heapq.heappush(self.pending_heap, (actual_start_time, self._seq_counter, task, winner))
            else:
                if pm is not None and pm.state == "IDLE":
                    pm.state = "ACTIVE"
                    pm.last_idle_start = None
                self.scheduler.submit(vm, pm, task, actual_start_time, self.event_heap)
                self._mark_vm_busy(winner)

        while self.event_heap or self.pending_heap:
            next_t = float('inf')
            if self.event_heap:
                next_t = min(next_t, self.event_heap[0][0])
            if self.pending_heap:
                next_t = min(next_t, self.pending_heap[0][0])
            self._process_events_up_to(next_t)

        makespan = self.clock - (self.tasks[0].arrival_time if self.tasks else 0.0)
        wall_clock = _time.perf_counter() - t_start

        used_vms = self.used_vms_ever
        active_pms = self.active_pms_ever
        vm_util = (float(np.mean([self.vm_busy_integral[v] / max(makespan, 1e-9) for v in used_vms])) * 100.0
                   if used_vms else 0.0)
        pm_util = (float(np.mean([self.pm_busy_integral[p] / max(makespan, 1e-9) for p in active_pms])) * 100.0
                   if active_pms else 0.0)
        vm_util_active = (float(np.mean([self.vm_busy_integral[v] / max(self.vm_busy_duration[v], 1e-9)
                                          for v in used_vms if self.vm_busy_duration[v] > 0])) * 100.0
                           if any(self.vm_busy_duration[v] > 0 for v in used_vms) else 0.0)
        pm_util_active = (float(np.mean([self.pm_busy_integral[p] / max(self.pm_busy_duration[p], 1e-9)
                                          for p in active_pms if self.pm_busy_duration[p] > 0])) * 100.0
                           if any(self.pm_busy_duration[p] > 0 for p in active_pms) else 0.0)


        total_pms_count = max(len(self.pms), 1)
        total_vms_count = max(len(self.vms), 1)
        pm_util_fleet = float(sum(self.pm_busy_integral[p.id] / max(makespan, 1e-9) for p in self.pms) / total_pms_count) * 100.0
        vm_util_fleet = float(sum(self.vm_busy_integral[v.id] / max(makespan, 1e-9) for v in self.vms) / total_vms_count) * 100.0

        wastage = max(0.0, 100.0 - pm_util)

        queued_count = getattr(self.scheduler, "queued_task_count", 0)
        total_wait = getattr(self.scheduler, "total_queue_wait_s", 0.0)
        avg_wait = total_wait / queued_count if queued_count > 0 else 0.0
        avg_active_pms = self._active_pm_time_integral / max(makespan, 1e-9)


        total_energy_kwh = self.total_energy_j / 3_600_000.0
        makespan_h = makespan / 3600.0
        mean_power_kw = (total_energy_kwh / makespan_h) if makespan_h > 0 else 0.0
        migration_energy_kwh = self.migration_energy_j / 3_600_000.0


        cand_diag = {}
        if hasattr(self.algo, 'get_candidate_diagnostics'):
            cand_diag = self.algo.get_candidate_diagnostics()


        dtkb_outputs = {}
        if hasattr(self.algo, 'dtkb') and self.algo.dtkb.model is not None:
            dtkb_outputs = self.algo.dtkb.get_validation_outputs()

        return SimResult(
            algorithm=self.algorithm_name, n_tasks=len(self.tasks), seed=self.seed,
            n_pms=len(self.pms), n_vms=len(self.vms),
            scheduling_policy=self.cfg.scheduling.policy,
            tasks_allocated=self.tasks_allocated, tasks_blocked=self.tasks_blocked,
            sla_violations=self.sla_violations,
            sla_violations_low=self.sla_violations_by_priority.get(0, 0),
            sla_violations_med=self.sla_violations_by_priority.get(1, 0),
            sla_violations_high=self.sla_violations_by_priority.get(2, 0),
            migrations=self.migrations,
            migration_energy_j=self.migration_energy_j,
            migration_energy_kwh=migration_energy_kwh,
            makespan_s=makespan,
            total_energy_joules=self.total_energy_j,
            total_energy_kwh=total_energy_kwh,
            mean_power_kw=mean_power_kw,
            active_pm_count=len(active_pms), avg_active_pm_count=avg_active_pms,
            used_vm_count=len(used_vms),
            vm_utilisation_pct=vm_util, pm_utilisation_pct=pm_util,
            vm_utilisation_active_pct=vm_util_active, pm_utilisation_active_pct=pm_util_active,
            vm_utilisation_fleet_pct=vm_util_fleet, pm_utilisation_fleet_pct=pm_util_fleet,
            resource_wastage_pct=wastage, wall_clock_runtime_s=wall_clock,
            avg_queue_wait_s=avg_wait, queued_task_count=queued_count,
            convergence_trace_mean=conv_means[:200],
            mean_l1_frac=cand_diag.get("mean_l1_frac", 0.0),
            mean_l2_frac=cand_diag.get("mean_l2_frac", 0.0),
            mean_final_frac=cand_diag.get("mean_final_frac", 0.0),
            mean_lambda=cand_diag.get("mean_lambda", 0.5),
            dtkb_test_accuracy=dtkb_outputs.get("dtkb_test_accuracy", 0.0),
            dtkb_test_precision=dtkb_outputs.get("dtkb_test_precision", 0.0),
            dtkb_test_recall=dtkb_outputs.get("dtkb_test_recall", 0.0),
            dtkb_n_training_records=dtkb_outputs.get("dtkb_n_training_records", 0),
            dtkb_retrain_count=dtkb_outputs.get("dtkb_retrain_count", 0),
        )
