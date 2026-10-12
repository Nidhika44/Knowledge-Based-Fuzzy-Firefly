import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from .entities import VirtualMachine, PhysicalMachine, Task
from .energy_model import pm_instantaneous_power_watts
from .config import FireflyConfig, EnergyModelConfig


def _marginal_energy_norm(vm: VirtualMachine, pm: PhysicalMachine, task: Task,
                           energy_cfg: EnergyModelConfig) -> float:
    if pm.state == 'SLEEP':
        return 1.0
    if pm.state == 'WAKING':
        return 0.8
    if pm.state == 'IDLE':
        return 0.3
    before = pm_instantaneous_power_watts(pm, energy_cfg)
    pm.used_mips += task.length_mi / max(task.exec_seconds_hint, 1e-6)
    after = pm_instantaneous_power_watts(pm, energy_cfg)
    pm.used_mips -= task.length_mi / max(task.exec_seconds_hint, 1e-6)
    delta = max(0.0, after - before)
    ref = max(pm.cpu_power_max + pm.ram_power_max + pm.storage_power_max, 1e-6)
    return float(np.clip(delta / ref, 0.0, 0.3))


def _sla_feasibility(vm: VirtualMachine, task: Task, pm: Optional[PhysicalMachine] = None,
                     energy_cfg: Optional[EnergyModelConfig] = None) -> float:

    core_speed = max(vm.mips_per_pe(), 1e-6)
    exec_time = task.length_mi / core_speed
    slack = max(0.0, task.deadline - task.arrival_time)
    if slack <= 0:
        return 0.0

    wake_lat = getattr(energy_cfg, 'wake_latency_s', 5.0) if energy_cfg else 5.0
    wake_delay = wake_lat if (pm is not None and pm.state in ('SLEEP', 'WAKING')) else 0.0


    q_len = getattr(vm, 'queue_len', 0)
    pes = max(vm.num_pes, 1)
    if q_len > 0:
        est_queue = (q_len / pes) * max(task.exec_seconds_hint, exec_time)
    else:
        busy_ratio = max(0.0, vm.util_mips())
        est_queue = (busy_ratio * 0.5 * max(task.exec_seconds_hint, 1.0)) if busy_ratio >= 1.0 else 0.0

    est_total = exec_time + wake_delay + est_queue
    return float(np.clip(1.0 - (est_total / max(slack, 1e-6)), 0.0, 1.0))


def _migration_penalty(vm: VirtualMachine) -> float:

    return float(np.clip(vm.ram_gb / 64.0, 0.0, 1.0))


def _consolidation_bonus(vm: VirtualMachine, pm: PhysicalMachine) -> float:
    if pm.state in ('SLEEP', 'WAKING'):
        return 0.0
    return float(np.clip(pm.util_overall(), 0.0, 1.0))


def _priority_match(vm: VirtualMachine, task: Task, pm: Optional[PhysicalMachine] = None) -> float:
    headroom = max(0.0, 1.0 - vm.util_mips())
    pm_awake = 1.0 if (pm is not None and pm.state == 'ACTIVE') else (0.5 if (pm and pm.state == 'IDLE') else 0.1)
    if task.priority == 2:
        return float(np.clip(headroom * pm_awake, 0.0, 1.0))
    elif task.priority == 1:
        return float(np.clip(0.5 * headroom + 0.5 * pm_awake, 0.0, 1.0))
    else:

        return float(np.clip(vm.util_mips(), 0.0, 1.0))


@dataclass
class FireflyAllocator:
    cfg: FireflyConfig
    energy_cfg: EnergyModelConfig
    rng: np.random.Generator
    convergence_log: Optional[List[float]] = None

    def fitness(self, vm_id: int, vms_by_id, pms_by_id, task: Task) -> float:
        vm = vms_by_id[vm_id]
        pm = pms_by_id.get(vm.host_pm) if vm.host_pm is not None else None
        if pm is None:
            return -1e9
        e = _marginal_energy_norm(vm, pm, task, self.energy_cfg)
        s = _sla_feasibility(vm, task, pm, self.energy_cfg)
        m = _migration_penalty(vm)
        b = _consolidation_bonus(vm, pm)
        p = _priority_match(vm, task, pm)
        c = self.cfg
        return (c.w_energy * (1.0 - e) + c.w_sla * s - c.w_migration * m +
                c.w_balance * b + c.w_priority * p)

    def select(self, candidate_vm_ids: List[int], vms_by_id, pms_by_id,
               task: Task, fitness_fn=None, preselect_score_fn=None) -> Tuple[int, List[float]]:
        if not candidate_vm_ids:
            return None, []
        fitness_fn = fitness_fn or (lambda vid: self.fitness(vid, vms_by_id, pms_by_id, task))

        if len(candidate_vm_ids) > self.cfg.max_population:
            if preselect_score_fn is not None:
                scored = sorted(candidate_vm_ids, key=preselect_score_fn, reverse=True)
                candidate_vm_ids = scored[:self.cfg.max_population]
            else:
                candidate_vm_ids = list(self.rng.choice(
                    candidate_vm_ids, size=self.cfg.max_population, replace=False))

        n = len(candidate_vm_ids)
        positions = np.arange(n, dtype=float)


        _fit_cache: Dict[int, float] = {}

        def _cached_fitness(idx: int) -> float:
            if idx not in _fit_cache:
                _fit_cache[idx] = fitness_fn(candidate_vm_ids[idx % n])
            return _fit_cache[idx]

        brightness = np.array([_cached_fitness(int(round(p)) % n) for p in positions])

        trace = []
        overall_best_idx = int(np.argmax(brightness))
        overall_best_val = float(brightness[overall_best_idx])
        overall_best_vm = candidate_vm_ids[int(round(positions[overall_best_idx])) % n]

        best_val = overall_best_val
        stale = 0
        for gen in range(self.cfg.max_generations):
            moved = False
            for i in range(n):
                for j in range(n):
                    if brightness[j] > brightness[i]:
                        r = abs(positions[i] - positions[j])
                        r_norm = r / max(float(n), 1.0)
                        beta = self.cfg.beta0 * np.exp(-self.cfg.gamma * (r_norm ** 2))
                        eps = self.rng.uniform(0, 1)
                        new_pos = positions[i] + beta * (positions[j] - positions[i]) + self.cfg.alpha * (eps - 0.5)
                        new_pos = float(np.clip(new_pos, 0, n - 1))
                        if new_pos != positions[i]:
                            positions[i] = new_pos
                            moved = True
            if moved:
                _fit_cache.clear()
            positions = np.clip(positions, 0, n - 1)
            idxs = np.round(positions).astype(int) % n
            brightness = np.array([_cached_fitness(k) for k in idxs])
            gen_best_idx = int(np.argmax(brightness))
            gen_best = float(brightness[gen_best_idx])
            trace.append(gen_best)


            if gen_best > overall_best_val:
                overall_best_val = gen_best
                overall_best_vm = candidate_vm_ids[int(round(positions[gen_best_idx])) % n]

            if abs(gen_best - best_val) < self.cfg.plateau_eps:
                stale += 1
            else:
                stale = 0
            best_val = max(best_val, gen_best)
            if stale >= self.cfg.plateau_patience:
                break

        return overall_best_vm, trace
