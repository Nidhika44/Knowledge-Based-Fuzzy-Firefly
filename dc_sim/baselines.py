import numpy as np
from typing import List, Optional, Dict
from .entities import VirtualMachine, PhysicalMachine, Task
from .knowledge_base import DTKB
from .fuzzy_system import FRBS
from .firefly import (_marginal_energy_norm, _sla_feasibility, _migration_penalty,
                       _consolidation_bonus, _priority_match)
from .config import FullConfig


def combined_fitness(vid: int, vms_by_id, pms_by_id, task: Task, energy_cfg, weights) -> float:
    vm = vms_by_id[vid]
    pm = pms_by_id.get(vm.host_pm) if vm.host_pm is not None else None
    if pm is None:
        return -1e9
    e = _marginal_energy_norm(vm, pm, task, energy_cfg)
    s = _sla_feasibility(vm, task, pm, energy_cfg)
    m = _migration_penalty(vm)
    b = _consolidation_bonus(vm, pm)
    p = _priority_match(vm, task, pm)
    w = weights
    return (w.w_energy * (1 - e) + w.w_sla * s - w.w_migration * m +
            w.w_balance * b + w.w_priority * p)


def _feasible_vm_ids(task: Task, vms_by_id: Dict[int, VirtualMachine], all_vm_ids) -> List[int]:
    hw_capable = [vid for vid in all_vm_ids if vms_by_id[vid].ram_gb >= task.ram_req_gb and vms_by_id[vid].storage_gb >= task.disk_req_gb]
    if not hw_capable:
        return []
    has_headroom = [vid for vid in hw_capable if vms_by_id[vid].can_take(task)]
    return has_headroom if has_headroom else hw_capable


class BaseAllocator:
    name = "base"

    def __init__(self, cfg: FullConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def prepare(self, vms):
        pass

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids) -> Optional[int]:
        raise NotImplementedError


class BestFitDecreasing(BaseAllocator):
    name = "BFD"

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        def residual(vid):
            vm = vms_by_id[vid]
            demand = task.length_mi / max(task.exec_seconds_hint, 1e-6)
            return vm.total_mips() - (vm.used_mips + demand)
        return min(feas, key=residual)


class ModifiedBestFit(BaseAllocator):
    name = "MBF"

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        def residual_norm(vid):
            vm = vms_by_id[vid]
            demand = task.length_mi / max(task.exec_seconds_hint, 1e-6)
            r_mips = (vm.total_mips() - (vm.used_mips + demand)) / max(vm.total_mips(), 1e-6)
            r_ram = (vm.ram_gb - task.ram_req_gb) / max(vm.ram_gb, 1e-6)
            r_disk = (vm.storage_gb - task.disk_req_gb) / max(vm.storage_gb, 1e-6)
            return 0.5 * r_mips + 0.25 * r_ram + 0.25 * r_disk
        return min(feas, key=residual_norm)


class UtilBand(BaseAllocator):
    name = "UtilBand"

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        def score(vid):
            vm = vms_by_id[vid]
            pm = pms_by_id.get(vm.host_pm)
            active_bonus = 1.0 if (pm is not None and pm.active()) else 0.0
            u = pm.util_overall() if pm is not None else 0.0
            band = 1.0 - abs(u - 0.6)
            return active_bonus * 2.0 + band
        return max(feas, key=score)


class _PopulationMetaheuristic(BaseAllocator):
    max_gen = 10

    def _fit(self, vid, vms_by_id, pms_by_id, task):
        return combined_fitness(vid, vms_by_id, pms_by_id, task, self.cfg.energy, self.cfg.firefly)

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        return self._search(feas, vms_by_id, pms_by_id, task)

    def _search(self, feas, vms_by_id, pms_by_id, task):
        raise NotImplementedError


class HardCapacityAllocator(BaseAllocator):
    name = "Capacity_Filter"

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = []
        for vid, vm in vms_by_id.items():
            if vm.can_take(task):
                feas.append(vid)
        if not feas:
            return None
        return int(self.rng.choice(feas))

class PSOAllocator(_PopulationMetaheuristic):
    name = "PSO"

    def _search(self, feas, vms_by_id, pms_by_id, task):
        n = len(feas)
        pop = min(15, n)
        idx = self.rng.integers(0, n, size=pop).astype(float)
        vel = np.zeros(pop)
        pbest = idx.copy()
        pbest_val = np.array([self._fit(feas[int(round(i)) % n], vms_by_id, pms_by_id, task) for i in idx])
        gbest = pbest[int(np.argmax(pbest_val))]
        gbest_val = float(np.max(pbest_val))
        w, c1, c2 = 0.6, 1.4, 1.4
        for _ in range(self.max_gen):
            r1, r2 = self.rng.uniform(0, 1, pop), self.rng.uniform(0, 1, pop)
            vel = w * vel + c1 * r1 * (pbest - idx) + c2 * r2 * (gbest - idx)
            idx = np.clip(idx + vel, 0, n - 1)
            vals = np.array([self._fit(feas[int(round(i)) % n], vms_by_id, pms_by_id, task) for i in idx])
            improve = vals > pbest_val
            pbest[improve] = idx[improve]
            pbest_val[improve] = vals[improve]
            if vals.max() > gbest_val:
                gbest_val = float(vals.max())
                gbest = idx[int(np.argmax(vals))]
        return feas[int(round(gbest)) % n]


class GAAllocator(_PopulationMetaheuristic):
    name = "GA"

    def _search(self, feas, vms_by_id, pms_by_id, task):
        n = len(feas)
        pop_size = min(15, n)
        pop = self.rng.integers(0, n, size=pop_size)
        for _ in range(self.max_gen):
            fitness = np.array([self._fit(feas[i], vms_by_id, pms_by_id, task) for i in pop])
            order = np.argsort(-fitness)
            elite = pop[order[:max(2, pop_size // 4)]]
            children = []
            while len(children) < pop_size - len(elite):
                p1, p2 = self.rng.choice(elite, size=2)
                child = p1 if self.rng.random() < 0.5 else p2
                if self.rng.random() < 0.2:
                    child = int(self.rng.integers(0, n))
                children.append(child)
            pop = np.concatenate([elite, np.array(children, dtype=int)])
        fitness = np.array([self._fit(feas[i], vms_by_id, pms_by_id, task) for i in pop])
        return feas[pop[int(np.argmax(fitness))]]


class DEAllocator(_PopulationMetaheuristic):
    name = "DE"

    def _search(self, feas, vms_by_id, pms_by_id, task):
        n = len(feas)
        if n == 0:
            return None
        if n == 1:
            return feas[0]
        pop_size = min(15, n)
        if pop_size < 4:
            return max(feas, key=lambda v: self._fit(v, vms_by_id, pms_by_id, task))
        pop = self.rng.uniform(0, n - 1, size=pop_size)
        F, CR = 0.6, 0.7
        fitness = np.array([self._fit(feas[int(round(p)) % n], vms_by_id, pms_by_id, task) for p in pop])
        for _ in range(self.max_gen):
            for i in range(pop_size):
                idxs = [k for k in range(pop_size) if k != i]
                a, b, c = pop[self.rng.choice(idxs, size=3, replace=False)]
                mutant = np.clip(a + F * (b - c), 0, n - 1)
                trial = mutant if self.rng.random() < CR else pop[i]
                val = self._fit(feas[int(round(trial)) % n], vms_by_id, pms_by_id, task)
                if val > fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
        return feas[int(round(pop[int(np.argmax(fitness))])) % n]


class ACOAllocator(_PopulationMetaheuristic):
    name = "ACO"

    def _search(self, feas, vms_by_id, pms_by_id, task):
        n = len(feas)
        pheromone = np.ones(n)
        n_ants = min(15, n)
        best_vid, best_val = feas[0], -1e18
        for _ in range(self.max_gen):
            probs = pheromone / pheromone.sum()
            chosen = self.rng.choice(n, size=n_ants, p=probs)
            vals = np.array([self._fit(feas[c], vms_by_id, pms_by_id, task) for c in chosen])
            pheromone *= 0.9
            for c, v in zip(chosen, vals):
                pheromone[c] += max(0.0, v) * 1.5
            gi = int(np.argmax(vals))
            if vals[gi] > best_val:
                best_val = float(vals[gi])
                best_vid = feas[chosen[gi]]
        return best_vid


class OriginalFireflyAllocator(_PopulationMetaheuristic):
    name = "Original_FA"

    def _fit(self, vid, vms_by_id, pms_by_id, task):

        vm = vms_by_id[vid]
        u = vm.util_mips()
        return (task.priority + 1.0) / (u + 0.01)

    def _search(self, feas, vms_by_id, pms_by_id, task):
        from .firefly import FireflyAllocator
        fa = FireflyAllocator(cfg=self.cfg.firefly, energy_cfg=self.cfg.energy, rng=self.rng)
        winner, _trace = fa.select(feas, vms_by_id, pms_by_id, task,
                                    fitness_fn=lambda vid: self._fit(vid, vms_by_id, pms_by_id, task))
        return winner


class PureFireflyAllocator(_PopulationMetaheuristic):
    name = "Pure_FA"

    def _search(self, feas, vms_by_id, pms_by_id, task):
        from .firefly import FireflyAllocator
        fa = FireflyAllocator(cfg=self.cfg.firefly, energy_cfg=self.cfg.energy, rng=self.rng)
        winner, _trace = fa.select(feas, vms_by_id, pms_by_id, task,
                                    fitness_fn=lambda vid: self._fit(vid, vms_by_id, pms_by_id, task))
        return winner


class FuzzyOnlyAllocator(BaseAllocator):
    name = "Fuzzy_Only"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.frbs = FRBS.build(cfg.fuzzy, cfg.infra)

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        scores = [(vid, self.frbs.score(task, vms_by_id[vid])) for vid in feas]
        return max(scores, key=lambda t: t[1])[0]


class KBOnlyAllocator(BaseAllocator):
    name = "KB_Only"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.dtkb = DTKB(cfg=cfg.dtkb, rng=rng)

    def fit_knowledge_base(self, vms, pms=None, cfg=None, dynamic=False, seed=None):
        if hasattr(self.dtkb, 'fit_with_warmup') and pms and cfg:
            self.dtkb.fit_with_warmup(vms, pms, cfg, dynamic=dynamic, seed=seed)
        else:
            self.dtkb.fit(vms)

    def prepare(self, vms):
        self.dtkb.fit(vms)

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        vms = [vms_by_id[i] for i in feas]
        cands = self.dtkb.query(task, vms, pms_by_id)
        pool = cands if cands else feas
        def residual(vid):
            vm = vms_by_id[vid]
            demand = task.length_mi / max(task.exec_seconds_hint, 1e-6)
            return vm.total_mips() - (vm.used_mips + demand)
        winner = min(pool, key=residual)
        v = vms_by_id[winner]
        pm = pms_by_id.get(v.host_pm) if pms_by_id else None
        pm_s = {'SLEEP': 0, 'WAKING': 1, 'IDLE': 2, 'ACTIVE': 3}.get(pm.state, 3) if pm else 3
        task._feat = [v.ram_gb, v.storage_gb, max(0.0, 1.0 - v.util_mips()),
                      task.ram_req_gb, task.disk_req_gb, float(task.priority), float(pm_s)]
        return winner

    def observe_outcome(self, task, vm, met_deadline: bool):
        if vm is None or not hasattr(task, '_feat'):
            return
        started_cleanly = task.start_time is not None and (task.start_time - task.arrival_time) <= 1.0
        success = bool(met_deadline and started_cleanly)
        self.dtkb._rolling_log_X.append(task._feat)
        self.dtkb._rolling_log_y.append(int(success))
        self.dtkb._tasks_since_retrain += 1
        self.dtkb.maybe_retrain()


class KBFPAAllocator(_PopulationMetaheuristic):
    name = "KB_FPA"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.dtkb = DTKB(cfg=cfg.dtkb, rng=rng)

    def fit_knowledge_base(self, vms, pms=None, cfg=None, dynamic=False, seed=None):
        if hasattr(self.dtkb, 'fit_with_warmup') and pms and cfg:
            self.dtkb.fit_with_warmup(vms, pms, cfg, dynamic=dynamic, seed=seed)
        else:
            self.dtkb.fit(vms)

    def prepare(self, vms):
        self.dtkb.fit(vms)

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        vms = [vms_by_id[i] for i in feas]
        cands = self.dtkb.query(task, vms, pms_by_id)
        pool = cands if cands else feas
        if len(pool) > 15:

            pool_vms = [vms_by_id[i] for i in pool]
            probs = self.dtkb.predict_proba(task, pool_vms, pms_by_id)
            pool = sorted(pool, key=lambda vid: probs.get(vid, 0.0), reverse=True)[:15]
        winner = self._search(pool, vms_by_id, pms_by_id, task)
        if winner is not None:
            v = vms_by_id[winner]
            pm = pms_by_id.get(v.host_pm) if pms_by_id else None
            pm_s = {'SLEEP': 0, 'WAKING': 1, 'IDLE': 2, 'ACTIVE': 3}.get(pm.state, 3) if pm else 3
            task._feat = [v.ram_gb, v.storage_gb, max(0.0, 1.0 - v.util_mips()),
                          task.ram_req_gb, task.disk_req_gb, float(task.priority), float(pm_s)]
        return winner

    def observe_outcome(self, task, vm, met_deadline: bool):
        if vm is None or not hasattr(task, '_feat'):
            return
        started_cleanly = task.start_time is not None and (task.start_time - task.arrival_time) <= 1.0
        success = bool(met_deadline and started_cleanly)
        self.dtkb._rolling_log_X.append(task._feat)
        self.dtkb._rolling_log_y.append(int(success))
        self.dtkb._tasks_since_retrain += 1
        self.dtkb.maybe_retrain()

    def _search(self, feas, vms_by_id, pms_by_id, task):
        n = len(feas)
        if n == 0:
            return None
        if n == 1:
            return feas[0]
        pop_size = min(15, n)
        if pop_size < 2:
            return feas[0]
        pop = self.rng.uniform(0, n - 1, size=pop_size)
        p_switch = 0.8
        fitness = np.array([self._fit(feas[int(round(x)) % n], vms_by_id, pms_by_id, task) for x in pop])
        best = pop[int(np.argmax(fitness))]
        for _ in range(self.max_gen):
            for i in range(pop_size):
                if self.rng.random() < p_switch:
                    levy = self.rng.standard_cauchy() * 0.5
                    cand = pop[i] + levy * (best - pop[i])
                else:
                    if pop_size >= 2:
                        j, k = self.rng.choice(pop_size, size=2, replace=False)
                        cand = pop[i] + self.rng.uniform(0, 1) * (pop[j] - pop[k])
                    else:
                        cand = pop[i]
                cand = float(np.clip(cand, 0, n - 1))
                val = self._fit(feas[int(round(cand)) % n], vms_by_id, pms_by_id, task)
                if val > fitness[i]:
                    pop[i] = cand
                    fitness[i] = val
            cur_best_val = self._fit(feas[int(round(best)) % n], vms_by_id, pms_by_id, task)
            if fitness.max() > cur_best_val:
                best = pop[int(np.argmax(fitness))]
        return feas[int(round(best)) % n]


class FuzzySLWAllocator(BaseAllocator):
    name = "Fuzzy_SLW"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.frbs = FRBS.build(cfg.fuzzy, cfg.infra)
        self.max_gen = 10

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        vms = [vms_by_id[i] for i in feas]
        cands = self.frbs.query(task, vms)
        pool = cands if cands else feas
        if len(pool) > 15:

            pool = sorted(pool, key=lambda vid: self.frbs.score(task, vms_by_id[vid]), reverse=True)[:15]
        n = len(pool)
        pop_size = min(15, n)
        pos = self.rng.uniform(0, n - 1, size=pop_size)
        fit = lambda x: combined_fitness(pool[int(round(x)) % n], vms_by_id, pms_by_id, task,
                                          self.cfg.energy, self.cfg.firefly)
        vals = np.array([fit(x) for x in pos])
        leader = pos[int(np.argmax(vals))]
        for gen in range(self.max_gen):
            a = 2.0 * (1 - gen / self.max_gen)
            for i in range(pop_size):
                r = self.rng.uniform(0, 1)
                A = 2 * a * r - a
                C = 2 * self.rng.uniform(0, 1)
                D = abs(C * leader - pos[i])
                pos[i] = np.clip(leader - A * D, 0, n - 1)
            vals = np.array([fit(x) for x in pos])
            if vals.max() > fit(leader):
                leader = pos[int(np.argmax(vals))]
        return pool[int(round(leader)) % n]


class OriginalPSOAllocator(PSOAllocator):
    name = "Original_PSO"
    def _fit(self, vid, vms_by_id, pms_by_id, task):
        vm = vms_by_id[vid]
        return (task.priority + 1.0) / (vm.util_mips() + 0.01)

class OriginalGAAllocator(GAAllocator):
    name = "Original_GA"
    def _fit(self, vid, vms_by_id, pms_by_id, task):
        vm = vms_by_id[vid]
        return (task.priority + 1.0) / (vm.util_mips() + 0.01)

class OriginalDEAllocator(DEAllocator):
    name = "Original_DE"
    def _fit(self, vid, vms_by_id, pms_by_id, task):
        vm = vms_by_id[vid]
        return (task.priority + 1.0) / (vm.util_mips() + 0.01)

class OriginalACOAllocator(ACOAllocator):
    name = "Original_ACO"
    def _fit(self, vid, vms_by_id, pms_by_id, task):
        vm = vms_by_id[vid]
        return (task.priority + 1.0) / (vm.util_mips() + 0.01)


class GreedyFitnessAllocator(BaseAllocator):
    name = "Greedy_Fitness"

    def allocate(self, task, vms_by_id, pms_by_id, all_vm_ids):
        feas = _feasible_vm_ids(task, vms_by_id, all_vm_ids)
        if not feas:
            return None
        return max(feas, key=lambda vid: combined_fitness(
            vid, vms_by_id, pms_by_id, task, self.cfg.energy, self.cfg.firefly))


def make_baseline(name: str, cfg: FullConfig, rng: np.random.Generator) -> BaseAllocator:
    registry = {
        "BFD": BestFitDecreasing,
        "MBF": ModifiedBestFit,
        "UtilBand": UtilBand,
        "DVFS": UtilBand,
        "Greedy_Fitness": GreedyFitnessAllocator,
        "PSO": PSOAllocator,
        "ACO": ACOAllocator,
        "GA": GAAllocator,
        "DE": DEAllocator,
        "Pure_FA": PureFireflyAllocator,
        "OriginalFA": OriginalFireflyAllocator,
        "Fuzzy_Only": FuzzyOnlyAllocator,
        "KB_Only": KBOnlyAllocator,
        "Capacity_Filter": HardCapacityAllocator,
        "OriginalPSO": OriginalPSOAllocator,
        "OriginalGA": OriginalGAAllocator,
        "OriginalDE": OriginalDEAllocator,
        "OriginalACO": OriginalACOAllocator,

        "KB_FPA": KBFPAAllocator,
        "Fuzzy_SLW": FuzzySLWAllocator,
    }
    if name not in registry:
        raise KeyError(f"Unknown baseline '{name}'")
    return registry[name](cfg, rng)
