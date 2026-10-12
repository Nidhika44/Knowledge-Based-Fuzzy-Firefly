import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .entities import VirtualMachine, PhysicalMachine, Task
from .knowledge_base import DTKB
from .fuzzy_system import FRBS
from .firefly import FireflyAllocator
from .config import FullConfig


@dataclass
class AdaptiveGate:
    lam: float = 0.5
    dtkb_hits: int = 0
    dtkb_attempts: int = 0
    frbs_hits: int = 0
    frbs_attempts: int = 0
    prior_alpha: float = 2.0

    def update(self, source: str, success: bool, credit: float = 1.0):
        if source == "dtkb":
            self.dtkb_attempts += 1
            if success:
                self.dtkb_hits += 1
        elif source == "frbs":
            self.frbs_attempts += 1
            if success:
                self.frbs_hits += 1


        p_dtkb = (self.dtkb_hits + self.prior_alpha) / (self.dtkb_attempts + 2.0 * self.prior_alpha)
        p_frbs = (self.frbs_hits + self.prior_alpha) / (self.frbs_attempts + 2.0 * self.prior_alpha)

        total_p = p_dtkb + p_frbs
        raw_lam = (p_dtkb / total_p) if total_p > 0 else 0.5
        self.lam = float(np.clip(raw_lam, 0.05, 0.95))


class KBFuzzyFirefly:

    def __init__(self, cfg: FullConfig, rng: np.random.Generator, variant: str = "full"):
        self.cfg = cfg
        self.rng = rng
        self.variant = variant
        self.dtkb = DTKB(cfg=cfg.dtkb, rng=rng)
        self.frbs = FRBS.build(cfg.fuzzy, cfg.infra)
        self.fa = FireflyAllocator(cfg=cfg.firefly, energy_cfg=cfg.energy, rng=rng)
        self.gate = AdaptiveGate()
        self.convergence_traces: List[List[float]] = []


        self._l1_fracs: List[float] = []
        self._l2_fracs: List[float] = []
        self._final_fracs: List[float] = []
        self.gate_trace: List[float] = []


        self._l1: set = set()
        self._l2: set = set()
        self._last_gated_scores: Dict[int, float] = {}


        self.dtkb_hits: int = 0
        self.dtkb_attempts: int = 0
        self.frbs_hits: int = 0
        self.frbs_attempts: int = 0

    def fit_knowledge_base(self, vms, pms=None, cfg=None, dynamic=False, seed=None):
        if hasattr(self.dtkb, 'fit_with_warmup') and pms and cfg:
            self.dtkb.fit_with_warmup(vms, pms, cfg, dynamic=dynamic, seed=seed)
        else:
            self.dtkb.fit(vms)
    def _candidate_list(self, task: Task, vms_by_id, pms_by_id, all_vm_ids) -> List[int]:
        vms = [vms_by_id[i] for i in all_vm_ids]
        pool_size = max(len(all_vm_ids), 1)


        self._l1 = set()
        self._l2 = set()

        if self.variant == "no_fuzzy":
            self._l1 = set(self.dtkb.query(task, vms, pms_by_id))
            self._l1_fracs.append(len(self._l1) / pool_size)
            self._l2_fracs.append(0.0)
            self._final_fracs.append(len(self._l1) / pool_size)
            return list(self._l1)

        if self.variant == "no_dtkb":
            self._l2 = set(self.frbs.query(task, vms))
            self._l1_fracs.append(0.0)
            self._l2_fracs.append(len(self._l2) / pool_size)
            self._final_fracs.append(len(self._l2) / pool_size)
            return list(self._l2)

        l1 = set(self.dtkb.query(task, vms, pms_by_id))
        l2 = set(self.frbs.query(task, vms))


        self._l1 = l1
        self._l2 = l2

        self._l1_fracs.append(len(l1) / pool_size)
        self._l2_fracs.append(len(l2) / pool_size)

        dtkb_probs = self.dtkb.predict_proba(task, vms, pms_by_id) if hasattr(self.dtkb, 'predict_proba') else {}

        if self.variant == "no_adaptive_gate":
            final = list(l1 | l2)
            self._final_fracs.append(len(final) / pool_size)
            self._last_gated_scores = {vid: 0.5 * dtkb_probs.get(vid, float(vid in l1)) + 0.5 * self.frbs.score(task, vms_by_id[vid])
                                       for vid in final}
            return final


        lam = self.gate.lam
        tau = getattr(self.cfg.fuzzy, 'gate_tau', 0.35)
        scored = {vid: lam * dtkb_probs.get(vid, float(vid in l1)) + (1.0 - lam) * self.frbs.score(task, vms_by_id[vid])
                  for vid in (l1 | l2)}
        cands = [vid for vid, sc in scored.items() if sc >= tau]
        if not cands:
            cands = list(l1 | l2)

        self._final_fracs.append(len(cands) / pool_size)
        self._last_gated_scores = scored
        return cands

    def allocate(self, task: Task, vms_by_id, pms_by_id, all_vm_ids) -> Optional[int]:
        self._latest_vms = list(vms_by_id.values())
        candidates = self._candidate_list(task, vms_by_id, pms_by_id, all_vm_ids)
        feasible = [vid for vid in candidates if vms_by_id[vid].can_take(task)]
        if not feasible:
            feasible = [vid for vid in all_vm_ids if vms_by_id[vid].can_take(task)]
        if not feasible:

            feasible = [vid for vid in all_vm_ids
                        if vms_by_id[vid].ram_gb >= task.ram_req_gb and vms_by_id[vid].storage_gb >= task.disk_req_gb]
        if not feasible:
            return None


        preselect_fn = None
        if self.variant == "no_dtkb":
            scores = {vid: self.frbs.score(task, vms_by_id[vid]) for vid in feasible}
            feasible.sort(key=scores.get, reverse=True)
            preselect_fn = scores.get
        elif self.variant == "no_fuzzy":
            scores = {vid: float(vid in self._l1) for vid in feasible}
            feasible.sort(key=scores.get, reverse=True)
            preselect_fn = scores.get
        else:

            scores = {vid: self._last_gated_scores.get(vid, self.frbs.score(task, vms_by_id[vid])) for vid in feasible}
            feasible.sort(key=scores.get, reverse=True)
            preselect_fn = scores.get

        if self.variant == "no_energy_fitness":

            def orig_fitness(vid):
                vm = vms_by_id[vid]
                p = task.length_mi / max(vm.mips_per_pe(), 1e-6)
                r = vm.util_mips()
                return -(p / (r + 0.01 + 1e-6))

            winner, trace = self.fa.select(feasible, vms_by_id, pms_by_id, task,
                                           fitness_fn=orig_fitness,
                                           preselect_score_fn=preselect_fn)
        else:


            winner, trace = self.fa.select(feasible, vms_by_id, pms_by_id, task, preselect_score_fn=preselect_fn)

        self.convergence_traces.append(trace)

        if winner is not None:
            self.gate_trace.append(self.gate.lam)
            task._l1_winner = (winner in self._l1)
            task._l2_winner = (winner in self._l2)

            v = vms_by_id[winner]
            pm = pms_by_id.get(v.host_pm) if v.host_pm is not None else None
            pm_s = {'SLEEP': 0, 'WAKING': 1, 'IDLE': 2, 'ACTIVE': 3}.get(pm.state, 3) if pm else 3
            task._feat = [v.ram_gb, v.storage_gb, max(0.0, 1.0 - v.util_mips()),
                          task.ram_req_gb, task.disk_req_gb, float(task.priority), float(pm_s)]

        return winner

    def observe_outcome(self, task: Task, vm: VirtualMachine, met_deadline: bool):
        if vm is None:
            return

        started_cleanly = task.start_time is not None and (task.start_time - task.arrival_time) <= 1.0
        success = bool(met_deadline and started_cleanly)

        in_l1 = getattr(task, '_l1_winner', False)
        in_l2 = getattr(task, '_l2_winner', False)
        if in_l1:
            self.dtkb_attempts += 1
            if success:
                self.dtkb_hits += 1
        if in_l2:
            self.frbs_attempts += 1
            if success:
                self.frbs_hits += 1

        if in_l1 and not in_l2:
            self.gate.update('dtkb', success, credit=1.0)
        elif in_l2 and not in_l1:
            self.gate.update('frbs', success, credit=1.0)
        elif in_l1 and in_l2:

            self.gate.update('dtkb', success, credit=0.5)
            self.gate.update('frbs', success, credit=0.5)

        if self.variant != 'no_dtkb' and hasattr(task, '_feat'):
            self.dtkb._rolling_log_X.append(task._feat)
            self.dtkb._rolling_log_y.append(int(success))
            self.dtkb._tasks_since_retrain += 1
            self.dtkb.maybe_retrain()

    def get_candidate_diagnostics(self) -> Dict[str, float]:
        n = max(len(self._l1_fracs), 1)
        return {
            "mean_l1_frac": float(np.mean(self._l1_fracs)) if self._l1_fracs else 0.0,
            "mean_l2_frac": float(np.mean(self._l2_fracs)) if self._l2_fracs else 0.0,
            "mean_final_frac": float(np.mean(self._final_fracs)) if self._final_fracs else 0.0,
            "mean_lambda": float(np.mean(self.gate_trace)) if self.gate_trace else 0.5,
            "dtkb_success_rate": float(self.dtkb_hits / max(self.dtkb_attempts, 1)),
            "frbs_success_rate": float(self.frbs_hits / max(self.frbs_attempts, 1)),
        }


def _would_meet_sla(vm: VirtualMachine, task: Task) -> bool:
    exec_time = task.length_mi / max(vm.mips, 1e-6)
    return task.arrival_time + exec_time <= task.deadline
