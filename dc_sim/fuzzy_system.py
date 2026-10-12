import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from .entities import VirtualMachine, Task, PhysicalMachine
from .config import FuzzyConfig


def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def three_term_membership(x: float, low: float, med: float, high: float) -> Dict[str, float]:
    span_lo = max(med - low, 1e-6)
    span_hi = max(high - med, 1e-6)

    if x <= low:
        mu_L = 1.0
    elif x < med:
        mu_L = (med - x) / span_lo
    else:
        mu_L = 0.0

    mu_M = triangular_mf(x, low, med, high)

    if x <= med:
        mu_H = 0.0
    elif x < high:
        mu_H = (x - med) / span_hi
    else:
        mu_H = 1.0

    return {"L": mu_L, "M": mu_M, "H": mu_H}


@dataclass
class FRBS:
    cfg: FuzzyConfig


    anchors: Dict[str, tuple] = None

    @classmethod
    def build(cls, cfg: FuzzyConfig, infra_cfg) -> "FRBS":
        anchors = {
            "ram": (0.10, 0.40, 0.80),
            "storage": (0.10, 0.40, 0.80),
            "cpu": (0.15, 0.50, 0.85),
            "bandwidth": (0.25, 0.50, 0.85),
            "migration_cost": (0.20, 0.50, 0.80),
            "deadline_slack": (0.20, 0.50, 0.80),
            "deadline_urgency": (0.20, 0.50, 0.80),
            "priority": (0.0, 1.0, 2.0),
        }
        return cls(cfg=cfg, anchors=anchors)

    def _dim_value(self, dim: str, vm: VirtualMachine, task: Task, pm_of_vm: PhysicalMachine = None) -> float:
        if dim == "ram":

            return max(0.0, (vm.ram_gb - task.ram_req_gb) / max(vm.ram_gb, 1e-6))
        if dim == "storage":

            return max(0.0, (vm.storage_gb - task.disk_req_gb) / max(vm.storage_gb, 1e-6))
        if dim == "cpu":
            return max(0.0, 1.0 - vm.util_mips())
        if dim == "bandwidth":
            return min(1.0, vm.bandwidth_gb / 2_000_000.0)
        if dim == "migration_cost":


            return max(0.0, 1.0 - (vm.ram_gb / 64.0))
        if dim in ("deadline_slack", "deadline_urgency"):

            core_speed = max(vm.mips_per_pe(), 1e-6)
            est_exec = task.length_mi / core_speed
            q_len = getattr(vm, 'queue_len', 0)
            pes = max(vm.num_pes, 1)
            if q_len > 0:
                est_queue = (q_len / pes) * est_exec
            else:
                busy_ratio = max(0.0, vm.util_mips())
                est_queue = (busy_ratio * 0.5 * est_exec) if busy_ratio >= 1.0 else 0.0
            est_total = est_exec + est_queue
            available_slack = max(0.0, task.deadline - task.arrival_time)

            return min(1.0, available_slack / (3.0 * max(est_total, 1e-6)))
        if dim == "priority":


            headroom = max(0.0, 1.0 - vm.util_mips())
            if task.priority == 2:
                return headroom
            elif task.priority == 1:
                return 0.5
            else:
                q_len = getattr(vm, 'queue_len', 0)
                return max(0.0, vm.util_mips() - 0.2 * q_len)
        raise ValueError(dim)

    def suitability(self, vm: VirtualMachine, task: Task) -> float:

        if task.ram_req_gb > vm.ram_gb or task.disk_req_gb > vm.storage_gb:
            return 0.0


        m_ram = three_term_membership(self._dim_value("ram", vm, task), *self.anchors["ram"])
        m_str = three_term_membership(self._dim_value("storage", vm, task), *self.anchors["storage"])
        m_cpu = three_term_membership(self._dim_value("cpu", vm, task), *self.anchors["cpu"])


        w_010 = m_ram["L"] + m_cpu["L"] - m_ram["L"] * m_cpu["L"]
        w_090 = m_ram["H"] * m_cpu["H"] * (m_str["M"] + m_str["H"])
        w_055 = max(0.0, (1.0 - w_010) - w_090)
        core_score = 0.10 * w_010 + 0.90 * w_090 + 0.55 * w_055


        w_ram, w_str, w_cpu, w_bw, w_mig, w_slk, w_pri = self.cfg.dim_weights
        w_core = w_ram + w_str + w_cpu

        slk_anchor = self.anchors.get("deadline_slack", self.anchors.get("deadline_urgency", (0.20, 0.50, 0.80)))
        m_bw = three_term_membership(self._dim_value("bandwidth", vm, task), *self.anchors["bandwidth"])
        m_mig = three_term_membership(self._dim_value("migration_cost", vm, task), *self.anchors["migration_cost"])
        m_slk = three_term_membership(self._dim_value("deadline_slack", vm, task), *slk_anchor)
        m_pri = three_term_membership(self._dim_value("priority", vm, task), *self.anchors["priority"])

        bw_score = m_bw["H"] * 0.9 + m_bw["M"] * 0.5 + m_bw["L"] * 0.1
        mig_score = m_mig["H"] * 0.9 + m_mig["M"] * 0.5 + m_mig["L"] * 0.1
        slk_score = m_slk["H"] * 0.9 + m_slk["M"] * 0.5 + m_slk["L"] * 0.1
        pri_score = m_pri["H"] * 0.9 + m_pri["M"] * 0.5 + m_pri["L"] * 0.1

        total_rass = w_core * core_score + w_bw * bw_score + w_mig * mig_score + w_slk * slk_score + w_pri * pri_score
        return float(np.clip(total_rass, 0.0, 1.0))

    def rule_class(self, n: float) -> str:
        if n < 0.25:
            return "low"
        if n < 0.50:
            return "lower_medium"
        if n < 0.75:
            return "upper_medium"
        return "high"

    def query(self, task: Task, vms: List[VirtualMachine], threshold: float = 0.5) -> List[int]:
        out = []
        for vm in vms:
            if task.ram_req_gb > vm.ram_gb or task.disk_req_gb > vm.storage_gb:
                continue
            s = self.suitability(vm, task)
            if s >= threshold:
                out.append(vm.id)
        return out

    def score(self, task: Task, vm: VirtualMachine) -> float:
        return self.suitability(vm, task)
