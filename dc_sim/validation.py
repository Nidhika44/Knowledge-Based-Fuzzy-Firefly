from dataclasses import dataclass, field
from typing import List, Dict, Any
import math


@dataclass
class ValidationOutcome:
    ok: bool
    reasons: List[str] = field(default_factory=list)

    def __bool__(self):
        return self.ok


def _is_bad_number(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    except Exception:
        return True


def validate_result(result: Dict[str, Any], n_tasks: int, tolerance: float = 1e-3) -> ValidationOutcome:
    reasons = []

    numeric_fields = [
        "vm_utilisation_pct", "pm_utilisation_pct",
        "vm_utilisation_active_pct", "pm_utilisation_active_pct",
        "vm_utilisation_fleet_pct", "pm_utilisation_fleet_pct",
        "total_energy_joules", "total_energy_kwh", "makespan_s", "sla_violations",
        "tasks_allocated", "tasks_blocked", "active_pm_count", "used_vm_count",
    ]
    for f in numeric_fields:
        if f in result and _is_bad_number(result[f]):
            reasons.append(f"'{f}' is NaN/inf/None ({result.get(f)!r})")

    for f in ("vm_utilisation_pct", "pm_utilisation_pct",
              "vm_utilisation_active_pct", "pm_utilisation_active_pct",
              "vm_utilisation_fleet_pct", "pm_utilisation_fleet_pct"):
        v = result.get(f)
        if v is not None and not _is_bad_number(v) and v > 100.0 + tolerance:
            reasons.append(f"'{f}' = {v:.2f}% exceeds 100% -- capacity-accounting bug")

    for f in ("total_energy_joules", "total_energy_kwh", "makespan_s", "sla_violations", "migrations",
              "tasks_allocated", "tasks_blocked", "active_pm_count", "used_vm_count"):
        v = result.get(f)
        if v is not None and not _is_bad_number(v) and v < 0:
            reasons.append(f"'{f}' = {v} is negative")


    ej = result.get("total_energy_joules")
    ekwh = result.get("total_energy_kwh")
    if ej is not None and ekwh is not None and not _is_bad_number(ej) and not _is_bad_number(ekwh):
        expected_kwh = ej / 3.6e6
        if abs(ekwh - expected_kwh) > 1e-3:
            reasons.append(f"Energy conversion mismatch: total_energy_kwh ({ekwh}) vs joules/3.6e6 ({expected_kwh})")

    if n_tasks > 0:
        allocated = result.get("tasks_allocated", 0) or 0
        blocked = result.get("tasks_blocked", 0) or 0

        if (allocated + blocked) != n_tasks:
            reasons.append(f"Task conservation violated: allocated ({allocated}) + blocked ({blocked}) != n_tasks ({n_tasks})")

        if allocated == 0 and blocked >= n_tasks:
            reasons.append(f"ALL {n_tasks} tasks were blocked -- allocator placed nothing")


        sla_v = result.get("sla_violations", 0) or 0
        if sla_v > n_tasks:
            reasons.append(f"SLA violations ({sla_v}) exceeds total task count ({n_tasks})")
        sla_low = result.get("sla_violations_low", 0) or 0
        sla_med = result.get("sla_violations_med", 0) or 0
        sla_high = result.get("sla_violations_high", 0) or 0
        if (sla_low + sla_med + sla_high > 0) and (sla_low + sla_med + sla_high != sla_v):
            reasons.append(f"Priority SLA sum ({sla_low}+{sla_med}+{sla_high}={sla_low+sla_med+sla_high}) != total SLA violations ({sla_v})")

    if (result.get("tasks_allocated", 0) or 0) > 0:
        if (result.get("makespan_s", 0) or 0) <= 0:
            reasons.append("tasks were allocated but makespan_s <= 0")
        if (result.get("total_energy_joules", 0) or 0) <= 0:
            reasons.append("tasks were allocated but total_energy_joules <= 0")
        if (result.get("used_vm_count", 0) or 0) == 0:
            reasons.append("tasks were allocated but used_vm_count == 0")

    return ValidationOutcome(ok=(len(reasons) == 0), reasons=reasons)

