from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PhysicalMachine:
    id: int
    ram_gb: float
    storage_gb: float
    mips: float
    cpu_power_max: float
    ram_power_max: float
    storage_power_max: float
    idle_fraction: float

    used_ram: float = 0.0
    used_storage: float = 0.0


    reserved_mips: float = 0.0


    used_mips: float = 0.0
    hosted_vms: List[int] = field(default_factory=list)


    state: str = "SLEEP"
    last_idle_start: Optional[float] = None

    def active(self) -> bool:
        return len(self.hosted_vms) > 0

    def util_ram(self) -> float:
        return self.used_ram / self.ram_gb if self.ram_gb > 0 else 0.0

    def util_storage(self) -> float:
        return self.used_storage / self.storage_gb if self.storage_gb > 0 else 0.0

    def util_mips(self) -> float:
        return self.used_mips / self.mips if self.mips > 0 else 0.0

    def util_overall(self) -> float:

        return (self.util_ram() + self.util_storage() + self.util_mips()) / 3.0

    def can_host(self, vm: "VirtualMachine", burst: float = 0.0) -> bool:
        return (self.used_ram + vm.ram_gb <= self.ram_gb and
                self.used_storage + vm.storage_gb <= self.storage_gb and
                self.reserved_mips + vm.total_mips() <= self.mips * (1.0 + burst))

    def place(self, vm: "VirtualMachine"):
        self.used_ram += vm.ram_gb
        self.used_storage += vm.storage_gb
        self.reserved_mips += vm.total_mips()
        self.hosted_vms.append(vm.id)
        vm.host_pm = self.id

    def remove(self, vm: "VirtualMachine"):
        self.used_ram -= vm.ram_gb
        self.used_storage -= vm.storage_gb
        self.reserved_mips -= vm.total_mips()
        if vm.id in self.hosted_vms:
            self.hosted_vms.remove(vm.id)
        vm.host_pm = None
        vm.host_pm = None


@dataclass
class VirtualMachine:
    id: int
    ram_gb: float
    storage_gb: float
    mips: float
    bandwidth_gb: float
    host_pm: Optional[int] = None
    assigned_tasks: List[int] = field(default_factory=list)
    used_mips: float = 0.0
    migrations: int = 0
    num_pes: int = 2


    queue_len: int = 0

    def util_mips(self) -> float:
        return self.used_mips / self.total_mips() if self.total_mips() > 0 else 0.0

    def total_mips(self) -> float:
        return self.mips * max(self.num_pes, 1)

    def mips_per_pe(self) -> float:
        return self.mips

    def can_take(self, task: "Task", burst: float = 0.10) -> bool:
        if task.ram_req_gb > self.ram_gb:
            return False
        if task.disk_req_gb > self.storage_gb:
            return False
        return self.used_mips + task.length_mi / max(task.exec_seconds_hint, 1e-6) \
            <= self.total_mips() * (1.0 + burst)


@dataclass
class Task:
    id: int
    arrival_time: float
    length_mi: float
    ram_req_gb: float
    disk_req_gb: float
    priority: int
    deadline: float
    exec_seconds_hint: float = 1.0
    assigned_vm: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    phase: str = "default"

    def sla_violated(self) -> bool:
        if self.finish_time is None:
            return True
        return self.finish_time > self.deadline
