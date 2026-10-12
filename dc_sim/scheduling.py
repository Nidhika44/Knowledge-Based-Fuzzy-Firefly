import heapq
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from collections import deque

from .entities import VirtualMachine, PhysicalMachine, Task


@dataclass
class _RunningTask:
    task_id: int
    remaining_mi: float
    rate_mips: float


class BaseVMScheduler:
    name = "base"

    def __init__(self):
        self.running: Dict[int, Dict[int, _RunningTask]] = {}
        self.last_update: Dict[int, float] = {}
        self.current_finish: Dict[Tuple[int, int], float] = {}
        self.total_queue_wait_s: float = 0.0
        self.queued_task_count: int = 0

    def _ensure(self, vm_id: int):
        self.running.setdefault(vm_id, {})
        self.last_update.setdefault(vm_id, 0.0)

    def _sync_vm_load(self, vm: VirtualMachine, pm: Optional[PhysicalMachine], new_load: float):
        delta = new_load - vm.used_mips
        vm.used_mips = max(0.0, new_load)
        if pm is not None:
            pm.used_mips = max(0.0, pm.used_mips + delta)

    def submit(self, vm: VirtualMachine, pm: Optional[PhysicalMachine], task: Task,
               now: float, heap: List) -> bool:
        raise NotImplementedError

    def notify_finish(self, vm: VirtualMachine, pm: Optional[PhysicalMachine],
                       task_id: int, now: float, heap: List) -> bool:
        raise NotImplementedError


class TimeSharedVMScheduler(BaseVMScheduler):
    name = "time_shared"

    def _flush(self, vm: VirtualMachine, now: float):
        self._ensure(vm.id)
        elapsed = now - self.last_update[vm.id]
        running = self.running[vm.id]
        if elapsed > 0 and running:
            for rt in running.values():
                rt.remaining_mi = max(0.0, rt.remaining_mi - rt.rate_mips * elapsed)
        self.last_update[vm.id] = now

    def _reschedule(self, vm: VirtualMachine, now: float, heap: List):
        running = self.running[vm.id]
        n = len(running)
        if n == 0:
            return
        rate = vm.total_mips() / n
        for rt in running.values():
            rt.rate_mips = rate
            finish = now + (rt.remaining_mi / rate if rate > 0 else float("inf"))
            self.current_finish[(vm.id, rt.task_id)] = finish
            heapq.heappush(heap, (finish, vm.id, rt.task_id))

    def submit(self, vm, pm, task, now, heap):
        self._ensure(vm.id)
        self._flush(vm, now)
        self.running[vm.id][task.id] = _RunningTask(
            task_id=task.id, remaining_mi=task.length_mi, rate_mips=0.0)
        self._reschedule(vm, now, heap)
        self._sync_vm_load(vm, pm, vm.total_mips() if self.running[vm.id] else 0.0)
        task.start_time = now
        return True

    def notify_finish(self, vm, pm, task_id, now, heap):
        self._ensure(vm.id)
        running = self.running[vm.id]
        if task_id not in running:
            return False
        self._flush(vm, now)
        rt = running[task_id]
        if rt.remaining_mi > 1e-6:
            return False
        del running[task_id]
        self.current_finish.pop((vm.id, task_id), None)
        self._reschedule(vm, now, heap)
        self._sync_vm_load(vm, pm, vm.total_mips() if running else 0.0)
        return True


class SpaceSharedVMScheduler(BaseVMScheduler):
    name = "space_shared"

    def __init__(self):
        super().__init__()
        self.busy_pes: Dict[int, int] = {}
        self.queue: Dict[int, deque] = {}

    def _ensure(self, vm_id: int):
        super()._ensure(vm_id)
        self.busy_pes.setdefault(vm_id, 0)
        self.queue.setdefault(vm_id, deque())

    def submit(self, vm, pm, task, now, heap):
        self._ensure(vm.id)
        mips_per_pe = vm.mips_per_pe()
        if self.busy_pes[vm.id] < vm.num_pes:
            self.busy_pes[vm.id] += 1
            self.running[vm.id][task.id] = _RunningTask(
                task_id=task.id, remaining_mi=task.length_mi, rate_mips=mips_per_pe)
            finish = now + task.length_mi / mips_per_pe
            self.current_finish[(vm.id, task.id)] = finish
            heapq.heappush(heap, (finish, vm.id, task.id))
            self._sync_vm_load(vm, pm, self.busy_pes[vm.id] * mips_per_pe)
            task.start_time = now
            vm.queue_len = len(self.queue[vm.id])
            return True
        else:
            self.queue[vm.id].append((task, now))
            self.queued_task_count += 1
            vm.queue_len = len(self.queue[vm.id])
            return False

    def notify_finish(self, vm, pm, task_id, now, heap):
        self._ensure(vm.id)
        running = self.running[vm.id]
        if task_id not in running:
            return False
        expected = self.current_finish.get((vm.id, task_id))
        if expected is None or abs(expected - now) > 1e-6:
            return False
        del running[task_id]
        self.current_finish.pop((vm.id, task_id), None)
        self.busy_pes[vm.id] = max(0, self.busy_pes[vm.id] - 1)
        mips_per_pe = vm.mips_per_pe()

        if self.queue[vm.id]:
            next_task, arrival = self.queue[vm.id].popleft()
            self.total_queue_wait_s += max(0.0, now - arrival)
            self.busy_pes[vm.id] += 1
            self.running[vm.id][next_task.id] = _RunningTask(
                task_id=next_task.id, remaining_mi=next_task.length_mi, rate_mips=mips_per_pe)
            finish = now + next_task.length_mi / mips_per_pe
            self.current_finish[(vm.id, next_task.id)] = finish
            heapq.heappush(heap, (finish, vm.id, next_task.id))
            next_task.start_time = now

        vm.queue_len = len(self.queue[vm.id])
        self._sync_vm_load(vm, pm, self.busy_pes[vm.id] * mips_per_pe)
        return True

    def queue_length(self, vm_id: int) -> int:
        return len(self.queue.get(vm_id, ()))


def make_scheduler(policy: str) -> BaseVMScheduler:
    if policy == "time_shared":
        return TimeSharedVMScheduler()
    if policy == "space_shared":
        return SpaceSharedVMScheduler()
    raise ValueError(f"Unknown scheduling policy '{policy}' (expected 'time_shared' or 'space_shared')")
