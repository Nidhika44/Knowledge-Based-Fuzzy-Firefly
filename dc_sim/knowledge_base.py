import numpy as np

_WARMUP_CACHE = {}
from dataclasses import dataclass, field
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from typing import List, Optional, Dict, Any
from .entities import VirtualMachine, Task, PhysicalMachine
from .config import DTKBConfig


def _compute_bin_edges(col: np.ndarray, n_bins: int = 3) -> np.ndarray:
    try:
        qs = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        return qs[1:-1]
    except Exception:
        return np.array([])


def _apply_bins(X: np.ndarray, bin_edges_list: List[np.ndarray]) -> np.ndarray:
    Xd = np.empty_like(X, dtype=np.int32)
    for c, edges in enumerate(bin_edges_list):
        if len(edges) == 0:
            Xd[:, c] = 0
        else:
            Xd[:, c] = np.searchsorted(edges, X[:, c])
    return Xd


@dataclass
class DTKB:
    cfg: DTKBConfig
    rng: np.random.Generator
    model: DecisionTreeClassifier = None
    train_accuracy_: float = 0.0
    test_accuracy_: float = 0.0
    test_precision_: float = 0.0
    test_recall_: float = 0.0
    confusion_matrix_: Optional[np.ndarray] = None
    n_training_records_: int = 0
    retrain_count_: int = 0
    _rolling_log_X: list = field(default_factory=list)
    _rolling_log_y: list = field(default_factory=list)
    _tasks_since_retrain: int = 0

    _bin_edges: Optional[List[np.ndarray]] = field(default_factory=list)

    def _priority_margin(self, priority: int) -> float:

        return {0: 1.05, 1: 1.15, 2: 1.30}.get(priority, 1.10)

    def _synthesize_historical_data(self, vms: List[VirtualMachine], n_records: int, fit_seed: Optional[int] = None):
        rng = np.random.default_rng((int(fit_seed), 54321)) if fit_seed is not None else self.rng
        vm_idx = rng.integers(0, len(vms), size=n_records)
        X, y = [], []
        for idx in vm_idx:
            vm = vms[idx]
            task_ram = rng.uniform(0.5, 48.0)
            task_disk = rng.uniform(1.0, 150.0)
            priority = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))


            used_mips = rng.uniform(0.0, vm.total_mips())
            headroom_ratio = (vm.total_mips() - used_mips) / max(vm.total_mips(), 1e-6)


            task_mi = rng.lognormal(9.0, 0.9)
            exec_time = task_mi / max(vm.mips, 1e-6)


            slack = rng.uniform(*{0: (1.5, 5.0), 1: (1.2, 3.0), 2: (1.01, 1.5)}[priority])


            est_queue_delay = max(0.0, (1.0 - headroom_ratio)) * 50.0
            sla_ok = (est_queue_delay + exec_time) <= (exec_time * slack)
            fits_cap = (vm.ram_gb >= task_ram) and (vm.storage_gb >= task_disk)
            suitable = int(fits_cap and sla_ok)

            pm_s = rng.choice([0.0, 1.0, 2.0])
            X.append([vm.ram_gb, vm.storage_gb, headroom_ratio, task_ram, task_disk, priority, pm_s])
            y.append(suitable)
        return np.array(X, dtype=float), np.array(y, dtype=int)

    def fit(self, vms: List[VirtualMachine], fit_seed: Optional[int] = None):
        X, y = self._synthesize_historical_data(vms, self.cfg.n_historical_records, fit_seed=fit_seed)
        self._fit_from_arrays(X, y, fit_seed=fit_seed)

    def fit_with_warmup(self, vms, pms, cfg_root, n_warm=3000, dynamic=False, seed: Optional[int] = None):
        global _WARMUP_CACHE

        warm_seed_val = int(seed) if seed is not None else 42
        cache_key = (warm_seed_val, dynamic)

        if cache_key in _WARMUP_CACHE:
            X, y = _WARMUP_CACHE[cache_key]
            self.warmup_positive_rate_ = float(y.mean()) if len(y) else None
            if len(np.unique(y)) > 1:
                self._fit_from_arrays(X, y, fit_seed=warm_seed_val)
            else:
                self.fit(vms, fit_seed=warm_seed_val)
            return

        from dc_sim.workload import generate_workload
        from dc_sim.config import DynamicWorkloadConfig, PhaseSpec
        from dc_sim.baselines import _feasible_vm_ids, BaseAllocator
        from dc_sim.simulator import DiscreteEventSimulator
        import copy


        rng_warm = np.random.default_rng((warm_seed_val, 42_000))

        if dynamic:
            ph = cfg_root.dynamic_workload.phases[0]
            ph = PhaseSpec(**{**ph.__dict__, 'n_tasks': n_warm})
            from dc_sim.workload import generate_dynamic_workload
            tasks = generate_dynamic_workload(DynamicWorkloadConfig(phases=(ph,)), rng_warm)
        else:
            tasks = generate_workload(cfg_root.workload, n_warm, rng_warm)

        feats, X, y = {}, [], []

        class Explore(BaseAllocator):
            def allocate(s, task, vb, pb, ids):
                f = _feasible_vm_ids(task, vb, ids)
                if not f:
                    return None
                w = int(s.rng.choice(f))
                v = vb[w]
                pm = pb.get(v.host_pm) if pb else None
                pm_s = {'SLEEP': 0, 'WAKING': 1, 'IDLE': 2, 'ACTIVE': 3}.get(pm.state, 3) if pm else 3
                feats[task.id] = [v.ram_gb, v.storage_gb, max(0.0, 1.0 - v.util_mips()),
                                  task.ram_req_gb, task.disk_req_gb, float(task.priority), float(pm_s)]
                return w

            def observe_outcome(s, task, vm, met):
                if task.id in feats:
                    X.append(feats.pop(task.id))
                    started_cleanly = task.start_time is not None and (task.start_time - task.arrival_time) <= 1.0
                    success = bool(met and started_cleanly)
                    y.append(int(success))

        sim = DiscreteEventSimulator(cfg_root, copy.deepcopy(pms), copy.deepcopy(vms), tasks,
                                     Explore(cfg_root, np.random.default_rng(0)), 'Warm', 1)
        sim.run()
        X = np.array(X, float)
        y = np.array(y, int)
        self.warmup_positive_rate_ = float(y.mean()) if len(y) else None
        if len(np.unique(y)) > 1:
            _WARMUP_CACHE[cache_key] = (X, y)
            self._fit_from_arrays(X, y, fit_seed=warm_seed_val)
        else:
            self.fit(vms, fit_seed=warm_seed_val)

    def _fit_from_arrays(self, X: np.ndarray, y: np.ndarray, fit_seed: Optional[int] = None):
        n_cols = X.shape[1]
        self._bin_edges = []
        for c in range(n_cols):
            if c == 6:
                self._bin_edges.append(np.array([0.5, 1.5]))
            else:
                self._bin_edges.append(_compute_bin_edges(X[:, c], n_bins=self.cfg.branching_factor))
        Xd = _apply_bins(X, self._bin_edges)
        classes, counts = np.unique(y, return_counts=True)
        strat = y if (len(classes) > 1 and counts.min() >= 2) else None
        split_seed = int(fit_seed) if fit_seed is not None else 42
        Xtr, Xte, ytr, yte = train_test_split(
            Xd, y, test_size=self.cfg.test_size,
            random_state=split_seed, stratify=strat)
        clf = DecisionTreeClassifier(criterion=self.cfg.criterion, max_depth=self.cfg.max_depth,
                                     ccp_alpha=self.cfg.ccp_alpha,
                                     random_state=split_seed)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        self.train_accuracy_ = accuracy_score(ytr, clf.predict(Xtr))
        self.test_accuracy_ = accuracy_score(yte, yp)
        self.test_precision_ = float(precision_score(yte, yp, zero_division=0))
        self.test_recall_ = float(recall_score(yte, yp, zero_division=0))
        self.confusion_matrix_ = confusion_matrix(yte, yp, labels=[0, 1])
        self.n_training_records_ = len(Xtr)
        self.model = clf

    def predict_proba(self, task: Task, vms: List[VirtualMachine], pms_by_vm=None) -> Dict[int, float]:
        if not vms or self.model is None:
            return {}
        n = len(vms)
        X = np.empty((n, 7), dtype=float)
        for i, vm in enumerate(vms):
            pm = pms_by_vm.get(vm.host_pm) if pms_by_vm else None
            pm_s = {'SLEEP': 0, 'WAKING': 1, 'IDLE': 2, 'ACTIVE': 3}.get(pm.state, 3) if pm else 3
            X[i, 0] = vm.ram_gb
            X[i, 1] = vm.storage_gb
            X[i, 2] = 1.0 - vm.util_mips()
            X[i, 3] = task.ram_req_gb
            X[i, 4] = task.disk_req_gb
            X[i, 5] = float(task.priority)
            X[i, 6] = float(pm_s)

        if self._bin_edges:
            Xd = _apply_bins(X, self._bin_edges)
        else:
            fresh_edges = [_compute_bin_edges(X[:, c], self.cfg.branching_factor) for c in range(X.shape[1])]
            Xd = _apply_bins(X, fresh_edges)
        if hasattr(self.model, "predict_proba"):
            classes = list(self.model.classes_)
            if 1 in classes:
                idx_1 = classes.index(1)
                probs = self.model.predict_proba(Xd)[:, idx_1]
            else:
                probs = np.zeros(n, dtype=float)
        else:
            preds = self.model.predict(Xd)
            probs = preds.astype(float)
        return {vms[i].id: float(probs[i]) for i in range(n)}

    def query(self, task: Task, vms: List[VirtualMachine], pms_by_vm=None) -> List[int]:
        probs = self.predict_proba(task, vms, pms_by_vm)
        return [vid for vid, p in probs.items() if p >= 0.5]

    def log_outcome(self, vm: VirtualMachine, task: Task, sla_ok: bool):
        headroom_ratio = 1.0 - vm.util_mips()
        self._rolling_log_X.append([vm.ram_gb, vm.storage_gb, headroom_ratio, task.ram_req_gb, task.disk_req_gb, task.priority, 2.0])
        self._rolling_log_y.append(int(sla_ok))
        self._tasks_since_retrain += 1

    def maybe_retrain(self, vms=None):
        if self._tasks_since_retrain < self.cfg.retrain_every_n_tasks:
            return
        if len(self._rolling_log_X) >= 200:
            X = np.array(self._rolling_log_X[-5000:], float)
            y = np.array(self._rolling_log_y[-5000:], int)
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) > 1 and counts.min() >= 20:
                self._fit_from_arrays(X, y)
                self.retrain_count_ += 1
        self._tasks_since_retrain = 0

    def get_validation_outputs(self) -> Dict[str, Any]:
        return {
            "dtkb_test_accuracy": self.test_accuracy_,
            "dtkb_test_precision": self.test_precision_,
            "dtkb_test_recall": self.test_recall_,
            "dtkb_confusion_matrix": (self.confusion_matrix_.tolist()
                                      if self.confusion_matrix_ is not None else None),
            "dtkb_n_training_records": self.n_training_records_,
            "dtkb_retrain_count": self.retrain_count_,
            "dtkb_label_definition": (
                "suitable=1 iff: (vm.ram_gb >= task_ram_req AND vm.storage_gb >= task_disk_req) "
                "AND (est_queue_delay + exec_time <= exec_time * slack), where "
                "est_queue_delay = max(0, 1-headroom_ratio) * 50.0s, "
                "slack ~ U({0:(1.5,5.0), 1:(1.2,3.0), 2:(1.01,1.5)}[priority])"
            ),
        }
