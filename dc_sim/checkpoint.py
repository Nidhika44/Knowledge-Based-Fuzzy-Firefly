import json
import os
import pickle
from typing import Optional, Dict, Any


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)

    @staticmethod
    def key(algorithm: str, workload_size: int, seed: int, policy: str = "time_shared") -> str:
        return f"{algorithm}__w{workload_size}__s{seed}__{policy}"

    def is_done(self, algorithm: str, workload_size: int, seed: int, policy: str = "time_shared") -> bool:
        return self.key(algorithm, workload_size, seed, policy) in self.manifest

    def load_result(self, algorithm: str, workload_size: int, seed: int, policy: str = "time_shared"):
        k = self.key(algorithm, workload_size, seed, policy)
        if k not in self.manifest:
            return None
        path = os.path.join(self.dir, f"{k}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_result(self, algorithm: str, workload_size: int, seed: int, result_obj,
                     policy: str = "time_shared"):
        k = self.key(algorithm, workload_size, seed, policy)
        path = os.path.join(self.dir, f"{k}.pkl")
        with open(path, "wb") as f:
            pickle.dump(result_obj, f)
        self.manifest[k] = {"path": path, "status": "done"}
        self._save_manifest()

    def all_results(self):
        out = []
        for k, meta in self.manifest.items():
            path = meta.get("path")
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    out.append(pickle.load(f))
        return out

    def reset(self):
        import shutil
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest = {}
