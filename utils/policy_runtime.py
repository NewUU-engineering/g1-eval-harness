"""
utils/policy_runtime.py

Small wrapper so adapters see a policy as:  numpy obs -> numpy action, plus reset().

Backends
  torch : torch.jit.load (the normal path; needs PyTorch installed)
  numpy : pure-numpy re-implementation for legged_gym / unitree_rl_gym TorchScript exports
          (PolicyExporterLSTM, or a plain nn.Sequential MLP of Linear + ELU/ReLU/Tanh).
          Reads the .pt zip directly, no torch needed. Useful on machines without PyTorch
          and as an independent cross-check of the torch path.
  auto  : torch if importable, else numpy.

IMPORTANT for recurrent policies: the LSTM hidden state lives INSIDE the exported module.
reset() must be called at the start of every episode, and you must never call the policy
with dummy inputs (e.g. to "probe" the input size), because that writes into the memory.
"""
import pickle
import zipfile
from typing import Optional
import numpy as np


class Policy:
    """numpy in, numpy out."""
    backend: str = "?"
    is_recurrent: bool = False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ----------------------------------------------------------------------------- torch
class TorchScriptPolicy(Policy):
    backend = "torch"

    def __init__(self, path: str, device: str = "cpu"):
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.module = torch.jit.load(path, map_location=self.device)
        self.module.eval()
        self.is_recurrent = hasattr(self.module, "reset_memory")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        t = self._torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            return self.module(t).squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        if self.is_recurrent:
            self.module.reset_memory()


# ----------------------------------------------------------------------------- numpy
_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


class _Obj:
    def __init__(self, *a, **k):
        self.__dict__["_s"] = {}

    def __setstate__(self, s):
        self.__dict__["_s"] = s if isinstance(s, dict) else {"_state": s}

    def __getattr__(self, k):
        try:
            return self.__dict__["_s"][k]
        except KeyError:
            raise AttributeError(k)


def _load_torchscript_tree(path: str):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *args):
                    (arr,) = storage
                    if len(size) == 0:
                        return arr[offset].copy()
                    it = arr.itemsize
                    return np.lib.stride_tricks.as_strided(arr[offset:], shape=size,
                                                           strides=[s * it for s in stride]).copy()
                return rebuild
            if mod == "torch" and name.endswith("Storage"):
                return name
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            return type(name, (_Obj,), {})

        def persistent_load(self, pid):
            _, stype, key, _loc, _numel = pid
            stype = stype if isinstance(stype, str) else getattr(stype, "__name__", str(stype))
            if stype not in _DTYPES:
                raise ValueError(f"Unsupported tensor storage {stype}")
            return (np.frombuffer(z.read(f"{root}/data/{key}"), dtype=_DTYPES[stype]),)

    return _U(z.open(f"{root}/data.pkl")).load()


def _children(mod) -> list:
    items = [(k, v) for k, v in mod._s.items() if k.isdigit()]
    return [v for _, v in sorted(items, key=lambda kv: int(kv[0]))]


def _act(name: str):
    if name == "ELU":
        return lambda y: np.where(y > 0, y, np.expm1(np.minimum(y, 0)))
    if name == "ReLU":
        return lambda y: np.maximum(y, 0)
    if name == "Tanh":
        return np.tanh
    if name == "Identity":
        return lambda y: y
    raise ValueError(f"numpy backend does not support activation {name}")


def _sequential(mod):
    layers = []
    for child in _children(mod):
        cname = type(child).__name__
        if "weight" in child._s:
            W, b = child._s["weight"].astype(np.float32), child._s.get("bias")
            b = np.zeros(W.shape[0], np.float32) if b is None else b.astype(np.float32)
            layers.append(("linear", W, b))
        else:
            layers.append(("act", _act(cname.split("___")[0]), None))
    return layers


def _run(layers, x):
    for kind, a, b in layers:
        x = a @ x + b if kind == "linear" else a(x)
    return x


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyTorchScriptPolicy(Policy):
    backend = "numpy"

    def __init__(self, path: str):
        root = _load_torchscript_tree(path)
        s = root._s
        if "memory" in s and "actor" in s:            # legged_gym PolicyExporterLSTM
            mem = s["memory"]._s
            n_layers = len([k for k in mem if k.startswith("weight_ih_l")])
            self.lstm = [tuple(mem[f"{n}_l{l}"].astype(np.float32)
                               for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")) for l in range(n_layers)]
            self.hidden = self.lstm[0][1].shape[1]
            self.actor = _sequential(s["actor"])
            self.is_recurrent = True
        elif any(k.isdigit() for k in s):              # plain nn.Sequential MLP
            self.lstm, self.actor = None, _sequential(root)
        elif "actor" in s:                             # PolicyExporter (MLP wrapped in .actor)
            self.lstm, self.actor = None, _sequential(s["actor"])
        else:
            raise ValueError("numpy backend: unrecognised TorchScript module layout; install torch.")
        self.reset()

    def reset(self) -> None:
        if self.lstm is not None:
            self.h = np.zeros((len(self.lstm), self.hidden), np.float32)
            self.c = np.zeros((len(self.lstm), self.hidden), np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.lstm is not None:
            H = self.hidden
            for l, (wi, wh, bi, bh) in enumerate(self.lstm):   # PyTorch gate order: i, f, g, o
                g = wi @ x + bi + wh @ self.h[l] + bh
                i, f, gg, o = _sig(g[:H]), _sig(g[H:2 * H]), np.tanh(g[2 * H:3 * H]), _sig(g[3 * H:])
                self.c[l] = f * self.c[l] + i * gg
                self.h[l] = o * np.tanh(self.c[l])
                x = self.h[l]
        return _run(self.actor, x).astype(np.float32)


def load_policy(path: str, backend: str = "auto", device: str = "cpu") -> Policy:
    if backend not in ("auto", "torch", "numpy"):
        raise ValueError("backend must be auto, torch or numpy")
    if backend in ("auto", "torch"):
        try:
            return TorchScriptPolicy(path, device)
        except (ImportError, OSError) as e:   # OSError: broken CUDA/torch install
            if backend == "torch":
                raise
            print(f"[policy_runtime] PyTorch unavailable ({type(e).__name__}); using numpy backend.")
    return NumpyTorchScriptPolicy(path)
