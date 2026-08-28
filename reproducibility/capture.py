"""Utilities for experiment provenance and reproducibility capture."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys, time, random
from pathlib import Path
import numpy as np

SEEDS = [42, 123, 2024, 3407, 7777]

def set_global_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def git_info(repo_root: str | Path = ".") -> dict:
    root=Path(repo_root)
    def run(*args):
        try: return subprocess.check_output(["git","-C",str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()
        except Exception: return None
    status=run("status","--porcelain") or ""
    return {"commit":run("rev-parse","HEAD"),"branch":run("branch","--show-current"),"status_porcelain":status,"is_dirty":bool(status)}

def environment_info() -> dict:
    info={"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"os":platform.system(),"os_release":platform.release(),"machine":platform.machine(),"processor":platform.processor()}
    try:
        import psutil
        info["cpu_count"]=psutil.cpu_count(logical=True); info["ram_bytes"]=psutil.virtual_memory().total
    except Exception: pass
    try:
        import torch
        info.update({"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda})
        if torch.cuda.is_available():
            prop=torch.cuda.get_device_properties(0)
            info["gpu"]=torch.cuda.get_device_name(0); info["gpu_count"]=torch.cuda.device_count(); info["gpu_properties"]={"total_memory_bytes":prop.total_memory,"multi_processor_count":prop.multi_processor_count}
    except Exception: pass
    for pkg in ["numpy","pandas","sklearn","yaml","pyarrow"]:
        try: info[pkg]=getattr(__import__(pkg),"__version__","unknown")
        except Exception: pass
    try:
        import torch_geometric; info["torch_geometric"]=torch_geometric.__version__
    except Exception: pass
    return info

def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str),encoding="utf-8")

def capture_environment(output_dir,repo_root="."):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    info=environment_info(); info["git"]=git_info(repo_root); write_json(out/"environment.json",info)
    try: (out/"pip_freeze.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True),encoding="utf-8")
    except Exception: pass
    try:
        n=subprocess.run(["nvidia-smi"],capture_output=True,text=True); (out/"nvidia_smi.txt").write_text(n.stdout+n.stderr,encoding="utf-8")
    except Exception: pass
    return info
