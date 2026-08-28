"""Create a repository-wide provenance manifest before or after experiments."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .capture import sha256_file, git_info, environment_info

SOURCE_DIRS = ['models','graph','data','training','experiments','figures','configs']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/reproducibility/experiment_manifest.json'); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; files=[]
    for d in SOURCE_DIRS:
        for p in (root/d).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                files.append({'path':str(p.relative_to(root)),'sha256':sha256_file(p),'bytes':p.stat().st_size})
    manifest={'git':git_info(root),'environment':environment_info(),'source_files':sorted(files,key=lambda x:x['path'])}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    print(args.output)
if __name__=='__main__': main()
