"""Create a machine-readable manifest for the exact dataset and chronological split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
try:
    import pandas as pd
except ImportError:
    pd = None
from .capture import sha256_file

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data/synthetic'); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--output',default='results/reproducibility/dataset_manifest.json'); args=ap.parse_args()
    d=Path(args.data_dir); cfg=yaml.safe_load(Path(args.config).read_text()); tables={}
    for name in ['students','institutions','verifiers','credentials','events']:
        p=d/f'{name}.parquet'; rec={'sha256':sha256_file(p), 'file_bytes':p.stat().st_size}
        if pd is not None:
            try:
                df=pd.read_parquet(p); rec.update({'rows':int(len(df)),'columns':list(df.columns)})
            except ImportError:
                rec['metadata_unavailable']='Install pyarrow or fastparquet to extract Parquet metadata.'
                tables[name]=rec; continue
        else:
            rec['metadata_unavailable']='Install pandas and a Parquet engine to extract metadata.'
            tables[name]=rec; continue
        if name=='events':
            if 'relation_type' in df: rec['relation_counts']={str(k):int(v) for k,v in df['relation_type'].value_counts().sort_index().items()}
            if 'fraud_label' in df: rec['fraud_count']=int(df['fraud_label'].sum())
            if 'fraud_type' in df: rec['fraud_type_counts']={str(k):int(v) for k,v in df['fraud_type'].value_counts(dropna=False).items()}
            if 'snapshot_id' in df: rec['snapshot_counts']={str(k):int(v) for k,v in df['snapshot_id'].value_counts().sort_index().items()}
        tables[name]=rec
    from graph.snapshots import chronological_split
    n=cfg['dataset']['n_snapshots']; tr,va,te=chronological_split(n,cfg['split']['train_ratio'],cfg['split']['val_ratio'])
    manifest={'config_file':args.config,'dataset_config':cfg['dataset'],'fraud_config':cfg['fraud'],'tables':tables,'split':{'train_snapshot_ids':tr,'validation_snapshot_ids':va,'test_snapshot_ids':te},'generated_from_repository':True}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    print(args.output)
if __name__=='__main__': main()
