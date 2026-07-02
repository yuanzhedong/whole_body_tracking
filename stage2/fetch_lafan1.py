"""Download + convert FULL LAFAN1 retargeted to G1 (lvhaidong/LAFAN1_Retargeting_Dataset, g1/*.csv,
79 clips = all subjects). Same 36-col format as AMASS; joints verified EXACT vs our existing clips.
Root uses direct (raw) convention — consistent with the AMASS converter. Output 41-D features."""
import sys, os
sys.path.insert(0,'stage2')
from huggingface_hub import HfApi, hf_hub_download
from amass_to_features import PERM
from export_g1_motion import build_features
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
def convert_csv(path, src_fps=30, target_fps=20):
    d=np.loadtxt(path,delimiter=',').astype(np.float64)
    if d.ndim<2 or d.shape[1]<36 or len(d)<4: return None
    pos=d[:,0:3]; quat=d[:,3:7]; joints=d[:,7:36][:,PERM]
    N=len(d); T=max(2,round(N/src_fps*target_fps)); ti=np.linspace(0,1,N); to=np.linspace(0,1,T)
    pr=np.stack([np.interp(to,ti,pos[:,c]) for c in range(3)],1)
    jr=np.stack([np.interp(to,ti,joints[:,c]) for c in range(29)],1)
    qr=Slerp(ti,Rotation.from_quat(quat))(to).as_quat()
    f=build_features(pr,qr,jr,dt=1.0/target_fps,to_yup=True)
    return f.astype(np.float32) if np.isfinite(f).all() else None
api=HfApi(); files=api.list_repo_files('lvhaidong/LAFAN1_Retargeting_Dataset',repo_type='dataset')
csvs=[f for f in files if f.startswith('g1/') and f.endswith('.csv')]
out='stage2/out/lafan1_feats'; os.makedirs(out,exist_ok=True); n=0; fr=0
print(f'{len(csvs)} LAFAN1 g1 clips', flush=True)
for f in csvs:
    try:
        lp=hf_hub_download('lvhaidong/LAFAN1_Retargeting_Dataset',f,repo_type='dataset',local_dir='/tmp/lafan1_dl')
        ft=convert_csv(lp)
        if ft is None: continue
        np.savez(os.path.join(out,'lafan1_'+os.path.basename(f).replace('.csv','')+'.npz'),motion=ft); n+=1; fr+=len(ft)
    except Exception: continue
print(f'LAFAN1 DONE: {n} clips, {fr} frames ({fr/72000:.2f}h) -> {out}', flush=True)
