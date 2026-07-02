import sys, os, random
sys.path.insert(0,'stage2')
from huggingface_hub import HfApi, hf_hub_download
from amass_to_features import convert_npy
import numpy as np
api=HfApi(); files=api.list_repo_files('fleaven/Retargeted_AMASS_for_robotics', repo_type='dataset')
g1=[f for f in files if f.startswith('g1/') and f.endswith('.npy')]
random.seed(7); random.shuffle(g1)           # broad random sample across ALL subsets for volume+diversity
out='stage2/out/amass_feats'; os.makedirs(out,exist_ok=True)
have=set(os.listdir(out)); n=len(have); target_frames=720000; frames=0   # ~10h @20fps
# count existing frames
for f in have:
    try: frames+=np.load(os.path.join(out,f))['motion'].shape[0]
    except: pass
print(f'start: {n} clips, {frames} frames', flush=True)
for f in g1:
    if frames>=target_frames: break
    name='amass_'+f.replace('g1/','').replace('/','_').replace('.npy','')[:70]+'.npz'
    if name in have: continue
    try:
        lp=hf_hub_download('fleaven/Retargeted_AMASS_for_robotics', f, repo_type='dataset', local_dir='/tmp/amass_dl')
        ft=convert_npy(lp)
        if ft is None or len(ft)<8: os.remove(lp); continue
        np.savez(os.path.join(out,name), motion=ft); frames+=len(ft); n+=1; os.remove(lp)
        if n%100==0: print(f'  {n} clips, {frames} frames ({frames/72000:.1f}h)', flush=True)
    except Exception: continue
print(f'AMASS10H DONE: {n} clips, {frames} frames ({frames/72000:.1f}h) -> {out}', flush=True)
