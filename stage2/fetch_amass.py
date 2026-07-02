import sys, os, random
sys.path.insert(0,'stage2')
from huggingface_hub import HfApi, hf_hub_download
from amass_to_features import convert_npy
import numpy as np
api=HfApi(); files=api.list_repo_files('fleaven/Retargeted_AMASS_for_robotics', repo_type='dataset')
g1=[f for f in files if f.startswith('g1/') and f.endswith('.npy')]
# dynamic/relevant subsets for our failing motions (sports, dynamic locomotion, floor/recovery, dance)
want=['ACCAD','BMLhandball','MOYO','DanceDB','CMU','KIT','SFU','EKUT']
sel=[f for f in g1 if any(s in f for s in want)]
random.seed(0); random.shuffle(sel); sel=sel[:600]   # cap for a first scale-up
out='stage2/out/amass_feats'; os.makedirs(out,exist_ok=True)
n=0
for f in sel:
    try:
        lp=hf_hub_download('fleaven/Retargeted_AMASS_for_robotics', f, repo_type='dataset', local_dir='/tmp/amass_dl')
        ft=convert_npy(lp)
        if ft is None or len(ft)<8: continue
        name='amass_'+f.replace('g1/','').replace('/','_').replace('.npy','')[:70]
        np.savez(os.path.join(out,name+'.npz'), motion=ft); n+=1
        if n%50==0: print(f'  converted {n}...', flush=True)
        os.remove(lp)
    except Exception as e:
        continue
print(f'AMASS DONE: converted {n} clips -> {out}', flush=True)
