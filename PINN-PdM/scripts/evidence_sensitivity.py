#!/usr/bin/env python3
"""Training-free sensitivity of the spectral prior (Review-03, P2).

For every band width sigma and harmonic count H, measures how often the
physical evidence alone (argmax of the mask-weighted spectrum over fault
types) identifies the true fault type of each faulty window. No model is
trained. Writes outputs/review03/evidence_sensitivity.json.

    python scripts/evidence_sensitivity.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os

import numpy as np
import torch

from src.utils.config import REPO_ROOT
from src.review03.data_real import *
from src.review03.spectral import build_masks, fault_evidence, valid_mask
def acc(arr, freqs, ratios, view, sigma, H):
    S=torch.tensor(getattr(arr,view)); y=torch.tensor(arr.y); sh=torch.tensor(arr.shaft_hz)
    m=build_masks(freqs, ratios, sh, H, sigma, "sum"); ev=fault_evidence(S,m,valid_mask(ratios,"cpu"))
    pred=ev[:,1:].argmax(1)+1; f=y>0
    return float((pred[f]==y[f]).float().mean()), {int(c):float((pred[y==c]==c).float().mean()) for c in y[f].unique()}
os.chdir(REPO_ROOT)
out={}
recs=load_cwru('data/raw/cwru'); arr,freqs=build_arrays(cwru_protocol(recs,'cross_load'),2048,1024)
r=class_ratios(SKF_6205,CWRU_CLASSES)
for split in ['train','test']:
  for view in ['env_spec','raw_spec']:
    for s in [2.,5.,10.,20.]:
      for H in [1,3,5]:
        out[f"cwru_{split}_{view}_s{s:g}_H{H}"]=acc(arr[split],freqs,r,view,s,H)
m=load_mfpt('data/raw/mfpt'); a2,f2=build_arrays(mfpt_protocol(m),2048,1024); r2=class_ratios(MFPT_BEARING,MFPT_CLASSES)
for split in ['train','test']:
  for view in ['env_spec','raw_spec']:
    out[f"mfpt_{split}_{view}_s5_H3"]=acc(a2[split],f2,r2,view,5.,3)
json.dump(out,open('outputs/review03/evidence_sensitivity.json','w'),indent=1)
for k,v in out.items(): print(k, round(v[0]*100,1), {c:round(x*100,1) for c,x in v[1].items()})
