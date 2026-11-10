from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MONOTONE=ROOT
V14=ROOT
V11=ROOT
sys.path.insert(0,str(ROOT/"src"/"generation"))
from core import Problem as _GenerationProblem  # noqa: E402

P=json.loads((ROOT/"configs"/"cms_v15_protocol_v1.json").read_text(encoding="utf-8"))


class Problem(_GenerationProblem):
    def audit(self,x,structures,device="cpu"):
        return super().audit(x,structures,device)


def load_npz(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(path)


def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise RuntimeError(path)
    with path.open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))


def margin(problem,prediction):
    width=problem.windows[:,1]-problem.windows[:,0]
    return np.min(np.minimum((prediction-problem.windows[:,0])/width,(problem.windows[:,1]-prediction)/width),axis=1)
