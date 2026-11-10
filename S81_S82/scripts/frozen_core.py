from __future__ import annotations
import csv, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/"configs/protocol_v1.json").read_text(encoding="utf-8"))

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(path)
def write_csv(path,rows):
    if not rows:raise RuntimeError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

class Problem:
    def __init__(self,name):
        self.name=name;self.cfg=P["cases"][name];self.space=self.cfg["space"]
        with (ROOT/self.cfg["dataset"]).open(encoding="utf-8",newline="") as f:self.rows=list(csv.DictReader(f))
        if self.cfg.get("parameter_mode")=="first_90_after_id":
            self.parameter_names=list(self.rows[0])[1:91];scale=float(self.cfg["dataset_composition_scale"]);self.x_obs=np.asarray([[float(r[n] or 0)/scale for n in self.parameter_names] for r in self.rows],np.float64)
        else:self.parameter_names=self.cfg["parameters"];self.x_obs=np.asarray([[float(r[n]) for n in self.parameter_names] for r in self.rows],np.float64)
        self.property_names=self.cfg["properties"];self.y_obs=np.asarray([[float(r[n]) if r[n] else np.nan for n in self.property_names] for r in self.rows],np.float64)
        self.windows=np.asarray(self.cfg["windows"],float);self.bandwidth=float(self.cfg["rbf_bandwidth"]);self.n=len(self.parameter_names)
        if self.space=="simplex":
            if name=="industrial_pp_100":
                self.lower=np.zeros(self.n);self.upper=np.minimum(1,self.x_obs.max(0)*float(self.cfg["ingredient_upper_margin"]));lookup={};masks=[]
                for row in self.x_obs:
                    key=tuple(np.flatnonzero(row>1e-12))
                    if key not in lookup:lookup[key]=len(masks);masks.append(row>1e-12)
                self.structure_masks=np.asarray(masks,bool);self.required_masks=self.structure_masks.copy();self.structure_names=["support_"+"_".join(map(str,np.flatnonzero(m))) for m in self.structure_masks]
            else:
                ranges=np.asarray(self.cfg["ranges"],float);self.lower=ranges[:,0];self.upper=ranges[:,1];masks=[];required=[];names=[]
                for b in self.cfg["branches"]:
                    m=np.zeros(self.n,bool);m[b["allowed"]]=True;q=np.zeros(self.n,bool);q[b["required"]]=True;masks.append(m);required.append(q);names.append(b["name"])
                self.structure_masks=np.asarray(masks,bool);self.required_masks=np.asarray(required,bool);self.structure_names=names
            self.step=np.full(self.n,float(self.cfg["precision"]));self.scale=np.maximum(self.upper-self.lower,self.step)
        else:
            ranges=np.asarray(self.cfg["ranges"],float);self.lower=ranges[:,0];self.upper=ranges[:,1];self.step=np.asarray(self.cfg["precision"],float);self.scale=self.upper-self.lower;self.structure_masks=np.ones((1,self.n),bool);self.required_masks=np.zeros((1,self.n),bool);self.structure_names=["global_box"]
        self.structure_count=len(self.structure_masks)
        if self.space=="simplex" and any(self.upper[m].sum()<1-1e-9 for m in self.structure_masks):raise ValueError(f"Unclosable structure in {name}")

    def structure_features(self,structures,device):return torch.as_tensor(self.structure_masks[np.asarray(structures,int)],dtype=torch.float32,device=device)

    def relaxed(self,logits,structures):
        masks=self.structure_features(structures,logits.device)
        lo=torch.as_tensor(self.lower,dtype=logits.dtype,device=logits.device);hi=torch.as_tensor(self.upper,dtype=logits.dtype,device=logits.device);span=torch.as_tensor(self.scale,dtype=logits.dtype,device=logits.device)
        if self.space=="simplex":
            positive=F.softplus(logits)+1e-8;x=positive*masks;x=x/x.sum(1,keepdim=True).clamp_min(1e-12);range_loss=(torch.relu(lo-x)+torch.relu(x-hi))*masks
            req=torch.as_tensor(self.required_masks[np.asarray(structures,int)],dtype=logits.dtype,device=logits.device);required_loss=torch.relu(torch.as_tensor(self.step,dtype=logits.dtype,device=logits.device)-x)*req
            eng=required_loss.sum(1)
            for c in self.cfg.get("ratio_constraints",[]):
                ratio=x[:,c["numerator"]].sum(1)/x[:,c["denominator"]].sum(1).clamp_min(1e-8);eng=eng+torch.relu(float(c["min"])-ratio)+torch.relu(ratio-float(c["max"]))
            return x,x,range_loss.sum(1),eng
        norm=.5+.6*torch.tanh(logits);raw=lo+norm*span;x=torch.clamp(raw,lo,hi);range_loss=(torch.relu(lo-raw)+torch.relu(raw-hi))/span
        return raw,x,range_loss.sum(1),torch.zeros(len(x),dtype=x.dtype,device=x.device)

    def predict(self,x):
        xo=torch.as_tensor(self.x_obs,dtype=x.dtype,device=x.device);scale=torch.as_tensor(self.scale,dtype=x.dtype,device=x.device);d2=torch.sum(((x[:,None]-xo[None])/scale)**2,2);pred=[]
        for j in range(self.y_obs.shape[1]):
            valid=np.isfinite(self.y_obs[:,j]);y=torch.as_tensor(self.y_obs[valid,j],dtype=x.dtype,device=x.device);w=torch.softmax(-d2[:,torch.as_tensor(valid,device=x.device)]/(2*self.bandwidth**2+1e-12),1);pred.append(w@y)
        return torch.stack(pred,1),torch.sqrt(torch.clamp(d2.min(1).values,min=0)+1e-12)

    def quality(self,pred,x):
        win=torch.as_tensor(self.windows,dtype=pred.dtype,device=pred.device);q=torch.zeros(len(x),dtype=x.dtype,device=x.device)
        for term in self.cfg["objective_terms"]:
            if term["kind"]=="property_max":s=(pred[:,term["index"]]-win[term["index"],0])/(win[term["index"],1]-win[term["index"],0])
            elif term["kind"]=="property_min":s=(win[term["index"],1]-pred[:,term["index"]])/(win[term["index"],1]-win[term["index"],0])
            else:s=x[:,term["indices"]].sum(1)/float(term["normalizer"])
            q=q+float(term["weight"])*torch.clamp(s,0,1)
        return q

    def _bounded_simplex(self,values,structure):
        mask=self.structure_masks[int(structure)];required=self.required_masks[int(structure)];lo=np.where(required,self.step,0.0);hi=np.where(mask,self.upper,0.0);residual=1-lo.sum();caps=hi-lo
        if residual< -1e-10 or caps.sum()<residual-1e-10:raise ValueError((self.name,structure,"cannot close"))
        weights=np.maximum(np.asarray(values,float),1e-12)*mask;low,high=0.0,1.0
        while np.minimum(caps,weights*high).sum()<residual:high*=2
        for _ in range(70):
            mid=(low+high)/2
            if np.minimum(caps,weights*mid).sum()<residual:low=mid
            else:high=mid
        return lo+np.minimum(caps,weights*high)

    def _lattice_simplex(self,x,structure):
        step=float(self.step[0]);mask=self.structure_masks[int(structure)];required=self.required_masks[int(structure)];low=np.where(required,1,0);high=np.floor(np.where(mask,self.upper,0)/step+1e-9).astype(int);scaled=x/step;units=np.clip(np.floor(scaled+1e-12).astype(int),low,high);diff=round(1/step)-units.sum();frac=scaled-np.floor(scaled)
        while diff:
            ids=np.flatnonzero(mask&((units<high) if diff>0 else (units>low)));order=ids[np.argsort(-frac[ids] if diff>0 else frac[ids])]
            if not len(order):raise RuntimeError("lattice closure failed")
            for i in order:
                if not diff:break
                units[i]+=1 if diff>0 else -1;diff+=-1 if diff>0 else 1
        return units*step

    def _repair_ratios(self,x,structure):
        if not self.cfg.get("ratio_constraints"):return x
        y=x.copy();step=float(self.step[0]);mask=self.structure_masks[int(structure)];required=self.required_masks[int(structure)];lo=np.where(required,step,0.0)
        for c in self.cfg["ratio_constraints"]:
            num=np.asarray(c["numerator"],int);den=np.asarray(c["denominator"],int)
            for _ in range(int(round(1/step))+2):
                ratio=y[num].sum()/max(y[den].sum(),1e-12)
                if ratio<=float(c["max"])+1e-12:break
                donors=[i for i in num if y[i]>=lo[i]+step-1e-12];receivers=[i for i in np.flatnonzero(mask) if i not in num and y[i]+step<=self.upper[i]+1e-12]
                if not donors or not receivers:break
                d=max(donors,key=lambda i:y[i]-lo[i]);r=max(receivers,key=lambda i:self.upper[i]-y[i]);y[d]-=step;y[r]+=step
            for _ in range(int(round(1/step))+2):
                ratio=y[num].sum()/max(y[den].sum(),1e-12)
                if ratio>=float(c["min"])-1e-12:break
                donors=[i for i in np.flatnonzero(mask) if i not in num and y[i]>=lo[i]+step-1e-12];receivers=[i for i in num if mask[i] and y[i]+step<=self.upper[i]+1e-12]
                if not donors or not receivers:break
                d=max(donors,key=lambda i:y[i]-lo[i]);r=max(receivers,key=lambda i:self.upper[i]-y[i]);y[d]-=step;y[r]+=step
        return y

    def project(self,values,structure):
        if self.space=="simplex":return self._repair_ratios(self._lattice_simplex(self._bounded_simplex(values,structure),structure),structure)
        return np.clip(np.rint((np.asarray(values)-self.lower)/self.step)*self.step+self.lower,self.lower,self.upper)

    def _structural(self,x,structure):
        if self.space=="box":return bool(np.all(x>=self.lower-1e-8) and np.all(x<=self.upper+1e-8) and np.max(np.abs((x-self.lower)/self.step-np.rint((x-self.lower)/self.step)))<=1e-6)
        mask=self.structure_masks[int(structure)];required=self.required_masks[int(structure)];ok=abs(x.sum()-1)<=1e-8 and np.all(x[~mask]<=1e-10) and np.all(x>=self.lower-1e-9) and np.all(x<=self.upper+1e-9) and np.all(x[required]>=self.step[required]-1e-9) and np.max(np.abs(x/self.step-np.rint(x/self.step)))<=1e-6
        for c in self.cfg.get("ratio_constraints",[]):
            ratio=x[c["numerator"]].sum()/max(x[c["denominator"]].sum(),1e-12);ok&=ratio>=c["min"]-1e-9 and ratio<=c["max"]+1e-9
        return bool(ok)

    def audit(self,x,structures,device):
        x=np.asarray(x,float);structures=np.asarray(structures,int)
        if not len(x):return {"predicted_properties":np.empty((0,len(self.property_names))),"task_quality":np.empty(0),"nearest_database_distance":np.empty(0),"structural_ok":np.empty(0,bool),"window_ok":np.empty(0,bool),"feasible":np.empty(0,bool)}
        pred=[];quality=[];distance=[]
        with torch.no_grad():
            for start in range(0,len(x),4096):
                t=torch.as_tensor(x[start:start+4096],dtype=torch.float32,device=device);p,d=self.predict(t);pred.append(p.cpu().numpy());quality.append(self.quality(p,t).cpu().numpy());distance.append(d.cpu().numpy())
        pred=np.vstack(pred);quality=np.concatenate(quality);distance=np.concatenate(distance);struct=np.asarray([self._structural(v,s) for v,s in zip(x,structures)]);window=np.all((pred>=self.windows[:,0])&(pred<=self.windows[:,1]),1);return {"predicted_properties":pred,"task_quality":quality,"nearest_database_distance":distance,"structural_ok":struct,"window_ok":window,"feasible":struct&window}

class Generator(nn.Module):
    def __init__(self,n,latent,hidden):
        super().__init__();self.latent_dim=latent;layers=[];width=latent+n
        for h in hidden:layers += [nn.Linear(width,h),nn.LayerNorm(h),nn.LeakyReLU(.2)];width=h
        layers.append(nn.Linear(width,n));self.net=nn.Sequential(*layers)
    def forward(self,z,structure_features):return self.net(torch.cat([z,structure_features],1))

def train(problem,seed,out,device,pilot=False):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);spec=P["training"];model=Generator(problem.n,spec["latent_dim"],spec["hidden_dims"]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=spec["learning_rates"][0]);phases=spec["pilot_phases"] if pilot else spec["phases"];rows=[];started=time.time()
    for phase,steps in enumerate(phases,1):
        for g in opt.param_groups:g["lr"]=spec["learning_rates"][phase-1]
        for step in range(1,steps+1):
            structures=np.repeat(np.arange(problem.structure_count),spec["batch_per_structure"]);rng.shuffle(structures);features=problem.structure_features(structures,device);z=torch.randn(len(structures),model.latent_dim,device=device);logits=model(z,features);raw,x,range_each,engineering_each=problem.relaxed(logits,structures);pred,_=problem.predict(x);win=torch.as_tensor(problem.windows,dtype=x.dtype,device=device);width=win[:,1]-win[:,0];window_each=((torch.relu(win[:,0]-pred)+torch.relu(pred-win[:,1]))/width).sum(1);quality=problem.quality(pred,x);div=[]
            for s in range(problem.structure_count):
                ids=np.flatnonzero(structures==s)
                if len(ids)>1:div.append(torch.std((x[ids]-torch.as_tensor(problem.lower,dtype=x.dtype,device=device))/torch.as_tensor(problem.scale,dtype=x.dtype,device=device),0,unbiased=False).mean())
            diversity=torch.stack(div).mean() if div else torch.zeros((),device=device);range_loss=range_each.mean();engineering=engineering_each.mean();window=window_each.mean();loss=spec["range_weight"]*range_loss+spec["engineering_weight"]*engineering
            if phase>=2:loss=loss+spec["window_weight"]*window
            if phase>=3:loss=loss-spec["objective_weight"]*quality.mean()
            if phase>=4:loss=loss-spec["diversity_weight"]*diversity
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),spec["gradient_clip"]);opt.step();rows.append({"phase":phase,"step":step,"loss":float(loss.detach()),"range_loss":float(range_loss.detach()),"engineering_loss":float(engineering.detach()),"window_loss":float(window.detach()),"mean_quality":float(quality.mean().detach()),"diversity":float(diversity.detach())})
    out.mkdir(parents=True,exist_ok=False);torch.save(model.state_dict(),out/"model.pt");write_csv(out/"training_loss.csv",rows);meta={"case":problem.name,"seed":seed,"pilot":pilot,"seconds":time.time()-started,"device":str(device),"parameters":sum(p.numel() for p in model.parameters()),"numeric_database_anchor_used":False};atomic(out/"metadata.json",meta);return model,meta

def _structure_schedule(count,n):return np.arange(count,dtype=int)%n
def make_pool(problem,source,seed,count,out,device,model=None):
    rng=np.random.default_rng(seed);torch.manual_seed(seed);structures=_structure_schedule(count,problem.structure_count)
    if source=="G":
        raw_parts=[];eval_parts=[];model.eval()
        with torch.no_grad():
            for start in range(0,count,2048):
                s=structures[start:start+2048];feat=problem.structure_features(s,device);logits=model(torch.randn(len(s),model.latent_dim,device=device),feat);raw,x,_,_=problem.relaxed(logits,s);raw_parts.append(raw.cpu().numpy());eval_parts.append(x.cpu().numpy())
        raw=np.vstack(raw_parts);proposal=np.vstack(eval_parts);code=1
    else:
        if problem.space=="simplex":
            proposal=[]
            for s in structures:
                mask=problem.structure_masks[s];alpha=np.full(mask.sum(),10**rng.uniform(-.5,.5));v=np.zeros(problem.n);v[mask]=rng.dirichlet(alpha);proposal.append(v)
            proposal=np.asarray(proposal);raw=proposal.copy()
        else:proposal=rng.uniform(problem.lower,problem.upper,(count,problem.n));raw=proposal.copy()
        code=2
    final=np.asarray([problem.project(v,int(s)) for v,s in zip(proposal,structures)]);audit=problem.audit(final,structures,device);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/"candidates.npz",raw_output=np.asarray(raw),pre_projection=np.asarray(proposal),design=final,structure=structures,source=np.full(count,code,np.uint8),**audit);summary={"case":problem.name,"source":source,"attempts":count,"accepted":int(audit["feasible"].sum()),"accepted_rate":float(audit["feasible"].mean()),"successful_structures":int(np.unique(structures[audit["feasible"]]).size),"mean_quality_accepted":float(audit["task_quality"][audit["feasible"]].mean()) if audit["feasible"].any() else None,"numeric_database_anchor_used":False};atomic(out/"summary.json",summary);return {"design":final,"structure":structures,"source":np.full(count,code,np.uint8),**audit},summary

def balanced_half(pool,count,structures):
    chosen=[];per=count//len(structures);rem=count%len(structures)
    for j,s in enumerate(structures):chosen.extend(np.flatnonzero(pool["structure"]==s)[:per+(j<rem)])
    return np.asarray(chosen,int)
def combine_equal(g,r,total):
    common=np.intersect1d(np.unique(g["structure"]),np.unique(r["structure"]));ig=balanced_half(g,total//2,common);ir=balanced_half(r,total-total//2,common);return {k:np.concatenate([g[k][ig],r[k][ir]]) for k in g}
def normalized_distance(problem,a,b):return float(np.linalg.norm((a-b)/problem.scale))

def select(problem,pool,per_source,combined=False):
    chosen=[];sources=[1,2] if combined else [None]
    for s in np.unique(pool["structure"]):
        for source in sources:
            ids=np.flatnonzero(pool["feasible"]&(pool["structure"]==s)&((pool["source"]==source) if source else True))
            if not len(ids):continue
            q=pool["task_quality"][ids];eligible=ids[q>=np.quantile(q,P["seed_selection"]["quality_floor_quantile"])];sel=[int(ids[np.argmax(q)])]
            while len(sel)<per_source:
                rest=[int(i) for i in eligible if i not in sel]
                if not rest:break
                sel.append(max(rest,key=lambda i:(min(normalized_distance(problem,pool["design"][i],pool["design"][j]) for j in sel),pool["task_quality"][i])))
            chosen.extend(sel)
    return np.asarray(chosen,int)

def mutate(problem,x,structure,rng):
    y=x.copy()
    if problem.space=="simplex":
        active=np.flatnonzero(problem.structure_masks[structure]);moves=int(rng.integers(P["ga"]["mutation_moves_min"],P["ga"]["mutation_moves_max"]+1));step=float(problem.step[0])
        for _ in range(moves):
            i,j=rng.choice(active,2,replace=False);amount=int(rng.integers(1,P["ga"]["mutation_quanta_max"]+1))*step;y[i]+=amount;y[j]-=amount
    else:
        for _ in range(int(rng.integers(1,5))):
            i=int(rng.integers(problem.n));y[i]+=rng.choice([-1,1])*int(rng.integers(1,P["ga"]["mutation_quanta_max"]+1))*problem.step[i]
    return problem.project(y,structure)

def run_ga(problem,pool,selected,budget,seed,out,device):
    if not len(selected):raise RuntimeError(f"No feasible seeds for {problem.name}")
    rng=np.random.default_rng(seed);quotas=np.full(len(selected),budget//len(selected),int);quotas[:budget%len(selected)]+=1;accepted_x=[];accepted_s=[];accepted_source=[];line_rows=[];total=0
    for lineage,(idx,quota) in enumerate(zip(selected,quotas)):
        structure=int(pool["structure"][idx]);source=int(pool["source"][idx]);pop=[pool["design"][idx].copy()];pq=[float(pool["task_quality"][idx])];line_accepted=0;used=0
        while used<quota:
            size=min(64,quota-used);children=[]
            for _ in range(size):
                child=pop[int(rng.integers(len(pop)))].copy()
                if len(pop)>1 and rng.random()<P["ga"]["crossover_probability"]:
                    a=rng.uniform(.15,.85);child=a*child+(1-a)*pop[int(rng.integers(len(pop)))]
                if rng.random()<P["ga"]["mutation_probability"]:child=mutate(problem,child,structure,rng)
                else:child=problem.project(child,structure)
                children.append(child)
            children=np.asarray(children);structures=np.full(size,structure,int);audit=problem.audit(children,structures,device);used+=size;total+=size
            for i in np.flatnonzero(audit["feasible"]):accepted_x.append(children[i]);accepted_s.append(structure);accepted_source.append(source);pop.append(children[i]);pq.append(float(audit["task_quality"][i]));line_accepted+=1
            order=np.argsort(-np.asarray(pq))[:P["ga"]["elite_population"]];pop=[pop[i] for i in order];pq=[pq[i] for i in order]
        initial=float(pool["task_quality"][idx]);line_rows.append({"lineage":lineage,"structure":structure,"source":source,"budget":int(quota),"accepted":line_accepted,"initial_quality":initial,"best_quality":max(pq),"quality_gain":max(pq)-initial})
    x=np.asarray(accepted_x,float).reshape((-1,problem.n));structures=np.asarray(accepted_s,int);audit=problem.audit(x,structures,device);out.mkdir(parents=True,exist_ok=False);write_csv(out/"lineages.csv",line_rows);np.savez_compressed(out/"accepted.npz",design=x,structure=structures,source=np.asarray(accepted_source,np.uint8),**audit);receipt={"attempts":int(total),"accepted":len(x),"accepted_rate":len(x)/total,"lineages":len(selected),"improved_lineages":sum(r["quality_gain"]>0 for r in line_rows),"numeric_database_anchor_used":False};atomic(out/"receipt.json",receipt);return {"design":x,"structure":structures,"source":np.asarray(accepted_source,np.uint8),**audit},receipt

def build_library(problem,upstream,search,out):
    table={}
    feasible={k:v[upstream["feasible"]] for k,v in upstream.items() if isinstance(v,np.ndarray) and len(v)==len(upstream["feasible"])}
    for stage,part in enumerate([feasible,search]):
        for i,row in enumerate(part["design"]):
            key=(int(part["structure"][i]),*np.round(row,8));q=float(part["task_quality"][i]);source=int(part["source"][i])
            if key not in table:table[key]=[row,int(part["structure"][i]),q,source,stage]
            else:table[key][2]=max(table[key][2],q);table[key][3]|=source;table[key][4]=max(table[key][4],stage)
    vals=list(table.values());design=np.asarray([v[0] for v in vals]);structures=np.asarray([v[1] for v in vals],int);audit=problem.audit(design,structures,"cpu");lib={"design":design,"structure":structures,"task_quality":audit["task_quality"],"predicted_properties":audit["predicted_properties"],"nearest_database_distance":audit["nearest_database_distance"],"source_bits":np.asarray([v[3] for v in vals],np.uint8),"latest_stage":np.asarray([v[4] for v in vals],np.uint8)}
    per=P["portfolio"][problem.name+"_per_structure"];ids=[]
    for s in np.unique(structures):
        group=np.flatnonzero(structures==s);q=lib["task_quality"][group];eligible=group[q>=np.quantile(q,P["portfolio"]["quality_floor_quantile"])];sel=[int(group[np.argmax(q)])]
        while len(sel)<per:
            rest=[int(i) for i in eligible if i not in sel]
            if not rest:break
            sel.append(max(rest,key=lambda i:(min(normalized_distance(problem,design[i],design[j]) for j in sel),lib["task_quality"][i])))
        ids.extend(sel)
    ids=np.asarray(ids,int);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"library.npz",**lib);np.savez_compressed(out/"portfolio.npz",**{k:v[ids] for k,v in lib.items()});return lib,ids
