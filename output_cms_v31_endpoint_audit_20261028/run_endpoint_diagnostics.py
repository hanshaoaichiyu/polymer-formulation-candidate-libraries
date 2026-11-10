"""Diagnose saved coordinates only; never invokes a search or training entry."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,ast,csv,gzip,hashlib,io,json,math,platform,sys,zipfile
import numpy as np

R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write_csv(p,rows):
 rows=list(rows)
 if not rows:raise ValueError(p)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'wt',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def py(x):
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def quantiles(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 return {'n':len(x),'mean':float(x.mean()) if len(x) else None,'p25':float(np.quantile(x,.25)) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'p75':float(np.quantile(x,.75)) if len(x) else None,'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None}
def ranks(v):
 order=np.argsort(v,kind='stable');r=np.empty(len(v),float);s=np.asarray(v)[order];starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1];stops=np.r_[starts[1:],len(v)]
 for a,b in zip(starts,stops):r[order[a:b]]=(a+b-1)/2
 return r
def rho(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
 if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
 return float(np.corrcoef(ranks(a),ranks(b))[0,1])

class FrozenProblem:
 def __init__(self):
  cfg=json.loads((R/'inputs/generation_protocol_v1.json').read_text(encoding='utf-8'))['cases']['industrial_pp_100'];self.cfg=cfg
  with (R/'inputs/industrial_pp_100_recipes.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  self.parameter_names=list(rows[0])[1:91];self.property_names=cfg['properties'];self.x_obs=np.asarray([[float(r[n] or 0)/100 for n in self.parameter_names] for r in rows],float)
  self.y_obs=np.asarray([[float(r[n]) if r[n] else np.nan for n in self.property_names] for r in rows],float)
  self.windows=np.asarray(cfg['windows'],float);self.bandwidth=cfg['rbf_bandwidth'];self.n=90;self.space='simplex';self.lower=np.zeros(90);self.upper=np.minimum(1,self.x_obs.max(0)*1.05);self.step=np.full(90,cfg['precision']);self.scale=np.maximum(self.upper,self.step)
  masks=[];lookup={}
  for x in self.x_obs:
   k=tuple(np.flatnonzero(x>1e-12))
   if k not in lookup:lookup[k]=len(masks);masks.append(x>1e-12)
  self.structure_masks=np.asarray(masks);self.required_masks=self.structure_masks.copy();self.structure_count=len(masks)
 def predict64(self,X):
  X=np.asarray(X,float);out=np.empty((len(X),5))
  for st in range(0,len(X),512):
   xx=X[st:st+512];d2=np.sum(((xx[:,None,:]-self.x_obs[None,:,:])/self.scale)**2,axis=2)
   for j in range(5):
    valid=np.isfinite(self.y_obs[:,j]);logits=-d2[:,valid]/(2*self.bandwidth**2+1e-12);logits-=logits.max(1,keepdims=True);w=np.exp(logits);w/=w.sum(1,keepdims=True);out[st:st+len(xx),j]=w@self.y_obs[valid,j]
  return out
 def predict32(self,X):
  import torch
  xo=torch.as_tensor(self.x_obs,dtype=torch.float32);sc=torch.as_tensor(self.scale,dtype=torch.float32);out=[]
  with torch.no_grad():
   for st in range(0,len(X),4096):
    xx=torch.as_tensor(np.asarray(X[st:st+4096]),dtype=torch.float32);d2=torch.sum(((xx[:,None]-xo[None])/sc)**2,2);pred=[]
    for j in range(5):
     valid=np.isfinite(self.y_obs[:,j]);y=torch.as_tensor(self.y_obs[valid,j],dtype=xx.dtype);w=torch.softmax(-d2[:,torch.as_tensor(valid)]/(2*self.bandwidth**2+1e-12),1);pred.append(w@y)
    out.append(torch.stack(pred,1).numpy())
  return np.concatenate(out)
 def score(self,p):
  q=np.zeros(len(p))
  for t in self.cfg['objective_terms']:
   j=t['index'];lo,hi=self.windows[j];v=(p[:,j]-lo)/(hi-lo) if t['kind']=='property_max' else (hi-p[:,j])/(hi-lo);q+=t['weight']*np.clip(v,0,1)
  return q
 def windows_state(self,p):
  width=np.ptp(self.windows,axis=1);low=(p-self.windows[:,0])/width;high=(self.windows[:,1]-p)/width;m=np.minimum(low,high);pen=np.sum(np.maximum(-low,0)**2+np.maximum(-high,0)**2,axis=1)
  return {'per':(p>=self.windows[:,0])&(p<=self.windows[:,1]),'margin_per':m,'margin':m.min(1),'penalty':pen,'low_fail':p<self.windows[:,0],'high_fail':p>self.windows[:,1]}
 def composition(self,X,s,lattice):
  X=np.asarray(X,float);mask=self.structure_masks[np.asarray(s,int)];req=self.required_masks[np.asarray(s,int)]
  flags={'inactive_or_support_fail':np.any((X>1e-10)&~mask,axis=1),'nonnegative_fail':np.any(X<self.lower-1e-9,axis=1),'minimum_amount_fail':np.any((X<self.step-1e-9)&req,axis=1),'upper_bound_fail':np.any(X>self.upper+1e-9,axis=1),'closure_fail':np.abs(X.sum(1)-1)>1e-8}
  if lattice:flags['lattice_fail']=np.max(np.abs(X/self.step-np.rint(X/self.step)),axis=1)>1e-6
  flags['all_ok']=~np.logical_or.reduce(list(flags.values()));return flags

def attach_frozen_repair(problem):
 tree=ast.parse((R/'provenance/frozen_generation_core.py').read_text(encoding='utf-8'))
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Problem')
 wanted=['_bounded_simplex','_lattice_simplex','_repair_ratios','project','_structural']
 methods=[x for x in cls.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in wanted]
 assert len(methods)==len(wanted)
 module=ast.Module(body=[ast.ClassDef(name='RepairOnly',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(module);ns={'np':np};exec(compile(module,'verified_frozen_repair_methods','exec'),ns)
 for n in wanted:setattr(problem,n,getattr(ns['RepairOnly'],n).__get__(problem,FrozenProblem))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,default=R.parent/'output_cms_v25/research_assets/完整保留轨迹档案_待确认公开条款.zip');ap.add_argument('--output',type=Path,default=R/'results');ap.add_argument('--no-float32',action='store_true');a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 rules=json.loads((R/'FROZEN_DIAGNOSTIC_RULES.json').read_text(encoding='utf-8'));assert sha(R/'FROZEN_DIAGNOSTIC_RULES.json')==(R/'FROZEN_DIAGNOSTIC_RULES.sha256').read_text().strip()
 assert json.loads((R/'ARCHIVE_AVAILABILITY.json').read_text(encoding='utf-8'))['count_gate_passed']
 problem=FrozenProblem();attach_frozen_repair(problem);e=load(R/'inputs/endpoint_all_40000.npz');summary=load(R/'inputs/per_start_visit_summary_40000.npz');best=load(R/'inputs/best_feasible_per_successful_start.npz');n=len(e['design']);assert n==40000
 x=e['best_continuous'];post=e['design'];s=e['structure'];rep=e['replicate'];start=e['global_start_id'];k=problem.structure_masks.sum(1).astype(int);M=1-k*.0001;H=np.sum(problem.structure_masks*problem.upper,axis=1)-1
 compC=problem.composition(x,s,False);compD=problem.composition(post,s,True)
 # Reapply the exact frozen methods; read saved starts and validate their repaired step-zero identities.
 replay=np.empty_like(post);firststep=np.full(n,-1,int);laststep=np.full(n,-1,int);rep_F=np.full(n,np.nan);rep_cwin=np.zeros(n,bool);firststep_index=np.full(n,-1,int);rep_visit=np.full(n,-1,int);rep_step=np.full(n,-1,int);rep_x=np.full((n,90),np.nan);rep_x_raw=np.full((n,90),np.nan);rep_p_orig=np.full((n,5),np.nan);rep_q_orig=np.full(n,np.nan);rep_margin_orig=np.full(n,np.nan);step0_checks=[];archive_binding=[]
 best_lookup={(int(r),int(i)):j for j,(r,i) in enumerate(zip(best['replicate'],best['global_start_id']))}
 with zipfile.ZipFile(a.archive) as z:
  for rr in range(1,6):
   ii=np.flatnonzero(rep==rr)
   for j in ii:replay[j]=problem.project(x[j],int(s[j]))
   assert np.array_equal(np.rint(replay[ii]/.0001),np.rint(post[ii]/.0001)),('endpoint repair',rr)
   data=load(io.BytesIO(z.read(f'trajectory_archive_20261024/replicate_{rr}/trajectory_all_unique_audited_visits.npz')))
   st=load(R/f'inputs/starts_replicate_{rr}.npz');iz=np.flatnonzero(data['accepted_step']==0);iz=iz[np.argsort(data['global_start_id'][iz])]
   restored0=np.rint(data['design'][iz].astype(float)/.0001)*.0001
   st_repaired=np.asarray([problem.project(xx,int(ss)) for xx,ss in zip(st['design'],st['structure'])]);assert np.array_equal(np.rint(st_repaired/.0001),np.rint(restored0/.0001))
   step0_checks.append({'replicate':rr,'all_8000_stepzero_coordinates_match_repaired_original_starts':True,'max_stored_stepzero_float32_tail_fraction':float(np.max(np.abs(restored0-data['design'][iz])))})
   for j in ii:
    visit=np.flatnonzero(data['global_start_id']==start[j]);ok=visit[data['feasible'][visit]]
    laststep[j]=int(data['accepted_step'][visit].max())
    if not len(ok):continue
    first=int(ok[np.argmin(data['accepted_step'][ok])]);firststep[j]=int(data['accepted_step'][first]);firststep_index[j]=first
    b=best_lookup[(rr,int(start[j]))];idx=int(best['source_visit_index'][b]);assert data['feasible'][idx] and data['global_start_id'][idx]==start[j]
    for key in ['design','structure','accepted_step','task_quality','predicted_properties']:assert np.array_equal(data[key][idx],best[key][b])
    rep_visit[j]=idx;rep_step[j]=int(data['accepted_step'][idx]);rep_x_raw[j]=data['design'][idx];rep_x[j]=np.rint(rep_x_raw[j]/.0001)*.0001;rep_p_orig[j]=data['predicted_properties'][idx];rep_q_orig[j]=data['task_quality'][idx];rep_margin_orig[j]=data['margin'][idx];rep_F[j]=data['continuous_objective'][idx];rep_cwin[j]=data['continuous_window_pass'][idx]
   print(f'run {rr}: frozen repair and saved representative links verified',flush=True)
 ever=summary['any_feasible_visit'];epok=e['feasible'];omitted=~epok&ever;never=~ever
 group=np.where(epok,'endpoint_qualified',np.where(ever,'endpoint_failed_ever_feasible','never_feasible'))
 assert int(epok.sum())==10388 and int(omitted.sum())==12842 and int(never.sum())==16770 and np.array_equal(ever,rep_visit>=0)
 print('replaying frozen kernel in float64, without fitting',flush=True)
 pc=problem.predict64(x);pd=problem.predict64(post);pr=np.full((n,5),np.nan);pr[ever]=problem.predict64(rep_x[ever]);qc=problem.score(pc);qd=problem.score(pd);qr=np.full(n,np.nan);qr[ever]=problem.score(pr[ever]);wc=problem.windows_state(pc);wd=problem.windows_state(pd);wr=problem.windows_state(pr)
 use32=not a.no_float32;torch_info=None;p32=np.full((n,5),np.nan);r32=np.full((n,5),np.nan)
 if use32:
  try:
   import torch
   torch.set_num_threads(4);torch_info={'version':torch.__version__,'threads':torch.get_num_threads(),'historical_original_thread_count':'not saved'}
   p32=problem.predict32(post);r32[ever]=problem.predict32(rep_x[ever]);print('original float32 kernel formula replayed on endpoints and representatives',flush=True)
  except ImportError:use32=False;torch_info={'available':False}
 pre_orig=compC['all_ok']&e['best_continuous_window_ok'];post_orig=epok
 pre64=compC['all_ok']&wc['per'].all(1);post64=compD['all_ok']&wd['per'].all(1)
 def transition(pre,post):return np.where(pre,np.where(post,'pre_pass_post_pass','pre_pass_post_fail_eligibility_lost'),np.where(post,'pre_fail_post_pass_recovered','pre_fail_post_fail'))
 trans=transition(pre_orig,post_orig);trans64=transition(pre64,post64)
 failurefamily=np.where(e['structural_ok'],np.where(e['window_ok'],'qualified','window_only'),np.where(e['window_ok'],'composition_only','composition_and_window'))
 delta=post-x;shift=np.linalg.norm(delta/problem.scale,axis=1);maxshift=np.abs(delta).max(1)
 def half_ulp(p):
  p=np.asarray(p,np.float32);return .5*np.maximum(np.nextafter(p,np.float32(np.inf)).astype(float)-p.astype(float),p.astype(float)-np.nextafter(p,np.float32(-np.inf)).astype(float))
 ccache=e['best_continuous_predicted_properties'];dcache=e['predicted_properties'];ce=np.maximum(np.abs(pc-ccache),half_ulp(ccache));de=np.maximum(np.abs(pd-dcache),half_ulp(dcache));width=np.ptp(problem.windows,axis=1)
 c_boundary=(np.minimum(np.abs(pc-problem.windows[:,0]),np.abs(pc-problem.windows[:,1]))<=ce).any(1)
 d_boundary=(np.minimum(np.abs(pd-problem.windows[:,0]),np.abs(pd-problem.windows[:,1]))<=de).any(1)
 c_mismatch=np.any(wc['per']!=e['best_continuous_property_window_ok'],axis=1);d_mismatch=np.any(wd['per']!=e['property_window_ok'],axis=1)
 stable=~(c_mismatch|d_mismatch|c_boundary|d_boundary)
 conditions={'primary_endpoint_qualified':epok,'primary_omitted':omitted,'primary_never_feasible':never,'historical_continuous_window_pass':e['best_continuous_window_ok'],'continuous_composition_pass_replay':compC['all_ok'],'historical_endpoint_composition_pass':e['structural_ok'],'historical_endpoint_window_pass':e['window_ok'],'pre_eligible_historical_window_plus_composition_replay':pre_orig,'post_eligible_historical':epok,'pre_eligible_float64_replay':pre64,'post_eligible_float64_replay':post64,'continuous_float64_window_mismatch':c_mismatch,'post_float64_window_mismatch':d_mismatch,'continuous_observed_precision_boundary':c_boundary,'post_observed_precision_boundary':d_boundary,'precision_stable_diagnostic':stable}
 storage_tail=np.full(n,np.nan);storage_tail[ever]=np.max(np.abs(rep_x[ever]-rep_x_raw[ever]),axis=1)
 feature={'active_count':k[s],'free_dimension':k[s]-1,'remaining_mass_fraction':M[s],'capacity_surplus_fraction':H[s],'repair_scaled_L2':shift,'repair_max_abs_fraction':maxshift,'repair_max_abs_percentage_points':maxshift*100,'continuous_Q_replay64':qc,'continuous_penalty_replay64':wc['penalty'],'continuous_objective_replay64':qc-10*wc['penalty'],'post_Q_replay64':qd,'post_window_penalty_replay64':wd['penalty'],'continuous_margin_replay64':wc['margin'],'post_margin_replay64':wd['margin'],'continuous_Q_historical':e['best_continuous_task_quality'],'continuous_penalty_historical':e['best_continuous_window_penalty'],'continuous_objective_historical':e['best_continuous_objective'],'post_Q_historical':e['task_quality'],'post_margin_historical':e['margin'],'representative_Q_historical':rep_q_orig,'representative_Q_replay64':qr,'representative_generating_continuous_objective_historical':rep_F,'representative_generating_continuous_window_pass_historical':rep_cwin,'endpoint_minus_representative_generating_continuous_objective':e['best_continuous_objective']-rep_F,'first_feasible_visit_step':firststep,'representative_accepted_step':rep_step,'last_retained_visit_step_not_endpoint_step':laststep,'representative_source_visit_index':rep_visit,'representative_storage_tail_max_abs_fraction':storage_tail}
 rows=[]
 for i in range(n):
  r={'replicate':int(rep[i]),'global_start_id':int(start[i]),'template_zero_based':int(s[i]),'original_group':str(group[i]),'original_failure_family':str(failurefamily[i]),'primary_transition':str(trans[i]),'float64_replay_transition':str(trans64[i]),'endpoint_accepted_step':None,'endpoint_step_status':'not archived','representative_continuous_Q':None,'representative_continuous_penalty':None,'representative_repair_displacement':None,'representative_step_minus_endpoint_step':None}
  r.update({name:py(v[i]) for name,v in conditions.items()});r.update({name:py(v[i]) for name,v in feature.items()})
  if not ever[i]:
   for name in ['representative_generating_continuous_window_pass_historical','representative_accepted_step','first_feasible_visit_step','representative_source_visit_index']:r[name]=None
  for stage,cf in [('continuous',compC),('post',compD)]:
   for name,v in cf.items():r[f'{stage}_{name}_diagnostic']=bool(v[i])
  for j,name in enumerate(problem.property_names):
   r[f'continuous_{name}_original_window_fail']=not bool(e['best_continuous_property_window_ok'][i,j]);r[f'post_{name}_original_window_fail']=not bool(e['property_window_ok'][i,j])
   for stage,ws,pp in [('continuous',wc,pc),('post',wd,pd)]:
    r[f'{stage}_{name}_margin_replay64']=float(ws['margin_per'][i,j]);r[f'{stage}_{name}_low_fail_replay64']=bool(ws['low_fail'][i,j]);r[f'{stage}_{name}_high_fail_replay64']=bool(ws['high_fail'][i,j]);r[f'{stage}_{name}_prediction_replay64']=float(pp[i,j])
   if use32:r[f'post_{name}_prediction_replay32']=float(p32[i,j])
  rows.append(r)
 write_csv(out/'per_start_diagnostic_ledger.csv.gz',rows);write_csv(out/'omitted_start_paired_ledger.csv.gz',[rows[i] for i in np.flatnonzero(omitted)])
 # Run/group failure and transition summaries keep their exact historical denominator.
 failure_rows=[];run_rows=[];process_rows=[]
 transitions=['pre_pass_post_pass','pre_pass_post_fail_eligibility_lost','pre_fail_post_pass_recovered','pre_fail_post_fail']
 for rr in range(1,6):
  ii=rep==rr;run_rows.append({'replicate':rr,'starts':int(ii.sum()),'endpoint_qualified':int((epok&ii).sum()),'omitted':int((omitted&ii).sum()),'never_feasible':int((never&ii).sum()),'any_feasible':int((ever&ii).sum()),'omission_rate_all_starts':float(omitted[ii].mean()),'omission_rate_among_successful_starts':float(omitted[ii].sum()/ever[ii].sum()),'omitted_precision_stable':int((omitted&ii&stable).sum())})
  for gg in ['all','endpoint_qualified','endpoint_failed_ever_feasible','never_feasible']:
   jj=ii if gg=='all' else ii&(group==gg);den=int(jj.sum());r={'replicate':rr,'group':gg,'denominator':den}
   for f in ['qualified','composition_only','window_only','composition_and_window']:r['original_post_'+f]=int((jj&(failurefamily==f)).sum())
   for t in transitions:r[t]=int((jj&(trans==t)).sum());r['float64_'+t]=int((jj&(trans64==t)).sum());r['precision_stable_'+t]=int((jj&stable&(trans==t)).sum())
   for j,name in enumerate(problem.property_names):r['original_continuous_fail_'+name]=int((jj&~e['best_continuous_property_window_ok'][:,j]).sum());r['original_post_fail_'+name]=int((jj&~e['property_window_ok'][:,j]).sum())
   for stage,cf in [('continuous',compC),('post',compD)]:
    for name,v in cf.items():r[stage+'_'+name+'_diagnostic']=int((jj&v).sum())
   r['continuous_float64_window_disagree']=int((jj&c_mismatch).sum());r['post_float64_window_disagree']=int((jj&d_mismatch).sum());failure_rows.append(r)
   if gg!='all':
    for name in ['repair_scaled_L2','repair_max_abs_percentage_points','continuous_margin_replay64','post_margin_replay64','continuous_Q_historical','continuous_penalty_historical','post_Q_historical','endpoint_minus_representative_generating_continuous_objective','representative_Q_historical','representative_accepted_step','first_feasible_visit_step']:
     process_rows.append({'replicate':rr,'group':gg,'feature':name,**quantiles(feature[name][jj])})
 write_csv(out/'run_counts_and_two_denominators.csv',run_rows);write_csv(out/'failure_reasons_and_transitions_by_run_group.csv',failure_rows);write_csv(out/'process_feature_summaries_by_run_group.csv',process_rows)
 support_rows=[]
 for rr in range(1,6):
  for ss in range(80):
   ii=(rep==rr)&(s==ss);den=int(ii.sum());o=int((ii&omitted).sum());v=int((ii&ever).sum());row={'replicate':rr,'template_zero_based':ss,'active_count':int(k[ss]),'free_dimension':int(k[ss]-1),'remaining_mass_fraction':float(M[ss]),'capacity_surplus_fraction':float(H[ss]),'starts':den,'endpoint_qualified':int((ii&epok).sum()),'omitted':o,'never_feasible':int((ii&never).sum()),'any_feasible':v,'omission_rate_all_starts':o/den,'omission_rate_among_successful_starts':o/v if v else None,'omitted_pre_continuous_eligible':int((ii&omitted&pre_orig).sum()),'omitted_pre_continuous_ineligible':int((ii&omitted&~pre_orig).sum()),'omitted_precision_stable':int((ii&omitted&stable).sum()),'repair_scaled_L2_mean':float(shift[ii].mean()),'repair_max_abs_pp_mean':float(maxshift[ii].mean()*100),'continuous_margin_mean':float(wc['margin'][ii].mean()),'post_margin_mean':float(wd['margin'][ii].mean())}
   for j,name in enumerate(problem.property_names):row['omitted_original_post_fail_'+name]=int((ii&omitted&~e['property_window_ok'][:,j]).sum())
   support_rows.append(row)
 write_csv(out/'support_by_run.csv',support_rows)
 support_mean=[]
 for ss in range(80):
  ssrows=[r for r in support_rows if r['template_zero_based']==ss];n_ever=sum(r['any_feasible'] for r in ssrows);n_omit=sum(r['omitted'] for r in ssrows);valid=[r['omission_rate_among_successful_starts'] for r in ssrows if r['omission_rate_among_successful_starts'] is not None]
  support_mean.append({'template_zero_based':ss,'active_count':int(k[ss]),'free_dimension':int(k[ss]-1),'remaining_mass_fraction':float(M[ss]),'capacity_surplus_fraction':float(H[ss]),'starts_all_runs':500,'omitted_all_runs':n_omit,'feasible_starts_all_runs':n_ever,'omission_rate_all_starts':n_omit/500,'omission_rate_among_successful_starts':float(np.mean(valid)) if valid else None,'conditional_pooled_within_support':n_omit/n_ever if n_ever else None,'conditional_run_count':len(valid),'omission_all_start_rate_run_SD':float(np.std([r['omission_rate_all_starts'] for r in ssrows],ddof=1))})
 write_csv(out/'support_five_run_descriptions.csv',support_mean)
 dim_rows=[];weight_rows=[]
 for rr in range(1,6):
  rrrows=[r for r in support_rows if r['replicate']==rr]
  for dd in sorted(set(k-1)):
   xrows=[r for r in rrrows if r['free_dimension']==dd];ns=sum(r['starts'] for r in xrows);no=sum(r['omitted'] for r in xrows);nv=sum(r['any_feasible'] for r in xrows);vals=[r['omission_rate_among_successful_starts'] for r in xrows if r['omission_rate_among_successful_starts'] is not None]
   dim_rows.append({'replicate':rr,'free_dimension':int(dd),'support_count':len(xrows),'starts':ns,'omitted':no,'any_feasible':nv,'omission_rate_all_starts':no/ns,'omission_rate_among_successful_starts_pooled':no/nv if nv else None,'conditional_support_equal':float(np.mean(vals)) if vals else None,'conditional_support_count':len(vals)})
  vv=[r['omission_rate_among_successful_starts'] for r in rrrows if r['omission_rate_among_successful_starts'] is not None]
  weight_rows.append({'replicate':rr,'all_supports':80,'conditional_support_count':len(vv),'omission_all_starts_support_equal':float(np.mean([r['omission_rate_all_starts'] for r in rrrows])),'omission_successful_starts_support_equal':float(np.mean(vv)),'omission_successful_starts_start_weighted':float(sum(r['omitted'] for r in rrrows)/sum(r['any_feasible'] for r in rrrows))})
 write_csv(out/'dimension_by_run.csv',dim_rows);write_csv(out/'two_weights_by_run.csv',weight_rows)
 correlation=[]
 for label,data in [(str(rr),[r for r in support_rows if r['replicate']==rr]) for rr in range(1,6)]+[('five_run_support_means',support_mean)]:
  for subset in ['all_supports','non_singleton_supports']:
   ddd=[r for r in data if subset=='all_supports' or r['free_dimension']>0]
   for feature_name in ['active_count','remaining_mass_fraction','capacity_surplus_fraction']:
    for target in ['omission_rate_all_starts','omission_rate_among_successful_starts']:
     valid=[r for r in ddd if r[target] is not None];xx=np.asarray([r[feature_name] for r in valid]);yy=np.asarray([r[target] for r in valid]);full=rho(xx,yy);leave=[rho(np.delete(xx,i),np.delete(yy,i)) for i in range(len(xx))];leave=[v for v in leave if v is not None]
     correlation.append({'run_or_population':label,'support_subset':subset,'feature':feature_name,'outcome':target,'supports':len(valid),'descriptive_Spearman':full,'leave_one_support_out_min':min(leave) if leave else None,'leave_one_support_out_max':max(leave) if leave else None,'leave_one_out_positive_count':sum(v>0 for v in leave),'leave_one_out_negative_count':sum(v<0 for v in leave),'p_value':'not tested'})
 write_csv(out/'limited_descriptive_support_correlations.csv',correlation)
 sorted_omit=sorted(support_mean,key=lambda r:(-r['omitted_all_runs'],r['template_zero_based']));concentration={'total_omitted':int(omitted.sum()),'largest_support_fraction':sorted_omit[0]['omitted_all_runs']/int(omitted.sum()),'five_largest_support_fraction':sum(r['omitted_all_runs'] for r in sorted_omit[:5])/int(omitted.sum()),'top_five_templates':[r['template_zero_based'] for r in sorted_omit[:5]],'omission_present_supports':sum(r['omitted_all_runs']>0 for r in support_mean),'singleton_templates':np.flatnonzero(k==1).tolist()}
 # Precision differences are explicit and cannot change the primary outcome groups.
 precision=[];precision_record_rows=[]
 for rr in range(1,6):
  ii=rep==rr;row={'replicate':rr,'endpoint_continuous_window_per_response_mismatch64':int((ii&c_mismatch).sum()),'endpoint_post_window_per_response_mismatch64':int((ii&d_mismatch).sum()),'endpoint_post_overall_feasibility_mismatch64':int((ii&(post64!=epok)).sum()),'endpoint_post_composition_mismatch64':int((ii&(compD['all_ok']!=e['structural_ok'])).sum()),'continuous_cached32_vs_replay64_max_abs':float(np.abs(pc[ii]-ccache[ii]).max()),'post_cached32_vs_replay64_max_abs':float(np.abs(pd[ii]-dcache[ii]).max()),'continuous_boundary_envelope_n':int((ii&c_boundary).sum()),'post_boundary_envelope_n':int((ii&d_boundary).sum()),'endpoint_repair_max_abs_difference_fraction':float(np.max(np.abs(replay[ii]-post[ii]))),'representative_storage_tail_max_abs_fraction':float(np.nanmax(np.abs(rep_x[ii]-rep_x_raw[ii]))),'representative_replay64_window_mismatch_original_pass':int((ii&ever&~wr['per'].all(1)).sum())}
  if use32:
   w32=problem.windows_state(p32);rw32=problem.windows_state(r32);row['endpoint_original_vs_float32_replay_window_mismatch']=int((ii&(w32['per'].all(1)!=e['window_ok'])).sum());row['endpoint_original_vs_float32_replay_max_abs']=float(np.max(np.abs(p32[ii]-dcache[ii])));row['representative_float32_replay_window_fail_n']=int((ii&ever&~rw32['per'].all(1)).sum())
  precision.append(row)
 for i in np.flatnonzero(c_mismatch|d_mismatch):
  for stage,pp,cache,oldflag,ws in [('continuous',pc,ccache,e['best_continuous_property_window_ok'],wc),('post',pd,dcache,e['property_window_ok'],wd)]:
   for j,name in enumerate(problem.property_names):
    if ws['per'][i,j]!=oldflag[i,j]:precision_record_rows.append({'replicate':int(rep[i]),'global_start_id':int(start[i]),'template_zero_based':int(s[i]),'primary_group':str(group[i]),'stage':stage,'response':name,'historical_property_window_ok':bool(oldflag[i,j]),'replay64_property_window_ok':bool(ws['per'][i,j]),'cached_prediction_float32':float(cache[i,j]),'replayed_prediction_float64':float(pp[i,j]),'lower_window':float(problem.windows[j,0]),'upper_window':float(problem.windows[j,1]),'signed_normalized_margin64':float(ws['margin_per'][i,j]),'abs_replay_minus_cached':float(abs(pp[i,j]-cache[i,j])),'reason':'same frozen formula with numerical precision/storage differences; not a physical mechanism'})
 for i in np.flatnonzero(ever&~wr['per'].all(1)):
  for j,name in enumerate(problem.property_names):
   if not wr['per'][i,j]:precision_record_rows.append({'replicate':int(rep[i]),'global_start_id':int(start[i]),'template_zero_based':int(s[i]),'primary_group':str(group[i]),'stage':'restored_best_feasible_representative','response':name,'historical_property_window_ok':True,'replay64_property_window_ok':False,'cached_prediction_float32':float(rep_p_orig[i,j]),'replayed_prediction_float64':float(pr[i,j]),'lower_window':float(problem.windows[j,0]),'upper_window':float(problem.windows[j,1]),'signed_normalized_margin64':float(wr['margin_per'][i,j]),'abs_replay_minus_cached':float(abs(pr[i,j]-rep_p_orig[i,j])),'reason':'original float32 accepted; float64 diagnostic on restored lattice crosses a numerical boundary. Preserve original feasible label.'})
 write_csv(out/'precision_difference_by_run.csv',precision)
 if precision_record_rows:write_csv(out/'precision_difference_records.csv',precision_record_rows)
 else:(out/'precision_difference_records.csv').write_text('replicate,global_start_id,stage,response\n',encoding='utf-8-sig')
 np.savez_compressed(out/'diagnostic_coordinates_predictions.npz',replicate=rep,global_start_id=start,structure=s,continuous_endpoint=x,repaired_endpoint_saved=post,repaired_endpoint_replayed=replay,continuous_prediction_replay64=pc,repaired_prediction_replay64=pd,representative_restored_lattice=rep_x,representative_prediction_replay64=pr,representative_source_visit=rep_visit,representative_continuous_objective_saved=rep_F,primary_group_code=np.where(epok,0,np.where(ever,1,2)),pre_comp_pass=compC['all_ok'],pre_window_historical=e['best_continuous_window_ok'],post_original_feasible=epok,precision_stable=stable,scale=problem.scale,upper=problem.upper,lower_active=np.full(90,.0001),windows=problem.windows)
 summary_out={'computed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'protocol_sha256':sha(R/'FROZEN_DIAGNOSTIC_RULES.json'),'original_counts':{'endpoint_qualified':int(epok.sum()),'omitted':int(omitted.sum()),'never_feasible':int(never.sum())},'run_counts':run_rows,'omitted_primary_transitions':{t:int((omitted&(trans==t)).sum()) for t in transitions},'omitted_precision_stable_transitions':{t:int((omitted&stable&(trans==t)).sum()) for t in transitions},'omitted_original_failure_families':{f:int((omitted&(failurefamily==f)).sum()) for f in ['composition_only','window_only','composition_and_window']},'omitted_original_response_failures':{name:int((omitted&~e['property_window_ok'][:,j]).sum()) for j,name in enumerate(problem.property_names)},'all_continuous_composition_failures':int((~compC['all_ok']).sum()),'all_post_composition_failures_historical':int((~e['structural_ok']).sum()),'paired_omitted_summaries':{name:quantiles(feature[name][omitted]) for name in ['endpoint_minus_representative_generating_continuous_objective','continuous_Q_historical','continuous_penalty_historical','post_Q_historical','representative_Q_historical','repair_scaled_L2','repair_max_abs_percentage_points','representative_accepted_step','first_feasible_visit_step']},'concentration':concentration,'precision':precision,'exact_endpoint_step_available':False,'representative_continuous_coordinate_available':False,'source_generation_or_optimization_rerun':False,'float32_replay':torch_info,'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},'repair_identity_max_abs_difference':float(np.max(np.abs(replay-post))),'count_references':'40000 original starts are dependent algorithm records, not independent materials. Fixed support characteristics have 80 units, not 400 independent units.'}
 dump(out/'SUMMARY.json',summary_out);dump(out/'STEP_ZERO_IDENTITY_CHECKS.json',step0_checks);dump(out/'EXECUTION.json',{'command':sys.argv,'script_sha256':sha(Path(__file__)),'frozen_rules_sha256':sha(R/'FROZEN_DIAGNOSTIC_RULES.json'),'archive_sha256':sha(a.archive),'prediction_replay_scope':'40000 continuous endpoints + 40000 repaired endpoints + 23230 restored best representatives; not all 3216626 visits','no_optimization_called':True,'no_training_called':True})
 print(json.dumps({k:summary_out[k] for k in ['original_counts','omitted_primary_transitions','omitted_original_failure_families','omitted_original_response_failures','paired_omitted_summaries','precision']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
