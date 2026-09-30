import json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

FEATURES=['satisfaction_score','total_spent','support_tickets','last_3_month_purchase_freq','avg_order_value','total_visits','email_open_rate','email_click_rate','avg_session_time','pages_per_session']
RS=42

def ece(y,p,n=10):
    y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,n+1); ids=np.digitize(p,edges[1:-1]); total=0; rows=[]
    for b in range(n):
        m=ids==b
        if m.any():
            pm=float(p[m].mean()); om=float(y[m].mean()); total += float(m.mean())*abs(pm-om); rows.append({'bin':b+1,'n':int(m.sum()),'prob_predicha_media':pm,'tasa_observada':om})
    return float(total),pd.DataFrame(rows)

def thr_metrics(y,p,t):
    pred=np.asarray(p)>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'threshold':float(t),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'recall':float(recall_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred)),'fraccion_intervenida':float(pred.mean())}

def main(data,out):
    df=pd.read_csv(data); X=df[FEATURES].copy(); y=df.churn.astype(int)
    Xdev,Xte,ydev,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=RS)
    Xtr,Xcal,ytr,ycal=train_test_split(Xdev,ydev,test_size=.2,stratify=ydev,random_state=RS)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xcal=Xcal.fillna(med); Xte=Xte.fillna(med)
    spw=float((ytr==0).sum()/(ytr==1).sum())
    m=XGBClassifier(n_estimators=50,max_depth=3,learning_rate=.1,subsample=.9,colsample_bytree=.9,reg_lambda=1,scale_pos_weight=spw,random_state=RS,eval_metric='logloss',n_jobs=2)
    m.fit(Xtr,ytr); pcal=m.predict_proba(Xcal)[:,1]; praw=m.predict_proba(Xte)[:,1]
    eps=1e-6; lcal=np.log(np.clip(pcal,eps,1-eps)/(1-np.clip(pcal,eps,1-eps))); lte=np.log(np.clip(praw,eps,1-eps)/(1-np.clip(praw,eps,1-eps)))
    pl=LogisticRegression(solver='lbfgs',random_state=RS).fit(lcal.reshape(-1,1),ycal); ppl=pl.predict_proba(lte.reshape(-1,1))[:,1]
    iso=IsotonicRegression(out_of_bounds='clip').fit(pcal,ycal); piso=iso.predict(praw)
    comps=[]; rel={}
    for name,p in [('Sin calibrar',praw),('Platt',ppl),('Isotónica',piso)]:
        ec,r=ece(yte,p); rel[name]=r; comps.append({'metodo':name,'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p),'brier':brier_score_loss(yte,p),'log_loss':log_loss(yte,p),'ece_10_bins':ec})
    comp=pd.DataFrame(comps)
    grid=pd.DataFrame([thr_metrics(yte,ppl,t) for t in np.linspace(.01,.99,99)])
    bestf=grid.loc[grid.f1.idxmax()].to_dict()
    costs=[]
    for ratio in [2,5,10]:
        q=grid.copy(); q['ratio_costo_fn_fp']=ratio; q['costo_relativo']=ratio*q.fn+q.fp; costs.append(q.loc[q.costo_relativo.idxmin()].to_dict())
    cap=[]; yy=yte.to_numpy()
    for frac in [.10,.20,.25,.30]:
        cut=float(np.quantile(ppl,1-frac)); sel=ppl>=cut; tp=int(np.sum(sel&(yy==1))); fp=int(np.sum(sel&(yy==0))); fn=int(np.sum((~sel)&(yy==1)))
        cap.append({'capacidad':frac,'threshold_aprox':cut,'clientes_seleccionados':int(sel.sum()),'captura_abandono':tp/int(np.sum(yy==1)),'tasa_abandono_grupo':tp/int(sel.sum()),'tp':tp,'fp':fp,'fn':fn})
    capdf=pd.DataFrame(cap); costdf=pd.DataFrame(costs)
    summary={'split':{'train':len(Xtr),'calibration':len(Xcal),'test':len(Xte)},'selected_calibration':'Platt','platt_coef':float(pl.coef_[0,0]),'platt_intercept':float(pl.intercept_[0]),'threshold_050':thr_metrics(yte,ppl,.5),'best_f1_threshold':bestf,'cost_scenarios':costs,'capacity_scenarios':cap}
    raw=comp[comp.metodo=='Sin calibrar'].iloc[0]; plc=comp[comp.metodo=='Platt'].iloc[0]
    assert plc.brier < raw.brier and plc.ece_10_bins < raw.ece_10_bins and abs(plc.roc_auc-raw.roc_auc)<1e-12
    out=Path(out); out.mkdir(parents=True,exist_ok=True); comp.to_csv(out/'calibration_comparison.csv',index=False); grid.to_csv(out/'threshold_curve.csv',index=False); costdf.to_csv(out/'cost_thresholds.csv',index=False); capdf.to_csv(out/'capacity_thresholds.csv',index=False); rel['Sin calibrar'].to_csv(out/'reliability_raw.csv',index=False); rel['Platt'].to_csv(out/'reliability_platt.csv',index=False); (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('VALIDACION_OK'); print(comp.to_string(index=False)); print(capdf.to_string(index=False))
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--data',required=True); a.add_argument('--out',required=True); z=a.parse_args(); main(z.data,z.out)
