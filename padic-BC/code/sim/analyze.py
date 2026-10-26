import numpy as np,glob,json
def mle_tau(S,smin,smax,nboot=200,rng=np.random.default_rng(0)):
    # truncated continuous power-law MLE on [smin,smax]
    from math import log
    x=S[(S>=smin)&(S<=smax)].astype(float)
    def fit(x):
        lo,hi=1.01,3.0
        L=np.log(x).mean()
        def g(t):   # d/dt of log-likelihood
            a=1-t; A=smin**a; B=smax**a
            return -L - ( (-1/a) + (B*np.log(smax)-A*np.log(smin))/(B-A) )*(-1) 
        # numeric maximisation instead (robust)
        ts=np.linspace(1.01,2.5,600)
        a=1-ts; ll=np.log(np.abs(a)/np.abs(smax**a-smin**a))-ts*L
        return ts[np.argmax(ll)]
    t=fit(x); bs=[fit(rng.choice(x,len(x))) for _ in range(nboot)]
    return t,np.std(bs),len(x)
def slope(t,y,a,b):
    m=(t>=a)&(t<=b)&(y>0); c=np.polyfit(np.log(t[m]),np.log(y[m]),1); return c[0]
rows=[]
for f in ["res_branching.npz"]+sorted(glob.glob("res_tree_*.npz"),key=lambda s: float(s.split("_")[2][:-4]))+["res_chain_0.0.npz"]:
    try: d=np.load(f)
    except Exception: continue
    S=d["S"];T=d["T"];fl=d["flag"];t=d["t"];N=d["N"];P=d["P"]
    Sf=S[fl==0]
    tau,err,n=mle_tau(S,10,2000)          # include unfinished: S>=... all have S counted so far; use finished + window
    tau_f,err_f,n_f=mle_tau(Sf,10,2000)
    delta=-slope(t,P,30,1000); theta=slope(t,N,30,1000)
    tau_scal=1+delta/(1+theta+delta)
    rows.append(dict(file=f,bc=float(d["bc"]),tau_S=float(tau_f),err=float(err_f),n=int(n_f),delta=float(delta),theta=float(theta),tau_from_scaling=float(tau_scal),tau_T=float(1+delta),unfinished=float((fl>0).mean())))
    print(f"{f:26s} bc={float(d['bc']):.4f} tau_S={tau_f:.3f}±{err_f:.3f} (n={n_f}) delta={delta:.3f} theta={theta:+.3f} tau_S(scaling)={tau_scal:.3f} unfinished={np.mean(fl>0):.4f}")
json.dump(rows,open("summary.json","w"),indent=1)
