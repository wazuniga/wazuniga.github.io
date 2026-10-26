import numpy as np,glob,json
from lp_analyze import load
res={}
for tag,unit in [('p7_1.0',1.0),('levy_1.0',np.log(7))]:
    A=load(tag); t=A[0,:,0]; lt=np.log(t); m=(t>=4)&(t<=900); x=lt[m]; ns=len(A)
    Y=A[:,:,3]/unit
    slope=np.mean([np.polyfit(x,y[m],1)[0] for y in Y]); Lam=1/slope      # predicted period: one level
    R=np.array([y[m]-np.polyval(np.polyfit(x,y[m],2),x) for y in Y])      # quadratic detrend
    def proj(L):
        M=np.c_[np.cos(2*np.pi*x/L),np.sin(2*np.pi*x/L)]
        cs=np.array([np.linalg.lstsq(M,r,rcond=None)[0] for r in R]); mu=cs.mean(0); cov=np.cov(cs.T)/ns
        T2=float(mu@np.linalg.solve(cov,mu)); F=T2*(ns-2)/(2*(ns-1)); return np.hypot(*mu),T2,F,cs
    amp,T2,F,cs=proj(Lam)
    from scipy.stats import f as fd
    p=fd.sf(F,2,ns-2)
    amps=np.hypot(cs[:,0],cs[:,1])
    scan=[(L,)+proj(L)[:2] for L in np.linspace(0.8,3.2,49)]
    best=max(scan,key=lambda r:r[2])
    print(f"{tag}: seeds={ns} slope={slope:.4f} predicted period ln t={Lam:.3f} (factor {np.exp(Lam):.2f}); amplitude at predicted period={amp:.4f}, T2={T2:.1f}, F={F:.1f}, p={p:.2g}; per-seed amplitudes {np.round(amps,4)}; scan max T2 at period {best[0]:.2f} (amp {best[1]:.4f}, T2 {best[2]:.1f})")
    mean=R.mean(0); sem=R.std(0,ddof=1)/np.sqrt(ns)
    res[tag]=dict(x=x.tolist(),mean=mean.tolist(),sem=sem.tolist(),slope=slope,Lam=Lam,amp=float(amp),p=float(p),scan=[list(map(float,r)) for r in scan],phase=[float(c) for c in cs.mean(0)])
json.dump(res,open('lp_result.json','w'))
