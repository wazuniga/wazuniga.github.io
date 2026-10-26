import numpy as np,glob,sys
def load(tag):
    A=np.array([np.load(f) for f in sorted(glob.glob(f'lps_{tag}_*.npy'))]); return A   # seeds x times x 4
def analyze(tag,unit,verbose=True):
    A=load(tag); t=A[0,:,0]; lt=np.log(t); ns=len(A)
    m=(t>=4)&(t<=900)
    out={}
    for name,col,logy in [('D',3,False),('N',1,True),('P',2,True)]:
        Y=A[:,:,col]/(unit if name=='D' else 1.0); Y=np.log(Y) if logy else Y
        # residual of each seed from a straight line in ln t over the window
        R=[];S=[]
        for y in Y:
            c=np.polyfit(lt[m],y[m],2 if name!='D' else 1); R.append(y[m]-np.polyval(c,lt[m])); S.append(c[-2])
        R=np.array(R); mean=R.mean(0); sem=R.std(0,ddof=1)/np.sqrt(ns)
        out[name]=(lt[m],mean,sem,np.mean(S),np.std(S,ddof=1)/np.sqrt(ns),R)
        if verbose: print(tag,name,'slope %.4f±%.4f'%(np.mean(S),np.std(S,ddof=1)/np.sqrt(ns)),'rms residual %.4f  typical sem %.4f  chi2/dof vs zero %.1f'%(np.sqrt((mean**2).mean()),sem.mean(),((mean/sem)**2).mean()))
    return out
def sinefit(x,y,sem,periods):
    best=None
    for Lm in periods:
        M=np.c_[np.cos(2*np.pi*x/Lm),np.sin(2*np.pi*x/Lm)]; w=1/sem
        c,res,_,_=np.linalg.lstsq(M*w[:,None],y*w,rcond=None); chi=(((M@c-y)*w)**2).sum()
        if best is None or chi<best[0]: best=(chi,Lm,np.hypot(*c),c)
    return best
if __name__=="__main__":
    tag=sys.argv[1]; unit=float(sys.argv[2])
    o=analyze(tag,unit)
    for name in ['D','N','P']:
        x,y,s,sl,se,R=o[name]; chi0=((y/s)**2).sum()
        b=sinefit(x,y,s,np.linspace(0.6,4.0,341))
        print(name,'chi2(no oscillation)=%.1f for %d points; best sine: period in ln t = %.2f (factor %.2f in t), amplitude %.4f, chi2=%.1f'%(chi0,len(x),b[1],np.exp(b[1]),b[2],b[0]))
        # per-seed amplitude at best period (phase-coherent): project each seed
        M=np.c_[np.cos(2*np.pi*x/b[1]),np.sin(2*np.pi*x/b[1])]
        cs=np.array([np.linalg.lstsq(M,r,rcond=None)[0] for r in R]); mu=cs.mean(0); cov=np.cov(cs.T)/len(cs)
        print('   coefficient mean',mu,' T2=',float(mu@np.linalg.solve(cov,mu)),'(seeds=%d)'%len(cs))
