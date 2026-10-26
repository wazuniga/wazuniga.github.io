import subprocess,sys,numpy as np,concurrent.futures as cf,os
import tempfile; TMPDIR=tempfile.gettempdir()  # scratch files for the C program
EXE="./aval"
def run(mode,sigma,b,s0,nruns,tmax,seed,L=40,nmax=20000,out=None):
    out=out or f"{TMPDIR}/r_{mode}_{sigma}_{b}_{seed}.bin"
    r=subprocess.run([EXE,str(mode),str(sigma),repr(b),str(s0),str(nruns),str(tmax),str(seed),str(L),out,str(nmax)],capture_output=True,text=True)
    a=np.array([[float(x) for x in l.split()] for l in r.stdout.strip().split("\n")])
    return a,out
def curv(mode,sigma,b,s0,nruns,tmax,seeds=(1,2),nmax=3000):
    with cf.ThreadPoolExecutor(2) as ex:
        full=list(ex.map(lambda sd: run(mode,sigma,b,s0,nruns//len(seeds),tmax,sd,nmax=nmax),seeds))
    res=[f[0] for f in full]
    fl=np.concatenate([np.fromfile(f[1],dtype=np.float32).reshape(-1,3)[:,2] for f in full])
    if (fl==2).mean()>0.002: return +9.0,0.0,0.0,0.0
    t=res[0][:,0]; N=np.mean([r[:,1] for r in res],axis=0); P=np.mean([r[:,2] for r in res],axis=0)
    lt=np.log(t); 
    def slope(y,a,b_):
        m=(t>=a)&(t<=b_)&(y>0)
        if m.sum()<3: return -50.0
        return np.polyfit(lt[m],np.log(y[m]),1)[0]
    e=slope(N,tmax/30,tmax/5.5); l=slope(N,tmax/5.5,tmax)
    return l-e,e,l,slope(P,tmax/10,tmax)
if __name__=="__main__":
    mode=int(sys.argv[1]);sigma=float(sys.argv[2]);s0=float(sys.argv[3]);lo=float(sys.argv[4]);hi=float(sys.argv[5])
    nruns=int(sys.argv[6]);tmax=float(sys.argv[7]);iters=int(sys.argv[8])
    for it in range(iters):
        b=0.5*(lo+hi);k,e,l,dp=curv(mode,sigma,b,s0,nruns,tmax)
        print(f"b={b:.5f} curv={k:+.3f} slopeN early={e:+.3f} late={l:+.3f} slopeP={dp:+.3f}",flush=True)
        if k>0: hi=b
        else: lo=b
    print("bc ~",0.5*(lo+hi))
