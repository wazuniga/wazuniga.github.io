import numpy as np,sys,json,time,concurrent.futures as cf
import tempfile; TMPDIR=tempfile.gettempdir()  # scratch files for the C program
from crit import run,curv
S0=0.25
configs=[("tree",0,0.25),("tree",0,0.5),("tree",0,0.75),("tree",0,1.0),("tree",0,1.5),("tree",0,2.5),("chain",1,0.0)]
res={}
def bisect(mode,sig,lo,hi,nruns,tmax,iters,log,nmax=3000):
    for it in range(iters):
        b=0.5*(lo+hi);k,e,l,dp=curv(mode,sig,b,S0,nruns,tmax,nmax=nmax)
        log.write(f"   b={b:.5f} curv={k:+.3f} early={e:+.3f} late={l:+.3f} slopeP={dp:+.3f}\n");log.flush()
        if k>0: hi=b
        else: lo=b
    return lo,hi
if __name__=="__main__":
    with open("production.log","w") as log:
        for name,mode,sig in configs:
            t0=time.time();log.write(f"== {name} sigma={sig}\n");log.flush()
            lo,hi=bisect(mode,sig,1.0,4.0,4000,100,6,log,nmax=1500)
            w=hi-lo;lo,hi=bisect(mode,sig,lo-w,hi+w,40000,300,5,log)
            w=hi-lo;lo,hi=bisect(mode,sig,lo-w,hi+w,80000,1000,4,log,nmax=6000)
            bc=0.5*(lo+hi);log.write(f"   bc={bc:.5f}  ({time.time()-t0:.0f}s)\n");log.flush()
            seeds=(101,202)
            with cf.ThreadPoolExecutor(2) as ex:
                out=list(ex.map(lambda sd: run(mode,sig,bc,S0,100000,1000,sd,nmax=60000,out=f"{TMPDIR}/prod_{name}_{sig}_{sd}.bin"),seeds))
            A=np.mean([o[0] for o in out],axis=0)
            rec=np.concatenate([np.fromfile(o[1],dtype=np.float32).reshape(-1,3) for o in out])
            np.savez_compressed(f"res_{name}_{sig}.npz",t=A[:,0],N=A[:,1],P=A[:,2],S=rec[:,0],T=rec[:,1],flag=rec[:,2],bc=bc,blo=lo,bhi=hi)
            log.write(f"   done ({time.time()-t0:.0f}s) flagged={rec[:,2].mean():.4f}\n");log.flush()
        # pure branching reference (no saturation): critical at b=1 exactly
        out=[run(2,0.0,1.0,S0,100000,1000,sd,out=f"{TMPDIR}/prod_mf_{sd}.bin") for sd in (101,202)]
        A=np.mean([o[0] for o in out],axis=0);rec=np.concatenate([np.fromfile(o[1],dtype=np.float32).reshape(-1,3) for o in out])
        np.savez_compressed("res_branching.npz",t=A[:,0],N=A[:,1],P=A[:,2],S=rec[:,0],T=rec[:,1],flag=rec[:,2],bc=1.0,blo=1.0,bhi=1.0)
        log.write("ALL DONE\n")
