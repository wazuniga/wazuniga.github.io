# critical point + production for a general-p tree (avalp) ; usage: lp.py p sigma lo hi nprod tmax
import numpy as np,sys,time,concurrent.futures as cf
import tempfile; TMPDIR=tempfile.gettempdir()  # scratch files for the C program
import crit; crit.EXE="./avalp"
from crit import run
from production import bisect,S0
p,sig,lo,hi,nprod,tmax=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6])
L={2:40,3:28,5:20,7:16,8:15}[p]
import functools
_r=crit.run
crit.run=lambda mode,sigma,b,s0,nruns,tm,seed,L_=40,nmax=20000,out=None:_r(mode,sigma,b,s0,nruns,tm,seed,L=L,nmax=nmax,out=out)
with open(f"lp_{p}_{sig}.log","w") as log:
    t0=time.time()
    lo,hi=bisect(p,sig,lo,hi,4000,100,7,log,nmax=1500)
    w=hi-lo;lo,hi=bisect(p,sig,lo-w,hi+w,40000,300,5,log)
    w=hi-lo;lo,hi=bisect(p,sig,lo-w,hi+w,60000,1000,4,log,nmax=6000)
    bc=0.5*(lo+hi);log.write(f"   bc={bc:.5f}  ({time.time()-t0:.0f}s)\n");log.flush()
    seeds=(101,202)
    with cf.ThreadPoolExecutor(2) as ex:
        out=list(ex.map(lambda sd: crit.run(p,sig,bc,S0,nprod//2,tmax,sd,nmax=60000,out=f"{TMPDIR}/lp_{p}_{sig}_{sd}.bin"),seeds))
    A=np.mean([o[0] for o in out],axis=0)
    rec=np.concatenate([np.fromfile(o[1],dtype=np.float32).reshape(-1,3) for o in out])
    np.savez_compressed(f"res_p{p}_{sig}.npz",t=A[:,0],N=A[:,1],P=A[:,2],D=A[:,3],S=rec[:,0],T=rec[:,1],flag=rec[:,2],bc=bc,blo=lo,bhi=hi)
    log.write(f"   done ({time.time()-t0:.0f}s)\nALL DONE\n")
