import numpy as np,sys,time,concurrent.futures as cf
import tempfile; TMPDIR=tempfile.gettempdir()  # scratch files for the C program
from crit import run,curv
from production import bisect,S0
name,mode,sig,lo,hi=sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])
nprod=int(sys.argv[6])
with open(f"extra_{name}_{sig}.log","w") as log:
    t0=time.time()
    lo,hi=bisect(mode,sig,lo,hi,4000,100,7,log,nmax=1500)
    w=hi-lo;lo,hi=bisect(mode,sig,lo-w,hi+w,40000,300,5,log)
    w=hi-lo;lo,hi=bisect(mode,sig,lo-w,hi+w,60000,1000,4,log,nmax=6000)
    bc=0.5*(lo+hi);log.write(f"   bc={bc:.5f}  ({time.time()-t0:.0f}s)\n");log.flush()
    seeds=(101,202)
    with cf.ThreadPoolExecutor(2) as ex:
        out=list(ex.map(lambda sd: run(mode,sig,bc,S0,nprod//2,1000,sd,nmax=60000,out=f"{TMPDIR}/prod_{name}_{sig}_{sd}.bin"),seeds))
    A=np.mean([o[0] for o in out],axis=0)
    rec=np.concatenate([np.fromfile(o[1],dtype=np.float32).reshape(-1,3) for o in out])
    np.savez_compressed(f"res_{name}_{sig}.npz",t=A[:,0],N=A[:,1],P=A[:,2],S=rec[:,0],T=rec[:,1],flag=rec[:,2],bc=bc,blo=lo,bhi=hi)
    log.write(f"   done ({time.time()-t0:.0f}s)\nALL DONE\n")
