# per-seed production at a known critical point.  usage: lp_seeds.py tree|levy p_or_mode sigma bc nruns tmax seed0 nseeds tag
import numpy as np,sys,os,subprocess,concurrent.futures as cf
import tempfile; TMPDIR=tempfile.gettempdir()  # scratch files for the C program
kind,pm,sig,bc,nruns,tmax,seed0,nseeds,tag=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],int(sys.argv[7]),int(sys.argv[8]),sys.argv[9]
def one(sd):
    out=f"{TMPDIR}/lps_{tag}_{sd}.bin"
    if kind=="tree":
        L={2:40,3:28,5:20,7:16,8:15}[int(pm)]
        cmd=["./avalp",pm,sig,bc,"0.25",nruns,tmax,str(sd),str(L),out,"60000"];env=None
    else:
        cmd=["./aval","3",sig,bc,"0.25",nruns,tmax,str(sd),"40",out,"60000"];env=dict(os.environ,FINE="1")
    r=subprocess.run(cmd,capture_output=True,text=True,env=env)
    a=np.array([[float(x) for x in l.split()] for l in r.stdout.strip().split("\n")])
    np.save(f"lps_{tag}_{sd}.npy",a); os.remove(out); return sd
with cf.ThreadPoolExecutor(2) as ex: list(ex.map(one,range(seed0,seed0+nseeds)))
print("done")
