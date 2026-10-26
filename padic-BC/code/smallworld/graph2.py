# Euclidean ring with power-law long-range links, same construction as the tree graph
import numpy as np, json
exec(open('graph.py').read().split("out={}")[0])
def levy_graph(l,sigma,k):
    n=1<<l; r=np.arange(1,n//2); pr=r**(-(1.0+sigma)); pr/=pr.sum()
    src=np.repeat(np.arange(n,dtype=np.int64),k)
    rr=rng.choice(r,size=src.size,p=pr)*rng.choice([-1,1],size=src.size)
    return sym(n,src,(src+rr)%n)
out={}
for sigma in [0.5,1.5]:
    out[f"levy_{sigma}"]=[]
    for l in range(8,19,2):
        s=stats(levy_graph(l,sigma,6)); out[f"levy_{sigma}"].append(s); print(sigma,l,s,flush=True)
json.dump(out,open('graph2.json','w'),indent=1)
