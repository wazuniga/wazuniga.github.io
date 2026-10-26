# Small-world diagnostics of sparse realizations of the hierarchical (tree) coupling
# and of a Euclidean ring lattice with the same mean degree.
import numpy as np, json, sys
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
rng=np.random.default_rng(1)
def tree_graph(l,sigma,k):
    n=1<<l; d=np.arange(1,l+1); pr=2.0**(-sigma*d); pr/=pr.sum()
    src=np.repeat(np.arange(n,dtype=np.int64),k)
    dd=rng.choice(d,size=src.size,p=pr)
    low=rng.integers(0,1<<62,size=src.size) & ((1<<(dd-1))-1)
    tgt=((src>>dd)<<dd) | ((((src>>(dd-1))&1)^1)<<(dd-1)) | low
    return sym(n,src,tgt)
def ring_graph(n,h):
    src=np.repeat(np.arange(n,dtype=np.int64),h); off=np.tile(np.arange(1,h+1),n)
    return sym(n,src,(src+off)%n)
def sym(n,src,tgt):
    A=coo_matrix((np.ones(src.size,dtype=np.int8),(src,tgt)),shape=(n,n)).tocsr()
    A=A+A.T; A.data[:]=1; A.setdiag(0); A.eliminate_zeros(); return A.tocsr()
def clustering(A,ns=3000):
    n=A.shape[0]; idx=rng.choice(n,min(ns,n),replace=False); c=[]
    for i in idx:
        nb=A.indices[A.indptr[i]:A.indptr[i+1]]; kk=nb.size
        if kk<2: c.append(0.0); continue
        sub=A[nb][:,nb]; c.append(sub.nnz/(kk*(kk-1)))
    return float(np.mean(c))
def stats(A,nsrc=24):
    n=A.shape[0]; nc,lab=connected_components(A,directed=False)
    cnt=np.bincount(lab); g=np.argmax(cnt); giant=np.where(lab==g)[0]
    srcs=rng.choice(giant,min(nsrc,giant.size),replace=False)
    D=shortest_path(A,method='D',unweighted=True,indices=srcs)
    D=D[:,giant]; L=float(D[D>0].mean())
    return dict(n=n,deg=float(A.nnz/n),giant=float(giant.size/n),L=L,C=clustering(A))
out={}
K=6
for sigma in [0.25,0.5,0.75,1.0,1.5]:
    out[f"tree_{sigma}"]=[]
    for l in range(8,19,2):
        s=stats(tree_graph(l,sigma,K)); out[f"tree_{sigma}"].append(s); print(sigma,l,s,flush=True)
out["ring"]=[]
for l in range(8,17,2):
    s=stats(ring_graph(1<<l,5)); out["ring"].append(s); print('ring',l,s,flush=True)
json.dump(out,open('graph.json','w'),indent=1)
