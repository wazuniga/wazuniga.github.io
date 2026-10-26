import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import expm
BLUE,ORANGE,AQUA,PURPLE,INK,MUTED="#2a78d6","#eb6834","#1baf7a","#8a4fd0","#0b0b0b","#8a8985"
COL=[BLUE,ORANGE,AQUA,PURPLE]; LS=["-","--","-.",":"]
plt.rcParams.update({"font.size":8,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK})
# four regions B_a = a + 4Z_2, a=0..3 (p=2, N=1); vol = 1/4
p=2; A=4; vol=np.full(A,0.25)
def d(a,b):                       # |a-b|_2 for a != b in {0,1,2,3}
    x=abs(a-b); k=0
    while x%2==0: x//=2; k+=1
    return 2.0**(-k)
alpha=np.array([1.0,1.2,0.8,1.5])     # decay rates
fp=np.array([1.0,1.5,0.8,1.2])        # f_a'(0)
fw=np.array([0.60,0.70,0.50,0.90])    # f_a'(0)*omega_a (intra-regional branching)
sig=1.0
def C(eps):                           # c_ab = eps * d_ab^{-(1+sigma)}
    c=np.zeros((A,A))
    for a in range(A):
        for b in range(A):
            if a!=b: c[a,b]=eps*d(a,b)**(-(1+sig))
    return c
def M(eps):
    c=C(eps); m=np.diag(-alpha+fw)
    for a in range(A):
        for b in range(A):
            if a!=b: m[a,b]=fp[a]*vol[a]*c[a,b]
    return m
def s(eps): return np.max(np.linalg.eigvals(M(eps)).real)
def mu(eps): return alpha-fw-fp*(C(eps)@vol)
eps=np.linspace(0,0.6,301)
S=np.array([s(e) for e in eps]); MU=np.array([mu(e) for e in eps])
ec=brentq(s,0,0.6)
print("eps_c",ec,"mu_a at eps_c",mu(ec),"M_aa",-alpha+fw)
fig,ax=plt.subplots(1,2,figsize=(6.8,2.7),constrained_layout=True)
ax[0].fill_between(eps,-MU.max(1),-MU.min(1),color=BLUE,alpha=0.12,lw=0,label=r"$[-\max_a\mu_a,\,-\min_a\mu_a]$")
ax[0].plot(eps,S,color=BLUE,lw=2,label=r"$s(M)$")
ax[0].axhline(0,color=MUTED,lw=0.8); ax[0].axvline(ec,color=MUTED,lw=0.8,ls=":")
ax[0].text(ec+0.01,-0.62,rf"$\varepsilon_c\approx{ec:.3f}$",color=INK,fontsize=7)
ax[0].set_xlabel(r"interregional coupling $\varepsilon$"); ax[0].set_ylabel("growth rate")
ax[0].set_title("(a) leading growth rate",fontsize=8,loc="left"); ax[0].legend(fontsize=7,loc="upper left",frameon=False)
ax[0].grid(color="#e6e6e3",lw=0.5)
t=np.linspace(0,40,400); U0=np.array([1.0,0,0,0]); Mc=M(ec)
U=np.array([expm(Mc*tt)@U0 for tt in t])
for a in range(A):
    ax[1].plot(t,U[:,a],color=COL[a],ls=LS[a],lw=1.6,label=f"region {a}")
ax[1].plot(t,U.sum(1),color=INK,lw=1.0,label="total")
ax[1].set_xlabel(r"time $t$"); ax[1].set_ylabel(r"regional activity $U_a(t)$")
ax[1].set_title(r"(b) critical network, $\varepsilon=\varepsilon_c$",fontsize=8,loc="left")
ax[1].legend(fontsize=7,loc="center right",frameon=False,ncol=1); ax[1].grid(color="#e6e6e3",lw=0.5)
fig.savefig("fig_regional.pdf"); plt.close(fig)
w,V=np.linalg.eig(Mc); i=np.argmax(w.real); v=np.abs(V[:,i].real); print("Perron vec",v/v.sum(), "U(40)/sum",U[-1]/U[-1].sum(), "total(40)",U[-1].sum())
print("intra-regional rates -alpha_a",-alpha)
