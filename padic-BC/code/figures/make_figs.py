import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
plt.rcParams.update({"font.size":8,"axes.linewidth":0.6,"font.family":"serif","mathtext.fontset":"cm",
 "xtick.direction":"out","ytick.direction":"out","axes.spines.top":False,"axes.spines.right":False,
 "legend.frameon":False,"pdf.fonttype":42})
BLUE,ORANGE,AQUA,INK,MUTED="#2a78d6","#eb6834","#1baf7a","#0b0b0b","#8a8985"
seq=LinearSegmentedColormap.from_list("seq",["#f3f7fd","#cde2fb","#86b6ef","#2a78d6","#104281","#0d366b"])

# ---------- Fig 1: connection matrices ----------
p,l,sig=2,6,1.0
n=p**l
def digits_rev(i):   # position on the tree boundary: bit-reversal makes balls contiguous
    r=0
    for k in range(l): r=r*p+(i//p**k)%p
    return r
def vp(m):
    k=0
    while m%p==0: m//=p; k+=1
    return k
# index neurons by tree position s; s -> p-adic integer i with reversed digits
inv=[digits_rev(s) for s in range(n)]
Wt=np.zeros((n,n)); We=np.zeros((n,n))
for a in range(n):
    for b in range(n):
        if a==b: continue
        d=p**(-vp(abs(inv[a]-inv[b])))          # |i-j|_p in (p^{-l+1},...,1)
        Wt[a,b]=(d*n)**(-(1+sig))               # distance in units of the smallest ball
        de=min(abs(a-b),n-abs(a-b))             # ring distance
        We[a,b]=de**(-(1.0+sig))
Wt/=Wt.sum(1).mean(); We/=We.sum(1).mean()      # same total coupling w0=1
fig,ax=plt.subplots(1,2,figsize=(6.8,3.0),constrained_layout=True)
vmin=min(Wt[Wt>0].min(),We[We>0].min()); vmax=max(Wt.max(),We.max())
for A,M,t in zip(ax,[Wt,We],["(a) hierarchical ($p$-adic) network","(b) Euclidean network (ring)"]):
    Mm=np.ma.masked_where(M<=0,M)
    im=A.imshow(Mm,cmap=seq,norm=LogNorm(vmin=vmin,vmax=vmax),interpolation="nearest")
    A.set_title(t,fontsize=8,loc="left"); A.set_xlabel("presynaptic neuron $j$"); A.set_ylabel("postsynaptic neuron $i$")
    A.set_xticks([0,16,32,48,63]); A.set_yticks([0,16,32,48,63])
    for s in ("top","right"): A.spines[s].set_visible(True)
cb=fig.colorbar(im,ax=ax,shrink=0.85,pad=0.02); cb.set_label("coupling $w_{ij}$ (total coupling $w_0=1$)")
fig.savefig("fig_connection_matrices.pdf"); plt.close(fig)

# ---------- Fig 2: level-resolved spectrum ----------
p,N,sig=2,1,1.0
J=np.arange(-12,13)                      # levels j: |x|=p^j
def Wlevels(w):                          # W_j = (1-p^-N) p^{jN} w(p^j)
    return (1-float(p)**-N)*float(p)**(J*N)*w
def what(W):                             # \hat w at level gamma = sum_{j<gamma} W_j - W_gamma/(p^N-1)
    c=np.concatenate([[0.0],np.cumsum(W)])[:-1]
    return c-W/(float(p)**N-1)
j0=-3
w_mono=np.where(J<=j0,1.0,float(p)**(-(J-j0)*(N+sig)))
w_mono=np.where(J<J.min()+0,w_mono,w_mono)
# core: constant below j0 ; add the mass of all levels below J.min() analytically (negligible)
w_inh=w_mono.copy(); w_inh[J>=2]*=-6.0   # inhibition at large hierarchical distance
out={}
for name,w in (("mono",w_mono),("inh",w_inh)):
    W=Wlevels(w); w0=np.sum(np.abs(Wlevels(w_mono)))  # normalise both by the excitatory mass
    out[name]=what(W)/w0
fig,ax=plt.subplots(1,2,figsize=(6.8,2.6),constrained_layout=True)
g=J
ax[0].plot(g,out["mono"],"-o",color=BLUE,ms=3,lw=1.4,label="excitatory, decreasing kernel")
ax[0].plot(g,out["inh"],"--s",color=ORANGE,ms=3,lw=1.4,label="inhibition at large distance")
ax[0].set_xlim(-6,12); ax[0].set_xticks(range(-6,13,2)); ax[0].set_xlabel(r"level $\gamma$ of the mode (ball of radius $p^{\gamma}$)")
ax[0].set_ylabel(r"$\hat w$ at level $\gamma$ (units of $w_0$)"); ax[0].set_title("(a) $p$-adic theory",fontsize=8,loc="left")
ax[0].legend(fontsize=7,loc="lower right")
kc=int(np.argmax(out["inh"])); ax[0].annotate(r"$\gamma_c$",(g[kc],out["inh"][kc]),textcoords="offset points",xytext=(0,6),ha="center",fontsize=8,color=INK)
q=np.linspace(0,6,400)
gauss=np.exp(-q**2/2)
dog=1.6*np.exp(-q**2/2)-1.2*np.exp(-(1.8*q)**2/2)
dog/= (1.6)   # same excitatory mass
ax[1].plot(q,gauss,"-",color=BLUE,lw=1.4,label="excitatory (Gaussian) kernel")
ax[1].plot(q,dog,"--",color=ORANGE,lw=1.4,label="difference of Gaussians")
qc=q[np.argmax(dog)]; ax[1].annotate(r"$q_c$",(qc,dog.max()),textcoords="offset points",xytext=(0,6),ha="center",fontsize=8,color=INK)
ax[1].set_xlabel(r"wave number $q$"); ax[1].set_ylabel(r"$\hat w(q)$ (units of $w_0$)")
ax[1].set_title("(b) Euclidean theory",fontsize=8,loc="left"); ax[1].legend(fontsize=7,loc="upper right")
ax[1].set_ylim(ax[0].get_ylim())
fig.savefig("fig_level_spectrum.pdf"); plt.close(fig)
print("gamma_c =",g[kc], "max what(inh) =",out["inh"].max())

# ---------- Fig 3: propagator ----------
p,N,sig=2,1,1.0
alpha=1.0; j0=0                     # kernel constant on |x|<=p^{j0}, power law beyond
def w_of(k): return 1.0 if k<=j0 else float(p)**(-(k-j0)*(N+sig))
# w0 = int w = core mass + tail
w0=p**(j0*N) + sum((1-p**-N)*p**(k*N)*w_of(k) for k in range(j0+1,400))
KMAX=400
def what_m(m):                      # \hat w at |xi| = p^m  (Lemma: radial Fourier transform)
    K=-m
    if K<=j0: s=p**(K*N)                       # all inside the core
    else: s=p**(j0*N)+sum((1-p**-N)*p**(k*N)*w_of(k) for k in range(j0+1,K+1))
    return s-p**(-m*N)*w_of(1-m)
c=0.9*alpha/w0                      # f'(0): slightly subcritical, mu = 0.1 alpha
mu=alpha-c*w0
MS=np.arange(-KMAX,-j0+1)           # support of \hat w: |xi|<=p^{-j0}
wh={m:what_m(m) for m in range(-60,-j0+2)}
def g_t(m,t):
    if m>-j0: return 0.0
    v=wh[m] if m>=-60 else w0
    return np.expm1(c*t*v)
def K_t(k,t):                       # explicit formula of Proposition (propagator)
    s=sum((1-p**-N)*float(p)**(j*N)*g_t(j,t) for j in range(-300,-k+1))
    return s-float(p)**(-k*N)*g_t(1-k,t)
ks=np.arange(1,15); r=2.0**ks
fig,ax=plt.subplots(1,2,figsize=(6.8,2.7),constrained_layout=True)
for t,col,ls in ((1.0,AQUA,"-"),(4.0,BLUE,"-"),(16.0,ORANGE,"-")):
    y=np.array([np.exp(-alpha*t)*K_t(k,t) for k in ks])
    ax[0].loglog(r,y,ls,marker="o",ms=3,color=col,lw=1.3,label=f"$t={t:g}$")
    tail=c*t*np.exp(-mu*t)*np.array([w_of(k) for k in ks])
    ax[0].loglog(r,tail,":",color=col,lw=1.0)
ax[0].set_xlabel(r"hierarchical distance $|x|_p$"); ax[0].set_ylabel(r"regular part of $\Delta(x,t)$")
ax[0].set_title("(a) $p$-adic theory",fontsize=8,loc="left"); ax[0].legend(fontsize=7,loc="lower left")
ax[0].text(0.97,0.95,"dotted: asymptote\n"+r"$f'(0)\,t\,e^{-\mu t}\,w(|x|_p)$",transform=ax[0].transAxes,ha="right",va="top",fontsize=7,color=INK)
D=1.0; x=np.logspace(0,np.log10(r.max()),300)
for t,col in ((1.0,AQUA),(4.0,BLUE),(16.0,ORANGE)):
    y=np.exp(-mu*t)*np.exp(-x**2/(4*D*t))/np.sqrt(4*np.pi*D*t)
    ax[1].loglog(x,y,"-",color=col,lw=1.3,label=f"$t={t:g}$")
ax[1].set_ylim(ax[0].get_ylim()); ax[1].set_xlabel(r"distance $|x|$"); ax[1].set_ylabel(r"$\Delta(x,t)$")
ax[1].set_title("(b) Euclidean theory (diffusion)",fontsize=8,loc="left"); ax[1].legend(fontsize=7,loc="lower left")
fig.savefig("fig_propagator.pdf"); plt.close(fig)
# numerical check of the tail
for t in (1.0,4.0,16.0):
    k=14; print(t, np.exp(-alpha*t)*K_t(k,t)/(c*t*np.exp(-mu*t)*w_of(k)))
