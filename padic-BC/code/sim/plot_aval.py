import numpy as np,json,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.linewidth":0.6,"font.family":"serif","mathtext.fontset":"cm",
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"pdf.fonttype":42})
INK,MUTED="#0b0b0b","#8a8985"
ramp=["#86b6ef","#5598e7","#2a78d6","#1c5cab","#104281","#0d366b"]   # sequential blue: sigma increasing
ORANGE="#eb6834"
rows={r["file"]:r for r in json.load(open("summary.json"))}
sigs=[0.25,0.5,0.75,1.0,1.5]
def ccdf(S):
    s=np.sort(S); n=len(s); xs=np.unique(np.round(np.logspace(0,np.log10(s.max()),60)))
    return xs,np.array([(s>=x).mean() for x in xs])
fig,ax=plt.subplots(1,2,figsize=(6.8,3.0),constrained_layout=True)
mk=["o","s","^","D","v","P"]
for sg,col,m in zip(sigs,ramp,mk):
    d=np.load(f"res_tree_{sg}.npz"); S=d["S"][d["flag"]==0]
    x,y=ccdf(S); k=(x<=3000)
    ax[0].loglog(x[k],y[k],"-",marker=m,ms=2.6,lw=1.0,color=col,label=rf"tree, $\sigma={sg:g}$")
d=np.load("res_chain_0.0.npz"); S=d["S"][d["flag"]==0]; x,y=ccdf(S); k=(x<=3000)
ax[0].loglog(x[k],y[k],"--",marker="x",ms=3,lw=1.0,color=ORANGE,label="Euclidean chain")
xx=np.array([10,3000.]); ax[0].loglog(xx,0.33*xx**-0.5,":",color=INK,lw=0.9); ax[0].text(25,0.33*25**-0.5*0.62,r"$S^{-1/2}$",fontsize=7)
ax[0].loglog(xx,0.75*xx**-0.108,":",color=INK,lw=0.9); ax[0].text(40,0.75*40**-0.108*1.25,r"$S^{-0.108}$",fontsize=7)
ax[0].set_xlabel("avalanche size $S$"); ax[0].set_ylabel(r"$P(\mathrm{size}\geq S)$"); ax[0].legend(fontsize=6.5,loc="lower left")
ax[0].set_title("(a) size distributions at criticality",fontsize=8,loc="left")
tau=[rows[f"res_tree_{s}.npz"]["tau_S"] for s in sigs]; err=[rows[f"res_tree_{s}.npz"]["err"] for s in sigs]
tsc=[rows[f"res_tree_{s}.npz"]["tau_from_scaling"] for s in sigs]
ax[1].axhline(1.5,color=MUTED,lw=0.7,ls=":"); ax[1].text(1.68,1.505,"mean field, 3/2",fontsize=7,ha="right",va="bottom",color=INK)
ax[1].axhline(1.108,color=ORANGE,lw=0.9,ls="--"); ax[1].text(0.13,1.113,"DP in one dimension, 1.108",fontsize=7,ha="left",va="bottom",color=INK)
ax[1].axvline(0.5,color=MUTED,lw=0.7,ls=":"); ax[1].text(0.52,1.56,r"$\sigma=N/2$",fontsize=7,color=INK)
ax[1].errorbar(sigs,tau,yerr=err,fmt="o-",color="#2a78d6",ms=4,lw=1.2,capsize=2,label="tree: fit of size distribution")
ax[1].plot(sigs,tsc,"s--",color="#104281",ms=3.5,lw=1.0,mfc="white",label=r"tree: from $\delta$ and $\vartheta$")
c=rows["res_chain_0.0.npz"]; ax[1].plot([1.62],[c["tau_S"]],"x",color=ORANGE,ms=6,mew=1.5,label="Euclidean chain (simulated)")
ax[1].set_xlabel(r"decay exponent $\sigma$ of the hierarchical coupling"); ax[1].set_ylabel(r"avalanche-size exponent $\tau_S$")
ax[1].set_ylim(0.9,1.6); ax[1].set_xlim(0.1,1.7); ax[1].legend(fontsize=6.5,loc="lower left",bbox_to_anchor=(0.0,0.0))
ax[1].set_title(r"(b) exponent versus $\sigma$ ($N=1$)",fontsize=8,loc="left")
fig.savefig("fig_avalanches.pdf"); print("saved")
