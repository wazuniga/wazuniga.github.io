import numpy as np,json,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.linewidth":0.6,"font.family":"serif","mathtext.fontset":"cm","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"pdf.fonttype":42})
r=json.load(open('lp_result.json'))
fig,ax=plt.subplots(2,1,figsize=(3.4,3.6),sharex=True,constrained_layout=True)
for a,(tag,col,lab) in zip(ax,[('p7_1.0','#2a78d6','tree, $p=7$'),('levy_1.0','#1baf7a','Euclidean chain, power-law coupling')]):
    d=r[tag]; x=np.array(d['x']); y=np.array(d['mean'])*1e3; s=np.array(d['sem'])*1e3
    a.axhline(0,color='#8a8985',lw=0.6,ls=':')
    a.fill_between(np.exp(x),y-s,y+s,color=col,alpha=0.25,lw=0); a.plot(np.exp(x),y,'-',color=col,lw=1.2)
    c=d['phase']; xx=np.linspace(x.min(),x.max(),300)
    if tag=='p7_1.0':
        a.plot(np.exp(xx),1e3*(c[0]*np.cos(2*np.pi*xx/d['Lam'])+c[1]*np.sin(2*np.pi*xx/d['Lam'])),'--',color='#0b0b0b',lw=0.8)
        t0=6.0
        for k in range(4): a.axvline(t0*np.exp(k*d['Lam']),color='#8a8985',lw=0.5,ls=':')
    a.set_xscale('log'); a.set_ylim(-9,9); a.set_title(lab,fontsize=8,loc='left'); a.set_ylabel(r'residual ($10^{-3}$ levels)')
ax[1].set_xlabel('time $t$')
fig.savefig('fig_logperiodic.pdf'); fig.savefig('fig_logperiodic.png',dpi=160)
