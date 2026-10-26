import json, numpy as np, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g=json.load(open('graph.json')); g2=json.load(open('graph2.json'))
sig=[0.25,0.5,0.75,1.0,1.5]; blues=['#9cc3ee','#5b9be0','#2a78d6','#1a549c','#0e2f5c']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,2,figsize=(7.4,3.3))
a=ax[0]
for s,c in zip(sig,blues):
    d=[r for r in g[f'tree_{s}'] if r['giant']>0.99]
    if d: a.plot([r['n'] for r in d],[r['L'] for r in d],'o-',color=c,ms=3.5,lw=1.4,label=f'tree, $\\sigma={s}$')
d=g2['levy_1.5']; a.plot([r['n'] for r in d],[r['L'] for r in d],'^--',color='#1baf7a',ms=3.5,lw=1.2,mfc='w',label='ring, power-law coupling, $\\sigma=1.5$')
d=g2['levy_0.5']; a.plot([r['n'] for r in d],[r['L'] for r in d],'s--',color='#1baf7a',ms=3.5,lw=1.2,mfc='w',label='ring, power-law coupling, $\\sigma=0.5$')
d=g['ring']; a.plot([r['n'] for r in d],[r['L'] for r in d],'D-',color='#eb6834',ms=3.5,lw=1.4,label='ring, nearest neighbors')
a.set_xscale('log',base=2); a.set_yscale('log'); a.set_xlabel('number of neurons $n$'); a.set_ylabel('mean path length'); a.set_title('(a) Sparse network: path length',loc='left',fontsize=9)
a.set_ylim(2,6e5); a.legend(frameon=False,fontsize=6.5,loc='upper left',ncol=2,columnspacing=0.8,handlelength=1.6)
a=ax[1]; res={}
def load(tag):
    out=[]
    for f in glob.glob(f'cov_{tag}_*.txt'):
        L=int(f.split('_')[-1][:-4]); x=np.loadtxt(f,ndmin=2)[:,0]; out.append((2**L,x.mean(),x.std()/np.sqrt(len(x)),len(x)))
    return np.array(sorted(out))
for s,c in zip(sig,blues):
    d=load(f'tree{s}'); res[f'tree_{s}']=d.tolist(); a.errorbar(d[:,0],d[:,1],d[:,2],fmt='o-',color=c,ms=3.5,lw=1.4)
for sg,mk in [(0.5,'s'),(1.5,'^')]:
    out=[]
    for f in glob.glob(f'lv_{sg}_*.txt'):
        L=int(f.split('_')[-1][:-4]); x=np.loadtxt(f,ndmin=2)[:,0]; out.append((2**L,x.mean(),x.std()/np.sqrt(len(x)),len(x)))
    d=np.array(sorted(out)); res[f'levy_{sg}']=d.tolist(); a.errorbar(d[:,0],d[:,1],d[:,2],fmt=mk+'--',color='#1baf7a',ms=3.5,lw=1.2,mfc='w')
d=load('ring'); res['ring']=d.tolist(); a.errorbar(d[:,0],d[:,1],d[:,2],fmt='D-',color='#eb6834',ms=3.5,lw=1.4)
a.set_xscale('log',base=2); a.set_yscale('log'); a.set_xlabel('number of neurons $n$'); a.set_ylabel('time to reach half of the neurons'); a.set_title('(b) Spike model: spreading time',loc='left',fontsize=9)
a.text(2**14.2,7186,'ring, n.n.',color='#eb6834',fontsize=7.5,va='center'); 
for s,c in zip(sig,blues):
    d=np.array(res[f'tree_{s}']); a.text(2**16.25,d[-1,1],f'$\\sigma={s}$',color=c,fontsize=7.5,va='center')
a.set_xlim(2**5.5,2**18.3)
plt.tight_layout(); plt.savefig('fig_smallworld.pdf'); plt.savefig('fig_smallworld.png',dpi=170)
json.dump(res,open('cover.json','w'),indent=1)
for k,d in res.items():
    d=np.array(d); m=d[:,0]>=2**10; print(k,'loglog slope n>=1024: %.2f'%np.polyfit(np.log(d[m,0]),np.log(d[m,1]),1)[0])
for k in ['tree_1.0']: 
    d=[r for r in g[k]]; print(k,'L slope %.2f'%np.polyfit(np.log([r['n'] for r in d[2:]]),np.log([r['L'] for r in d[2:]]),1)[0])
