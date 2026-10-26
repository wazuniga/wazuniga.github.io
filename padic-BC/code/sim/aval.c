// Avalanches of the effective spike model with firing-rate function
// f(s)=fmax(1-exp(-s/s0)), on (i) a p=2 tree (hierarchical, p-adic kernel),
// (ii) a 1D chain with nearest-neighbour coupling, (iii) no saturation (pure branching),
// (iv) mode 3: a 1D chain with power-law coupling w(r)=r^-(1+sigma)/(2 zeta(1+sigma)), r>=1.
// Exact continuous-time simulation by thinning: f(s) <= f'(0) s.
// Units: alpha=1 (decay rate), w0 = sum_j w_ij = 1.  b = f'(0) w0 / alpha.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
static uint64_t s[4];
static inline uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static inline uint64_t rnd(void){uint64_t r=rotl(s[1]*5,7)*9,t=s[1]<<17;s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return r;}
static inline double U(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static void seedrng(uint64_t x){for(int i=0;i<4;i++){x+=0x9e3779b97f4a7c15ULL;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;s[i]=z^(z>>31);}}
// ---- hash: key -> count, linear probing, backward-shift deletion
#define HB 23
#define HS (1u<<HB)
static uint64_t *hk; static int32_t *hv; static uint32_t hn=0;
static inline uint32_t hsh(uint64_t k){k^=k>>33;k*=0xff51afd7ed558ccdULL;k^=k>>33;k*=0xc4ceb9fe1a85ec53ULL;k^=k>>33;return (uint32_t)(k&(HS-1));}
static inline int32_t hget(uint64_t k){uint32_t i=hsh(k);while(hv[i]){if(hk[i]==k)return hv[i];i=(i+1)&(HS-1);}return 0;}
static inline void hadd(uint64_t k){uint32_t i=hsh(k);while(hv[i]){if(hk[i]==k){hv[i]++;return;}i=(i+1)&(HS-1);}hk[i]=k;hv[i]=1;hn++;}
static inline void hsub(uint64_t k){uint32_t i=hsh(k);while(hk[i]!=k||!hv[i])i=(i+1)&(HS-1);
  if(--hv[i])return; hn--; uint32_t j=i;
  for(;;){j=(j+1)&(HS-1);if(!hv[j])break;uint32_t h=hsh(hk[j]);
    if(((j-h)&(HS-1))>=((j-i)&(HS-1))){hk[i]=hk[j];hv[i]=hv[j];hv[j]=0;i=j;}}
}
static void hclear(void){memset(hv,0,sizeof(int32_t)*HS);hn=0;}
int main(int argc,char**argv){
  if(argc<10){fprintf(stderr,"usage: mode(0 tree,1 chain,2 branching) sigma b s0 nruns tmax seed L outfile\n");return 1;}
  int mode=atoi(argv[1]);double sigma=atof(argv[2]),b=atof(argv[3]),s0=atof(argv[4]);
  long nruns=atol(argv[5]);double tmax=atof(argv[6]);seedrng(strtoull(argv[7],0,10));int L=atoi(argv[8]);
  const char*out=argv[9];
  hk=malloc(sizeof(uint64_t)*HS);hv=calloc(HS,sizeof(int32_t));
  double wd[64],cum[64],tot=0;                 // w(d): coupling at tree distance d ; sphere size 2^(d-1)
  for(int d=1;d<=L;d++){wd[d]=pow(2.0,-d*(1.0+sigma));tot+=ldexp(1.0,d-1)*wd[d];}
  double c=0;for(int d=1;d<=L;d++){wd[d]/=tot;c+=ldexp(1.0,d-1)*wd[d];cum[d]=c;}
  // ---- mode 3: power-law chain. Jump table up to R0, continuous Pareto tail beyond.
  #define R0 (1L<<20)
  #define WT (1L<<16)
  #define RC 128
  long NDIR=getenv("NDIR")?atol(getenv("NDIR")):300;
  double *lc=0,*wt=0,Z=0,ctab=0;
  if(mode==3){
    lc=malloc(sizeof(double)*(R0+1));wt=malloc(sizeof(double)*(WT+1));double c2=0;
    for(long r=1;r<=R0;r++){c2+=pow((double)r,-(1.0+sigma));lc[r]=c2;}
    ctab=c2;Z=c2+pow((double)R0,-sigma)/sigma-0.5*pow((double)R0,-1.0-sigma)+(1.0+sigma)*pow((double)R0,-2.0-sigma)/12.0;
    for(long r=1;r<=WT;r++)wt[r]=pow((double)r,-(1.0+sigma))/(2.0*Z);
  }
  int NMAX=(argc>10)?atoi(argv[10]):60000;uint64_t *pos=malloc(sizeof(uint64_t)*(NMAX+8));
  // time grid t_k = 2^(k/4)
  double gstep=getenv("FINE")?8.0:4.0;int NG=0;double tg[400];for(int k=0;;k++){double t=pow(2.0,k/gstep);if(t>tmax)break;tg[NG++]=t;}
  double *sumD=calloc(400,sizeof(double));
  double *sumN=calloc(NG,sizeof(double));long *surv=calloc(NG,sizeof(long));
  FILE*fo=fopen(out,"wb");
  const uint64_t mask=(L>=64)?~0ULL:((1ULL<<L)-1);
  for(long r=0;r<nruns;r++){
    long n=1;double t=0,S=1;int g=0,flag=0;
    pos[0]=(mode==1)?(1ULL<<40):(mode==3)?(1ULL<<61):(rnd()&mask);uint64_t pos0=pos[0];
    if(mode==0){for(int d=0;d<=L;d++)hadd(((uint64_t)d<<48)|(pos[0]>>d));}
    else if(mode==1||mode==3)hadd(pos[0]);
    while(n>0){
      t+=-log(1.0-U())/(n*(1.0+b));
      while(g<NG&&tg[g]<t){sumN[g]+=n;surv[g]++;
        if(mode==3&&gstep>4.0){for(long j=0;j<n;j++){uint64_t dd=(pos[j]>pos0)?pos[j]-pos0:pos0-pos[j];sumD[g]+=log(dd?(double)dd:1.0);}}
        g++;}
      if(t>tmax){flag=1;break;}
      if(n>=NMAX||hn>0.55*HS){flag=2;while(g<NG){sumN[g]+=n;surv[g]++;g++;}break;}
      long i=(long)(U()*n);uint64_t x=pos[i];
      if(U()*(1.0+b)<1.0){                      // decay
        if(mode==0){for(int d=0;d<=L;d++)hsub(((uint64_t)d<<48)|(x>>d));}
        else if(mode==1||mode==3)hsub(x);
        pos[i]=pos[--n];
      }else{                                    // attempted emission
        uint64_t y;double acc=1.0;
        if(mode==0){
          double u=U();int d=1;while(d<L&&cum[d]<u)d++;
          uint64_t low=(d>1)?(rnd()&((1ULL<<(d-1))-1)):0;
          y=((x>>d)<<d)|((((x>>(d-1))&1)^1ULL)<<(d-1))|low;
          double sy=0;int32_t cprev=hget(y);       // d=0: spikes on y itself (no self coupling)
          for(int dd=1;dd<=L;dd++){int32_t cd=hget(((uint64_t)dd<<48)|(y>>dd));sy+=wd[dd]*(cd-cprev);cprev=cd;if(cd==n)break;}
          double z=sy/s0;acc=(z<1e-12)?1.0:(-expm1(-z)/z);
        }else if(mode==1){
          y=(U()<0.5)?x+1:x-1;
          double sy=0.5*(hget(y-1)+hget(y+1));double z=sy/s0;acc=(z<1e-12)?1.0:(-expm1(-z)/z);
        }else if(mode==3){
          double u=U()*Z;uint64_t r;
          if(u<ctab){long lo=1,hi=R0;while(lo<hi){long m=(lo+hi)/2;if(lc[m]<u)lo=m+1;else hi=m;}r=lo;}
          else{double q=R0*pow(1.0-U(),-1.0/sigma);r=(q>1.1e15)?(uint64_t)1.1e15:(uint64_t)q+1;}
          y=(rnd()&1)?x+r:x-r;
          if(n<=NDIR){   // direct sum of the input
            double sy=0;
            for(long j=0;j<n;j++){uint64_t dd=(pos[j]>y)?pos[j]-y:y-pos[j];
              if(dd==0)continue; sy+=(dd<=WT)?wt[dd]:pow((double)dd,-(1.0+sigma))/(2.0*Z);}
            double z=sy/s0;acc=(z<1e-12)?1.0:(-expm1(-z)/z);
          }else{
            // Exact alternative for large n.  f(s)/(f'(0)s)=(1-e^{-z})/z=E_v[exp(-v z)], v~U(0,1),
            // and exp(-v z)=prod_j exp(-v w_j/s0): each spike vetoes independently (Poisson).
            // Near spikes (r<=RC): exact product from the occupancy hash.  Far spikes: Poisson
            // thinning with the bound w_j<=w(RC+1).
            double v=U(),sn=0;
            for(long r=1;r<=RC;r++){int32_t c=hget(y+r)+hget(y-r);if(c)sn+=wt[r]*c;}
            acc=exp(-v*sn/s0);
            if(acc>0){
              double lam=v*wt[RC+1]/s0,mean=n*lam;long K=0;
              double Lp=exp(-mean),pp=1.0;for(;;){pp*=U();if(pp<=Lp)break;K++;}   // Poisson(mean), mean small
              for(long k=0;k<K&&acc>0;k++){long j=(long)(U()*n);uint64_t dd=(pos[j]>y)?pos[j]-y:y-pos[j];
                if(dd<=RC)continue;
                double wj=(dd<=WT)?wt[dd]:pow((double)dd,-(1.0+sigma))/(2.0*Z);
                if(U()*wt[RC+1]<wj)acc=0;}
            }
          }
        }else y=0;
        if(U()<acc){
          if(mode==0){for(int d=0;d<=L;d++)hadd(((uint64_t)d<<48)|(y>>d));}
          else if(mode==1||mode==3)hadd(y);
          pos[n++]=y;S+=1;
        }
      }
    }
    if(flag){ if(mode<2)hclear(); else if(mode==3){for(long j=0;j<n;j++)hsub(pos[j]);} }
    float rec[3]={(float)S,(float)t,(float)flag};fwrite(rec,sizeof(float),3,fo);
  }
  fclose(fo);
  for(int k=0;k<NG;k++){if(gstep>4.0)printf("%g %g %g %g\n",tg[k],sumN[k]/nruns,(double)surv[k]/nruns,sumD[k]/(sumN[k]>0?sumN[k]:1));else printf("%g %g %g\n",tg[k],sumN[k]/nruns,(double)surv[k]/nruns);}
  return 0;
}
