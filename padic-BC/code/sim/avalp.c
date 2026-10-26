// General-p tree version of aval.c (mode 0).  Avalanches of the effective spike model with firing-rate function
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
  // Tree with branching p (the ball of level d has p^d neurons), coupling w ~ p^{-d(1+sigma)}.
  if(argc<10){fprintf(stderr,"usage: p sigma b s0 nruns tmax seed L outfile [NMAX]\n");return 1;}
  uint64_t p=strtoull(argv[1],0,10);double sigma=atof(argv[2]),b=atof(argv[3]),s0=atof(argv[4]);
  long nruns=atol(argv[5]);double tmax=atof(argv[6]);seedrng(strtoull(argv[7],0,10));int L=atoi(argv[8]);
  const char*out=argv[9];
  hk=malloc(sizeof(uint64_t)*HS);hv=calloc(HS,sizeof(int32_t));
  uint64_t pw[64];pw[0]=1;for(int d=1;d<=L;d++)pw[d]=pw[d-1]*p;          // p^L must fit in 2^48
  double wd[64],cum[64],tot=0;                 // sphere of level d has (p-1)p^(d-1) neurons
  for(int d=1;d<=L;d++){wd[d]=pow((double)p,-d*(1.0+sigma));tot+=(double)(p-1)*pw[d-1]*wd[d];}
  double c=0;for(int d=1;d<=L;d++){wd[d]/=tot;c+=(double)(p-1)*pw[d-1]*wd[d];cum[d]=c;}
  int NMAX=(argc>10)?atoi(argv[10]):60000;uint64_t *pos=malloc(sizeof(uint64_t)*(NMAX+8));
  int NG=0;double tg[400];for(int k=0;;k++){double t=pow(2.0,k/8.0);if(t>tmax)break;tg[NG++]=t;}  // t_k=2^(k/8)
  double *sumN=calloc(NG,sizeof(double)),*sumD=calloc(NG,sizeof(double));long *surv=calloc(NG,sizeof(long));
  FILE*fo=fopen(out,"wb");
  for(long r=0;r<nruns;r++){
    long n=1;double t=0,S=1;int g=0,flag=0;
    pos[0]=(uint64_t)(U()*pw[L]);uint64_t x0=pos[0];
    for(int d=0;d<=L;d++)hadd(((uint64_t)d<<48)|(pos[0]/pw[d]));
    while(n>0){
      t+=-log(1.0-U())/(n*(1.0+b));
      while(g<NG&&tg[g]<t){sumN[g]+=n;surv[g]++;
        for(long j=0;j<n;j++){int d=0;while(pos[j]/pw[d]!=x0/pw[d])d++;sumD[g]+=d;}   // level of the smallest ball containing spike and seed
        g++;}
      if(t>tmax){flag=1;break;}
      if(n>=NMAX||hn>0.55*HS){flag=2;while(g<NG){sumN[g]+=n;surv[g]++;g++;}break;}
      long i=(long)(U()*n);uint64_t x=pos[i];
      if(U()*(1.0+b)<1.0){
        for(int d=0;d<=L;d++)hsub(((uint64_t)d<<48)|(x/pw[d]));
        pos[i]=pos[--n];
      }else{
        double u=U();int d=1;while(d<L&&cum[d]<u)d++;
        uint64_t dig=(x/pw[d-1])%p,nd=(uint64_t)(U()*(p-1));if(nd>=dig)nd++;   // another child of the level-d ball
        uint64_t y=(x/pw[d])*pw[d]+nd*pw[d-1]+(uint64_t)(U()*pw[d-1]);
        double sy=0;int32_t cprev=hget(y);
        for(int dd=1;dd<=L;dd++){int32_t cd=hget(((uint64_t)dd<<48)|(y/pw[dd]));sy+=wd[dd]*(cd-cprev);cprev=cd;if(cd==n)break;}
        double z=sy/s0,acc=(z<1e-12)?1.0:(-expm1(-z)/z);
        if(U()<acc){for(int dd=0;dd<=L;dd++)hadd(((uint64_t)dd<<48)|(y/pw[dd]));pos[n++]=y;S+=1;}
      }
    }
    if(flag){for(long j=0;j<n;j++)for(int d=0;d<=L;d++)hsub(((uint64_t)d<<48)|(pos[j]/pw[d]));}
    float rec[3]={(float)S,(float)t,(float)flag};fwrite(rec,sizeof(float),3,fo);
  }
  fclose(fo);
  for(int k=0;k<NG;k++)printf("%g %g %g %g\n",tg[k],sumN[k]/nruns,(double)surv[k]/nruns,sumD[k]/(sumN[k]>0?sumN[k]:1));
  return 0;
}
