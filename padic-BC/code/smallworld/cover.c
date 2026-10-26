// Avalanches of the effective spike model with firing-rate function
// f(s)=fmax(1-exp(-s/s0)), on (i) a p=2 tree (hierarchical, p-adic kernel),
// (ii) a 1D chain with nearest-neighbour coupling, (iii) no saturation (pure branching).
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
  int NMAX=(argc>10)?atoi(argv[10]):60000;uint64_t *pos=malloc(sizeof(uint64_t)*(NMAX+8));
  // time grid t_k = 2^(k/4)
  int NG=0;double tg[200];for(int k=0;;k++){double t=pow(2.0,k/4.0);if(t>tmax)break;tg[NG++]=t;}
  unsigned char*vis=malloc((size_t)1<<L);          // target level: stop when activity leaves the ball with 2^D neurons
  FILE*fo=fopen(out,"w");
  const uint64_t mask=(L>=64)?~0ULL:((1ULL<<L)-1);
  for(long r=0;r<nruns;r++){
    long n=1;double t=0;int flag=0;uint64_t nv=1;memset(vis,0,(size_t)1<<L);
    uint64_t x0=(mode==1)?0:(rnd()&mask);pos[0]=x0;vis[x0]=1;
    if(mode==0){for(int d=0;d<=L;d++)hadd(((uint64_t)d<<48)|(pos[0]>>d));}
    else if(mode==1)hadd(pos[0]);
    while(n>0){
      t+=-log(1.0-U())/(n*(1.0+b));
      if(t>tmax){flag=1;break;}
      if(n>=NMAX||hn>0.55*HS){flag=2;break;}
      long i=(long)(U()*n);uint64_t x=pos[i];
      if(U()*(1.0+b)<1.0){
        if(mode==0){for(int d=0;d<=L;d++)hsub(((uint64_t)d<<48)|(x>>d));}
        else if(mode==1)hsub(x);
        pos[i]=pos[--n];
      }else{
        uint64_t y;double acc=1.0;int dy;
        if(mode==0){
          double u=U();int d=1;while(d<L&&cum[d]<u)d++;
          uint64_t low=(d>1)?(rnd()&((1ULL<<(d-1))-1)):0;
          y=((x>>d)<<d)|((((x>>(d-1))&1)^1ULL)<<(d-1))|low;
          double sy=0;int32_t cprev=hget(y);
          for(int dd=1;dd<=L;dd++){int32_t cd=hget(((uint64_t)dd<<48)|(y>>dd));sy+=wd[dd]*(cd-cprev);cprev=cd;if(cd==n)break;}
          double z=sy/s0;acc=(z<1e-12)?1.0:(-expm1(-z)/z);
          (void)dy;     // y lies outside the ball of 2^(dy-1) neurons around the seed
        }else{
          y=((U()<0.5)?x+1:x-1)&mask;
          double sy=0.5*(hget((y-1)&mask)+hget((y+1)&mask));double z=sy/s0;acc=(z<1e-12)?1.0:(-expm1(-z)/z);
          (void)dy;  // |y-x0|>=2^(dy-1)
        }
        if(U()<acc){
          if(mode==0){for(int d=0;d<=L;d++)hadd(((uint64_t)d<<48)|(y>>d));}
          else hadd(y);
          pos[n++]=y;
          if(!vis[y]){vis[y]=1;nv++;if(2*nv>=((uint64_t)1<<L)){flag=3;break;}}
        }
      }
    }
    hclear();
    if(flag==3){fprintf(fo,"%g %ld\n",t,n);fflush(fo);}
    else if(flag)fprintf(stderr,"run %ld ended flag %d t=%g n=%ld\n",r,flag,t,n);
  }
  fclose(fo);
  return 0;
}
