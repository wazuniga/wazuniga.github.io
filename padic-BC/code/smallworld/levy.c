// Spreading time of the effective spike model on a Euclidean ring of n=2^L neurons with
// power-law coupling w(r) ~ r^-(1+sigma) (ring distance r), total coupling 1.
// Same rules as cover.c: decay rate 1, emission attempts at rate b per spike, accepted
// with probability f(s)/(f'(0)s), f(s)=fmax(1-exp(-s/s0)).  Output: time until half of
// the neurons have fired at least once, and number of spikes present at that time.
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
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: sigma b s0 nruns seed L outfile\n");return 1;}
  double sigma=atof(argv[1]),b=atof(argv[2]),s0=atof(argv[3]);long nruns=atol(argv[4]);
  seedrng(strtoull(argv[5],0,10));int L=atoi(argv[6]);long N=1L<<L,R=N/2-1;
  double *w=malloc(sizeof(double)*(R+1)),*cum=malloc(sizeof(double)*(R+1)),tot=0;
  for(long r=1;r<=R;r++){w[r]=pow((double)r,-(1.0+sigma));tot+=2*w[r];}
  double c=0;for(long r=1;r<=R;r++){w[r]/=tot;c+=2*w[r];cum[r]=c;}
  int *occ=calloc(N,sizeof(int));unsigned char*vis=malloc(N);
  long NMAX=20*N;long *pos=malloc(sizeof(long)*NMAX);
  FILE*fo=fopen(argv[7],"w");
  for(long run=0;run<nruns;run++){
    memset(occ,0,sizeof(int)*N);memset(vis,0,N);
    long n=1,nv=1;double t=0;int ok=0;pos[0]=0;occ[0]=1;vis[0]=1;
    while(n>0&&n<NMAX-1){
      t+=-log(1.0-U())/(n*(1.0+b));
      long i=(long)(U()*n),x=pos[i];
      if(U()*(1.0+b)<1.0){occ[x]--;pos[i]=pos[--n];continue;}
      double u=U();long lo=1,hi=R;while(lo<hi){long m=(lo+hi)/2;if(cum[m]<u)lo=m+1;else hi=m;}
      long y=(rnd()&1)?(x+lo)&(N-1):(x-lo)&(N-1);
      double sy=0;for(long r=1;r<=R;r++)sy+=w[r]*(occ[(y+r)&(N-1)]+occ[(y-r)&(N-1)]);
      double z=sy/s0,acc=(z<1e-12)?1.0:(-expm1(-z)/z);
      if(U()<acc){occ[y]++;pos[n++]=y;if(!vis[y]){vis[y]=1;if(2*(++nv)>=N){ok=1;break;}}}
    }
    if(ok){fprintf(fo,"%g %ld\n",t,n);fflush(fo);}
  }
  fclose(fo);return 0;
}
