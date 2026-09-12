// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
extern "C" int atlas_mango_flash_prefill(const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,void*,uint64_t,uint64_t,uint64_t,int,int,float,float,void*);
int main() {
 const int t=2048,h=32; const size_t plane=t*h*128*2, state=h*128*128*4;
 const size_t sizes[]={3*plane,plane,t*h*2,h*4,h*128*4,state,plane,3*plane,57065600,((state+t*h*2+127)/128)*128+20};
 void* p[10];
 for(int i=0;i<10;++i){ auto e=cudaMalloc(&p[i],sizes[i]);if(e){printf("alloc %d\n",e);return 1;}cudaMemset(p[i],0,sizes[i]); }
 cudaMemset(p[6],0xff,sizes[6]);
 int e=atlas_mango_flash_prefill(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],sizes[7],sizes[8],sizes[9],t,h,1.0f/sqrtf(128),-5,nullptr);
 if(e){printf("launch %d %s\n",e,cudaGetErrorString((cudaError_t)e));return 2;}
 auto sync=cudaDeviceSynchronize();if(sync){printf("sync %d %s\n",sync,cudaGetErrorString(sync));return 3;}
 auto* host=new uint16_t[plane/2];cudaMemcpy(host,p[6],plane,cudaMemcpyDeviceToHost);
 for(size_t i=0;i<plane/2;i++){if(host[i]){printf("nonzero output at %zu\n",i);return 4;}}
 printf("NATIVE_FLASH_SMOKE_OK\n");for(auto x:p)cudaFree(x);delete[] host;return 0;
}
