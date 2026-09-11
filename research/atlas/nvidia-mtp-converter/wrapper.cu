// SPDX-License-Identifier: AGPL-3.0-only
// Calls the vendored Atlas kernel with the same grids and scalar math as
// weight_map/loaders_fp8.rs::quantize_to_nvfp4. Never retains source tensors.
#include <cuda_runtime.h>
#include <cmath>
#include "quantize_bf16_to_nvfp4.cu"
extern "C" const char* atlas_error(int code) { return cudaGetErrorString((cudaError_t)code); }
extern "C" int atlas_quantize(const void* source, void* packed, void* scales,
                               float* scale2, unsigned int n, unsigned int k) {
    if (!n || !k || k % 16 || (unsigned long long)n*k > 0xffffffffULL)
        return (int)cudaErrorInvalidValue;
    void *in=nullptr, *out=nullptr, *sf=nullptr, *mx=nullptr;
    cudaError_t status = cudaSuccess;
    float maximum=0.0f;
    const size_t total=(size_t)n*k;
    const unsigned int blocks=(unsigned int)((total/256 < 1) ? 1 : (total/256 > 1024 ? 1024 : total/256));
#define CHECK(call) do { status=(call); if(status!=cudaSuccess) goto cleanup; } while(0)
    CHECK(cudaMalloc(&in,total*2)); CHECK(cudaMalloc(&out,total/2));
    CHECK(cudaMalloc(&sf,total/16)); CHECK(cudaMalloc(&mx,4));
    CHECK(cudaMemcpy(in,source,total*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(mx,0,4));
    nvfp4_global_absmax<<<blocks,256>>>((const __nv_bfloat16*)in,(float*)mx,(unsigned int)total);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&maximum,mx,4,cudaMemcpyDeviceToHost));
    if (!std::isfinite(maximum)) { status=cudaErrorInvalidValue; goto cleanup; }
    *scale2=maximum > 0.0f ? maximum/(6.0f*448.0f) : 1.0f;
    quantize_bf16_to_nvfp4<<<n,256>>>((const __nv_bfloat16*)in,(unsigned char*)out,(unsigned char*)sf,*scale2,n,k);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(packed,out,total/2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(scales,sf,total/16,cudaMemcpyDeviceToHost));
cleanup:
    if(in) cudaFree(in); if(out) cudaFree(out); if(sf) cudaFree(sf); if(mx) cudaFree(mx);
    return (int)status;
}
