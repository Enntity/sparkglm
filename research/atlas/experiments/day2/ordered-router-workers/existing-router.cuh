// SPDX-License-Identifier: AGPL-3.0-only
// Harness-only fixed-shape adapter around the unchanged included production kernel.
extern "C" cudaError_t launch_ordered_bf16_3x288(
    const __nv_bfloat16* A,const __nv_bfloat16* B,__nv_bfloat16* C,cudaStream_t stream){
    if(!A||!B||!C)return cudaErrorInvalidValue;
    dense_gemm_bf16_router<<<dim3(5,1,1),dim3(16,16,1),0,stream>>>(A,B,C,3,288,4096);
    return cudaGetLastError();
}
