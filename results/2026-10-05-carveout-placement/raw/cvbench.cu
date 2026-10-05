// cvbench: GB10 display carveout vs cudaMalloc memory, same tests on both.
// Run under `spark display-carveout --lock FILE -- cvbench` so the carveout fd
// arrives in ATLAS_DISPLAY_CARVEOUT_FD / _SIZE, imported exactly as Atlas does.
//   stream : grid-stride 16 B loads over 1.5 GiB            -> GB/s
//   reuse  : 4 MiB region read 256 times with ld.cg (L2)    -> GB/s (L2-cacheable?)
//   random : 16 B ld.cg at hashed 128 B slots over a span   -> M accesses/s (TLB reach)
//   chase  : one thread, random permutation, 64 KiB stride  -> ns per hop (miss latency)
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include <unistd.h>

#define CK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA %s:%d %s\n", #x, __LINE__, cudaGetErrorString(r_)); exit(1); } } while (0)
#define CD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_; cuGetErrorString(r_, &s_); printf("CU %s:%d %s\n", #x, __LINE__, s_); exit(1); } } while (0)

static const size_t SPAN = 1536ull << 20;

__global__ void stream_read(const uint4* p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) *out = acc;
}

__global__ void reuse_read(const uint4* p, size_t n, int passes, unsigned* out) {
    unsigned acc = 0;
    for (int pass = 0; pass < passes; ++pass)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
            uint4 v = __ldcg(p + i);
            acc ^= v.x ^ v.y ^ v.z ^ v.w ^ pass;
        }
    if (acc == 0x12345678u) *out = acc;
}

__device__ __forceinline__ uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}

__global__ void random_read(const char* p, size_t slots, int per_thread, unsigned* out) {
    unsigned acc = 0;
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    for (int j = 0; j < per_thread; ++j) {
        uint64_t s = mix(t * 1315423911ull + j) % slots;
        uint4 v = __ldcg(reinterpret_cast<const uint4*>(p + s * 128));
        acc ^= v.x ^ v.w;
    }
    if (acc == 0x12345678u) *out = acc;
}

__global__ void build_chain(char* p, const uint32_t* perm, size_t nodes, size_t stride) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nodes; i += (size_t)gridDim.x * blockDim.x) {
        size_t from = perm[i], to = perm[(i + 1) % nodes];
        *reinterpret_cast<uint64_t*>(p + from * stride) = reinterpret_cast<uint64_t>(p + to * stride);
    }
}

__global__ void chase(const char* start, long hops, uint64_t* out) {
    const uint64_t* q = reinterpret_cast<const uint64_t*>(start);
    for (long i = 0; i < hops; ++i) q = reinterpret_cast<const uint64_t*>(__ldcg(q));
    *out = reinterpret_cast<uint64_t>(q);
}

template <class F>
static float best_ms(F launch, int reps = 3) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch(); CK(cudaDeviceSynchronize());  // warm
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); if (ms < best) best = ms;
    }
    return best;
}

static void run(const char* name, char* p, unsigned* out, uint64_t* out64, int sms) {
    int grid = sms * 8, block = 256;
    float ms = best_ms([&] { stream_read<<<grid, block>>>((const uint4*)p, SPAN / 16, out); });
    printf("%-10s stream            %8.1f GB/s\n", name, SPAN / ms / 1e6);
    const size_t region = 4ull << 20; const int passes = 256;
    ms = best_ms([&] { reuse_read<<<grid, block>>>((const uint4*)p, region / 16, passes, out); });
    printf("%-10s reuse 4 MiB (L2)  %8.1f GB/s\n", name, region * (double)passes / ms / 1e6);
    for (size_t span : {size_t(16) << 20, size_t(256) << 20, SPAN}) {
        const int per = 64; const int g = sms * 16;
        ms = best_ms([&] { random_read<<<g, block>>>(p, span / 128, per, out); });
        double acc = (double)g * block * per;
        printf("%-10s random %5zu MiB    %8.1f M acc/s\n", name, span >> 20, acc / ms / 1e3);
    }
    for (size_t span : {size_t(16) << 20, size_t(256) << 20, SPAN}) {
        const size_t stride = 64ull << 10, nodes = span / stride;
        std::vector<uint32_t> perm(nodes); std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), std::mt19937(42));
        uint32_t* dperm; CK(cudaMalloc(&dperm, nodes * 4)); CK(cudaMemcpy(dperm, perm.data(), nodes * 4, cudaMemcpyHostToDevice));
        build_chain<<<64, 256>>>(p, dperm, nodes, stride); CK(cudaDeviceSynchronize()); CK(cudaFree(dperm));
        const long hops = 20000;
        ms = best_ms([&] { chase<<<1, 1>>>(p + (size_t)perm[0] * stride, hops, out64); });
        printf("%-10s chase  %5zu MiB    %8.1f ns/hop\n", name, span >> 20, ms * 1e6 / hops);
    }
}

int main() {
    CK(cudaFree(0));
    int dev = 0, sms = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    unsigned* out; uint64_t* out64; CK(cudaMalloc(&out, 4)); CK(cudaMalloc(&out64, 8));

    char* plain; CK(cudaMalloc(&plain, SPAN)); CK(cudaMemset(plain, 1, SPAN));

    const char* fd_s = getenv("ATLAS_DISPLAY_CARVEOUT_FD");
    const char* size_s = getenv("ATLAS_DISPLAY_CARVEOUT_SIZE");
    char* carve = nullptr;
    if (fd_s && size_s) {
        int fd = atoi(fd_s); size_t size = strtoull(size_s, nullptr, 10);
        CUmemGenericAllocationHandle h;
        CD(cuMemImportFromShareableHandle(&h, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        close(fd);
        CUmemAllocationProp prop = {};
        if (cuMemGetAllocationPropertiesFromHandle(&prop, h) == CUDA_SUCCESS)
            printf("carveout props: type %d location %d/%d handleTypes %d compression %d gpuDirectRDMA %d usage %d\n",
                   (int)prop.type, (int)prop.location.type, prop.location.id, (int)prop.requestedHandleTypes,
                   (int)prop.allocFlags.compressionType, (int)prop.allocFlags.gpuDirectRDMACapable, (int)prop.allocFlags.usage);
        size_t gran = 0; CUmemAllocationProp ask = {}; ask.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ask.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ask.location.id = dev;
        if (cuMemGetAllocationGranularity(&gran, &ask, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS)
            printf("device recommended granularity %zu KiB\n", gran >> 10);
        CUdeviceptr va;
        CD(cuMemAddressReserve(&va, size, 2 << 20, 0, 0));
        CD(cuMemMap(va, size, 0, h, 0));
        CD(cuMemRelease(h));
        CUmemAccessDesc access = {}; access.location.type = CU_MEM_LOCATION_TYPE_DEVICE; access.location.id = dev;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CD(cuMemSetAccess(va, size, &access, 1));
        carve = (char*)va; CK(cudaMemset(carve, 1, SPAN));
        printf("carveout %zu MiB mapped at %p\n", size >> 20, carve);
    } else {
        printf("no carveout in the environment; cudaMalloc only\n");
    }
    for (int round = 0; round < 2; ++round) {
        printf("--- round %d\n", round + 1);
        run("cudaMalloc", plain, out, out64, sms);
        if (carve) run("carveout", carve, out, out64, sms);
    }
    return 0;
}
