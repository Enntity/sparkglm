// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "fixture.hpp"
#define CUDA(call) do { const auto error=(call); if(error!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(error)); } while(0)
constexpr size_t GUARD=256, LIMIT=128u*1024u*1024u;
static size_t live_bytes=0,peak_bytes=0;
class Guarded {
    uint8_t* allocation=nullptr;
    size_t bytes, payload, total;
public:
    explicit Guarded(size_t n):bytes(n),payload(std::max(n,size_t(4))),total(payload+2*GUARD) {
        require(total<LIMIT && live_bytes<LIMIT-total,"live device allocations exceed128 MiB");
        CUDA(cudaMalloc(reinterpret_cast<void**>(&allocation),total));
        live_bytes+=total; peak_bytes=std::max(peak_bytes,live_bytes);
        CUDA(cudaMemset(allocation,0xa5,total));
    }
    ~Guarded() { if(allocation) { cudaFree(allocation); live_bytes-=total; } }
    Guarded(const Guarded&)=delete;
    void* data() const { return allocation+GUARD; }
    template<class T> T* as() const { return reinterpret_cast<T*>(data()); }
    void upload(const void* host) { if(bytes) CUDA(cudaMemcpy(data(),host,bytes,cudaMemcpyHostToDevice)); }
    void poison() { if(bytes) CUDA(cudaMemset(data(),0xff,bytes)); }
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> result(total);
        CUDA(cudaMemcpy(result.data(),allocation,total,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<GUARD;++i)
            require(result[i]==0xa5 && result[GUARD+payload+i]==0xa5,"allocation red zone modified");
        for(size_t i=bytes;i<payload;++i)
            require(result[GUARD+i]==0xa5,"unused payload modified");
        return result;
    }
    void unchanged(const void* original) const {
        const auto result=read();
        require(bytes==0 || std::memcmp(result.data()+GUARD,original,bytes)==0,"input bytes modified");
    }
    std::vector<float> floats() const { const auto bytes=read();std::vector<float> result(this->bytes/4);std::memcpy(result.data(),bytes.data()+GUARD,this->bytes);return result; }
    std::vector<uint16_t> output() const {
        const auto result=read(); std::vector<uint16_t> values(bytes/2);
        std::memcpy(values.data(),result.data()+GUARD,bytes); return values;
    }
};
