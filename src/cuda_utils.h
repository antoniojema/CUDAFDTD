#pragma once

#include "common.h"

template<typename T>
inline T* newCUDA(const size_t n) {
    void *ptr;
    cudaMalloc(&ptr, n * sizeof(T));
    return (T*) ptr;
}

template<typename T>
inline void copyToDeviceCUDA(T* const dst, const T* const src, const size_t n) {
    cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void copyToHostCUDA(T*const dst, const T*const src, const size_t n) {
    cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost);
}