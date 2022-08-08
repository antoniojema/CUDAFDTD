#pragma once

#include "CudaVector.h"
#include "common.h"
#include "cuda_utils.h"

template <typename T>
inline CudaVector<T>::CudaVector(const size_t n){
    this->alloc(n);
}

template <typename T>
inline CudaVector<T>::CudaVector(const size_t n, T*const src){
    this->alloc(n);
    this->fill(src);
}

template <typename T>
inline CudaVector<T>::~CudaVector(){
    this->del();
}

template <typename T>
inline void CudaVector<T>::alloc(const size_t n){
    if (this->ptr == nullptr) {
        this->ptr = newCUDA<T>(n);
        this->size = n;
    }
}

template <typename T>
inline void CudaVector<T>::del(){
    if (this->ptr != nullptr) {
        cudaFree(this->ptr);
        this->ptr = nullptr;
        this->size = 0;
    }
}

template <typename T>
inline void CudaVector<T>::fill(const T*const src){
    if (this->ptr != nullptr) {
        copyToDeviceCUDA(this->ptr, src, this->size);
    }
}

template <typename T>
inline void CudaVector<T>::retrieve(T*const dst){
    if (this->ptr != nullptr) {
        copyToHostCUDA(dst, this->ptr, this->size);
    }
}

template <typename T>
inline T* CudaVector<T>::getPtr(){
    return this->ptr;
}

template <typename T>
inline bool CudaVector<T>::isAllocated(){
    return !(this->ptr == nullptr);
}

