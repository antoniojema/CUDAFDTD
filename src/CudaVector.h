#pragma once

#include "common.h"
#include "typedef.h"
#include <array>

template <typename T>
class CudaVector {
public:
    using type = T;
public:
    CudaVector() = default;
    CudaVector(const size_t n);
    CudaVector(const size_t n, T*const src);
    ~CudaVector();

    void alloc(const size_t n);
    void del();
    void fill(const T*const src);
    void retrieve(T*const dst);
    T* getPtr();
    bool isAllocated();

private:
    T *ptr {nullptr};
    size_t size {0};
};

#include "CudaVector_impl.h"
