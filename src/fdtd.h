#pragma once

#include <array>
#include <vector>
#include <memory>
#include "common.h"
#include "typedef.h"

enum class Orientation {
    None=-1, X, Y, Z
};

enum class Field {
    None=-1, E, H
};

enum class Bound {
    None=-1, L, U
};

template<typename T, size_t Size>
class CudaArray {
public:
    __host__ __device__ T& operator[] (size_t i) {
        return this->_elems[i];
    }

    __host__ __device__ const T& operator[](size_t i) const {
        return this->_elems[i];
    }

    T _elems[Size];
};

template <typename T, size_t Dim>
using Box = CudaArray<CudaArray<T, Dim>, 2>;

class HostField {
public:
    HostField(const Field _ft, const Orientation _or, const Box<ssize_t,3> _sim_bounds);
    ~HostField() {if (this->f != nullptr) delete[] this->f;}

    real_t& getElem(const ssize_t i, const ssize_t j, const ssize_t k);
    const real_t& getElem(const ssize_t i, const ssize_t j, const ssize_t k) const;

    size_t getAllocSize() const;
    size_t getAllocSizeInDim(const Orientation _or) const;
    
    real_t* getPtr() {return &this->f[0];}
    const real_t* getPtr() const {return &this->f[0];}

    size_t getAlloc(const Bound _lu, const Orientation _or) {return this->alloc[(int)_lu][(int)_or];}
    size_t getSweep(const Bound _lu, const Orientation _or) {return this->sweep[(int)_lu][(int)_or];}

private:
    Box<ssize_t, 3> sweep {0,0,0, 0,0,0};
    Box<ssize_t, 3> alloc {0,0,0, 0,0,0};
    real_t* f {nullptr};
    Field eh {Field::None};
    Orientation xyz {Orientation::None};

    friend class DeviceField;
};


class DeviceField {
public:
    DeviceField(HostField& _hf);
    ~DeviceField() {if (this->device_f != nullptr) cudaFree(this->device_f);}
    
    __device__ DeviceField(const DeviceField& _rhs) = default;

    __device__ size_t getAllocSize() const;
    __device__ size_t getAllocSizeInDim(const Orientation _or) const;
    
    __device__ real_t& getElem(const ssize_t i, const ssize_t j, const ssize_t k);

    __device__ bool inSweep(const ssize_t i, const ssize_t j, const ssize_t k);

    __device__ size_t getAlloc(const Bound _lu, const Orientation _or) {return this->alloc[(int)_lu][(int)_or];}
    __device__ size_t getSweep(const Bound _lu, const Orientation _or) {return this->sweep[(int)_lu][(int)_or];}

    void retrieveToHost();

private:
    Box<ssize_t, 3> sweep {0,0,0, 0,0,0};
    Box<ssize_t, 3> alloc {0,0,0, 0,0,0};
    real_t *device_f {nullptr};
    HostField *host_f {nullptr};
    Field eh {Field::None};
    Orientation xyz {Orientation::None};
};

void runFDTDTest();
