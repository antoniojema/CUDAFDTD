#pragma once

#include <array>
#include <vector>
#include <memory>
#include "common.h"
#include "typedef.h"
#include "NDArray.h"

enum class Orientation {
    None=-1, X, Y, Z
};

enum class FieldType {
    None=-1, E, H
};

enum class Bound {
    None=-1, L, U
};

template <typename T>
class CudaArray3D {
public:
    using value_type = T;
    constexpr static size_t Dim = 3;

    CudaArray3D() = default;

    CudaArray3D(const std::array<size_t, Dim> _size) {this->set(_size);}

    template <typename VecType>
    CudaArray3D(const std::array<size_t, Dim> _dim, VecType& _src) {this->setAndCopy(_dim, _src);}

    ~CudaArray3D() {this->del();}

    T** getDevicePtr() {return this->ptr_device_lvl2;}

    void set(const std::array<size_t, Dim> _dim);

    template <typename VecType>
    void copy(const VecType& _src);
    
    template <typename VecType>
    void setAndCopy(const std::array<size_t, Dim> _dim, VecType& _src);
    
    template <typename VecType>
    void retrieve(VecType& _dst);

    void del();

private:
    std::array<size_t, 2> dim {0,0};
    std::vector<std::vector<T*>> ptr_device_lvl0 {};
    std::vector<T**> ptr_device_lvl1 {};
    T ***ptr_device_lvl2 {};
};


class Field : NDArray<real_t, 3> {
public:
    Field(const FieldType _ft, const Orientation _or, const Box<ssize_t,3> _sim_bounds);

    real_t& getElem(const ssize_t i, const ssize_t j, const ssize_t k);
    const real_t& getElem(const ssize_t i, const ssize_t j, const ssize_t k) const;

    size_t getAllocSize(const Orientation _xyz) const;
    
    real_t* getPtr() {return &this->f[0][0][0];}
    const real_t* getPtr() const {return &this->f[0][0][0];}

    size_t getAlloc(const Bound _lu, const Orientation _or) {return this->alloc[(int)_lu][(int)_or];}
    size_t getSweep(const Bound _lu, const Orientation _or) {return this->sweep[(int)_lu][(int)_or];}

private:
    Box<ssize_t, 3> sweep {{0,0,0}, {0,0,0}};
    Box<ssize_t, 3> alloc {{0,0,0}, {0,0,0}};
    NDArray<real_t, 3> f {};
    FieldType eh {FieldType::None};
    Orientation xyz {Orientation::None};

    friend class DeviceField;
};


class DeviceField {
public:
    DeviceField(Field& _hf);
    ~DeviceField() {if (this->device_f != nullptr) cudaFree(this->device_f);}

    __device__ size_t getAllocSize() const;
    __device__ size_t getAllocSizeInDim(const Orientation _or) const;
    
    __device__ real_t& getElem(const ssize_t i, const ssize_t j, const ssize_t k);

    __device__ bool inSweep(const ssize_t i, const ssize_t j, const ssize_t k);

    __device__ size_t getAlloc(const Bound _lu, const Orientation _or) {return this->alloc[(int)_lu][(int)_or];}
    __device__ size_t getSweep(const Bound _lu, const Orientation _or) {return this->sweep[(int)_lu][(int)_or];}

    void retrieveToHost();

private:
    Box<ssize_t, 3> sweep {{0,0,0}, {0,0,0}};
    Box<ssize_t, 3> alloc {{0,0,0}, {0,0,0}};
    real_t *device_f {nullptr};
    Field *host_f {nullptr};
    FieldType eh {FieldType::None};
    Orientation xyz {Orientation::None};
};

void runFDTDTest();
