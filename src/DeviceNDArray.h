#pragma once

#include "common.h"
#include "typedef.h"
#include "box.h"
#include <array>
#include <vector>


template <typename T, size_t Lvl>
struct Ptr {
    using type = Ptr<T,Lvl-1>::type*;
};

template <typename T>
struct Ptr<T,0> {
    using type = T;
};


template <typename T, size_t Dim>
class DeviceNDArray {
public:
    using type = T;
    using elem_type = DeviceNDArray<T, Dim-1>;
    
    DeviceNDArray() = default;
    DeviceNDArray(const Box<ssize_t,Dim>& _lims) {this->set(_lims);}
    ~DeviceNDArray();
    
    void set(const Box<ssize_t,Dim>& _lims);
    void free();
    
    template<typename VecType> void copy(const VecType& _vec);
    template<typename VecType> void copyFromZero(const VecType& _vec);
    
    template<typename VecType> void retrieve(VecType& _vec);
    template<typename VecType> void retrieveToZero(VecType& _vec);

    Ptr<T,Dim>::type getDeviceMoved();

private:
    std::array<size_t,Dim> _size {0};
    std::array<ssize_t,Dim> _index0 {0};
    std::vector<elem_type> _elems {};
    Ptr<T,Dim>::type _device_ptr;
};


template <typename T>
class DeviceNDArray<T,1> {
public:
    using type = T;
    using elem_type = T;

    DeviceNDArray() = default;
    DeviceNDArray(const Box<ssize_t,1>& _lims) {this->set(_lims);}
    ~DeviceNDArray();

    void set(const Box<ssize_t,1>& _lims);
    void free();
    
    template<typename VecType> void copy(const VecType& _vec);
    template<typename VecType> void copyFromZero(const VecType& _vec);
    
    template<typename VecType> void retrieve(VecType& _vec);
    template<typename VecType> void retrieveToZero(VecType& _vec);

    T* getDeviceMoved();

private:
    size_t _size {0};
    ssize_t _index0 {0};
    T *_device_ptr {nullptr};
};

#include "DeviceNDArray_impl.h"