#pragma once

#include "DeviceNDArray.h"
#include "common.h"
#include "cuda_utils.h"

/****************************/
/*   DeviceNDArary<T,Dim>   */
/****************************/
template <typename T, size_t Dim>
DeviceNDArray<T,Dim>::~DeviceNDArray() {
    free();
}

template <typename T, size_t Dim>
void DeviceNDArray<T, Dim>::set(const Box<ssize_t,Dim>& _lims) {
    free();
    for (size_t i = 0; i < Dim; i++) {
        _size[i] = _lims[1][i] - _lims[0][i];
        _index0[i] = _lims[0][i];
    }
    _elems.resize(_size[0]);
    _elems.shrink_to_fit();
    
    Box<ssize_t,Dim-1> _lims_down;
    for (size_t i = 0; i < Dim-1; i++) {
        _lims_down[0][i] = _lims[0][i+1];
        _lims_down[1][i] = _lims[1][i+1];
    }
    for (auto& e : _elems) {
        e.set(_lims_down);
    }
}

template <typename T, size_t Dim>
void DeviceNDArray<T, Dim>::free() {
    _elems.clear();
    if (_device_ptr != nullptr) {
        cudaFree(_device_ptr);
        _device_ptr = nullptr;
    }
}

template <typename T, size_t Dim>
template <typename VecType>
void DeviceNDArray<T, Dim>::copy(const VecType& _vec) {
    for (size_t i = 0; i < _size[0]; i++) {
        _elems[i].copy(_vec[(ssize_t) i + _index0[0]]);
    }
}

template <typename T, size_t Dim>
template <typename VecType>
void DeviceNDArray<T, Dim>::copyFromZero(const VecType& _vec) {
    for (size_t i = 0; i < _size[0]; i++) {
        _elems[i].copyFromZero(_vec[i]);
    }
}

template <typename T, size_t Dim>
template <typename VecType>
void DeviceNDArray<T, Dim>::retrieve(VecType& _vec) const {
    for (size_t i = 0; i < _size[0]; i++) {
        _elems[i].retrieve(_vec[(ssize_t) i + _index0[0]]);
    }
}

template <typename T, size_t Dim>
template <typename VecType>
void DeviceNDArray<T, Dim>::retrieveToZero(VecType& _vec) const {
    for (size_t i = 0; i < _size[0]; i++) {
        _elems[i].retrieveToZero(_vec[i]);
    }
}

template <typename T, size_t Dim>
Ptr<T,Dim>::type DeviceNDArray<T, Dim>::getDeviceMoved() {
    if (_device_ptr == nullptr) {
        std::vector<typename Ptr<T,Dim-1>::type> lower_moved_ptrs {_size[0]};
        for (size_t i = 0; i < _size[0]; i++) {
            lower_moved_ptrs[i] = _elems[i].getDeviceMoved();
        }
        _device_ptr = newCUDA<Ptr<T,Dim-1>::type>(_size[0]);
        copyToDeviceCUDA(_device_ptr, &lower_moved_ptrs[0], _size[0]);
    }
    return _device_ptr - _index0[0];
}



/**************************/
/*   DeviceNDArray<T,1>   */
/**************************/
template <typename T>
DeviceNDArray<T,1>::~DeviceNDArray() {
    free();
}

template <typename T>
void DeviceNDArray<T, 1>::set(const Box<ssize_t,1>& _lims) {
    free();
    _size = _lims[1][0] - _lims[0][0];
    _index0 = _lims[0][0];
    _device_ptr = newCUDA<type>(_size);
};

template <typename T>
void DeviceNDArray<T, 1>::free() {
    if (_device_ptr != nullptr) {
        cudaFree(_device_ptr);
        _device_ptr = nullptr;
    }
};

template <typename T>
template <typename VecType>
void DeviceNDArray<T, 1>::copy(const VecType& _vec) {
    if (_device_ptr != nullptr) {
        copyToDeviceCUDA(_device_ptr, &_vec[_index0], _size);
    }
};

template <typename T>
template <typename VecType>
void DeviceNDArray<T, 1>::copyFromZero(const VecType& _vec) {
    if (_device_ptr != nullptr) {
        copyToDeviceCUDA(_device_ptr, &_vec[0], _size);
    }
};

template <typename T>
template <typename VecType>
void DeviceNDArray<T, 1>::retrieve(VecType& _vec) const {
    if (_device_ptr != nullptr) {
        copyToHostCUDA(&_vec[_index0], _device_ptr, _size);
    }
};

template <typename T>
template <typename VecType>
void DeviceNDArray<T, 1>::retrieveToZero(VecType& _vec) const {
    if (_device_ptr != nullptr) {
        copyToHostCUDA(&_vec[0], _device_ptr, _size);
    }
};

template <typename T>
T* DeviceNDArray<T, 1>::getDeviceMoved() {
    return (_device_ptr != nullptr) ? (_device_ptr - _index0) : nullptr;
};

template <typename T>
const T* DeviceNDArray<T, 1>::getDeviceMoved() const {
    return (_device_ptr != nullptr) ? (_device_ptr - _index0) : nullptr;
};
