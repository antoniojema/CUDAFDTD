/*
    Developers: Miguel D. Ruiz-Cabello. Antonio J. Martin Valverde
    Institution: UGR, Electromagnetic group of Granada
    Proyect: FDTDpp
*/

#pragma once

#include "common.h"
#include <array>
#include <initializer_list>
#include <iostream>
#include <tuple>
#include <algorithm>
#include "typedef.h"

template<typename T, size_t Size>
class Array {
public:
    constexpr __host__ __device__ T& operator[] (size_t i) {
        return this->_elems[i];
    }

    constexpr __host__ __device__ const T& operator[](size_t i) const {
        return this->_elems[i];
    }

    __host__ __device__ Array& operator=(std::array<T, Size> _rhs) {
        for (size_t i = 0; i < Size; i++) {
            _elems[i] = _rhs[i];
        }
        return *this;
    }

    T _elems[Size];
};

template <typename T, size_t D >
class Box {
public:
    // Attributes
    using TypeArray =  std::array<T, D>;

    Array<Array<T,D>, 2 > bounds;
    Array<T,D> & i = bounds[0];
    Array<T,D> & e = bounds[1];
    
    // Constructors
    Box(){}

    Box (const std::array<T,D> &_i, const std::array<T,D>  &_e ) {
        this->i = _i; this->e = _e;
    }
    
    Box (const Box & rhs) {
        *this = rhs;
    }
    Box& operator = (const Box &rhs){
        this->bounds = rhs.bounds;
        return *this;
    }

    // Logical operators
    bool operator == (const Box &rhs) const{
        for (size_t lu  = 0; lu  < 2; lu ++)
        for (size_t xyz = 0; xyz < D; xyz++)
            if (this->bounds[lu][xyz] != rhs[lu][xyz])
                return false;
        return true;
    }

    bool operator != (const Box &rhs) const{
        return !((*this) == rhs);
    }

    // AJMV: Esto es para la ordenacion de Boxes cuando sea necesario
    //       Prioriza el limite inferior y el primer eje (generalmente X)
    bool operator < (const Box &rhs) const{
        for (size_t lu  = 0; lu  < 2; lu ++)
        for (size_t xyz = 0; xyz < D; xyz++)
            if (this->bounds[lu][xyz] != rhs.bounds[lu][xyz])
                return this->bounds[lu][xyz] < rhs.bounds[lu][xyz];
        return false;
    }

    // Methods
    __host__ __device__ Array<T,D>& operator [] (const size_t i){
        return bounds[i];
    }
    __host__ __device__ const Array<T,D>& operator [] (const size_t i) const{
        return bounds[i];
    }

    bool contains(const std::array<T, D> point, bool exclude_end = false) const {
        T sub = (exclude_end) ? 1 : 0;
        for (size_t dir = 0; dir < D; dir++)
            if (point[dir] < this->i[dir] || this->e[dir]-sub < point[dir])
                return false;
        return true;
    }

    bool isSubsetOf(const Box<T, D>& other) const {
        for (size_t dir = 0; dir < D; dir++)
            if (this->i[dir] < other.i[dir] || this->e[dir] > other.e[dir] )
                return false;
        return true;
    }

    bool isSupersetOf(const Box<T, D>& other) const {
        return other.isSubsetOf(*this);
    }
};
