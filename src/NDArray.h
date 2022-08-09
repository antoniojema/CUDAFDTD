#pragma once

#include "typedef.h"
#include "NDVector.h"
#include <array>

/**
    Class template NDArray<T,N>.
    Provides a fixed-bounded N-dimensional std::vector-like
    structure whose ranges may begin at any positive or negative
    index. Bounds can be resetted but reallocation may occur.
    The index in the dimension `n` vary between the limits
    `lim0(n)` and `lim1(n)-1` (both included).
*/
template <typename T, size_t N>
class NDArray : private NDVector<T, N> {
public:
    using ContainerType = NDVector<T,N>;
    using typename ContainerType::ElementType;

public:
    NDArray() = default;
    
    NDArray(const std::array<size_t,N>& size);
    NDArray(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1);
    NDArray(const Box<ssize_t, N>& _lims);
    
    NDArray(const NDArray& rhs);
    NDArray& operator=(const NDArray& rhs);
    
    NDArray(NDArray&& rhs) noexcept;
    NDArray& operator=(NDArray&& rhs) noexcept;

    void reset(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1);
    void reset(const Box<ssize_t, N>& _lims);
    
    using ContainerType::operator[];
    using ContainerType::at;
    using ContainerType::fill;
    using ContainerType::lim0;
    using ContainerType::lim1;
    using ContainerType::size;
    using ContainerType::begin;
    using ContainerType::end;
    using ContainerType::rbegin;
    using ContainerType::rend;
    using ContainerType::cbegin;
    using ContainerType::cend;
    using ContainerType::crbegin;
    using ContainerType::crend;
    using ContainerType::back;
    using ContainerType::front;

private:
    ContainerType& getVector();
    const ContainerType& getVector() const;
};

/**
    Class template NDArray<T,N>.
    Provides a fixed-bounded N-dimensional std::vector-like
    structure whose ranges may begin at any positive or negative
    index. Bounds can be resetted but reallocation may occur.
    The index in the dimension `n` vary between the limits
    `lim0(n)` and `lim1(n)-1` (both included).
*/
template <typename T>
class NDArray<T,1> : private NDVector<T,1> {
public:
    using ContainerType = NDVector<T,1>;
    using typename ContainerType::ElementType;

public:
    NDArray() = default;
    
    NDArray(const size_t size);
    NDArray(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1);
    NDArray(const ssize_t i0, const ssize_t i1);
    
    NDArray(const NDArray& rhs);
    NDArray& operator=(const NDArray& rhs);

    NDArray(NDArray&& rhs) noexcept;
    NDArray& operator=(NDArray&& rhs) noexcept;

    void reset(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1);
    void reset(const ssize_t i0, const ssize_t i1);
    
    using ContainerType::operator[];
    using ContainerType::at;
    using ContainerType::fill;
    using ContainerType::lim0;
    using ContainerType::lim1;
    using ContainerType::size;
    using ContainerType::begin;
    using ContainerType::end;
    using ContainerType::rbegin;
    using ContainerType::rend;
    using ContainerType::cbegin;
    using ContainerType::cend;
    using ContainerType::crbegin;
    using ContainerType::crend;
    using ContainerType::back;
    using ContainerType::front;

    ContainerType& getVector();
    const ContainerType& getVector() const;
};

#include "NDArray_impl.h"
