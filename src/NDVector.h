#pragma once

#include "typedef.h"
#include "box.h"
#include <vector>
#include <array>

/**
    Class template NDVector<T,N>.
    Provides a N-dimensional std::vector-like structure whose
    ranges may begin at any positive or negative index.
    The index in the dimension `n` vary between the limits
    `lim0(n)` and `lim1(n)-1` (both included).
*/
template <typename T, size_t N>
class NDVector : private std::vector<NDVector<T, N-1>> {
public:
    using ValueType = T;
    using ElementType = NDVector<T, N-1>;
    using ContainerType = std::vector<ElementType>;

public:
    NDVector() = default;

    NDVector(const std::array<size_t,N>& size);
    NDVector(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1);
    NDVector(const Box<ssize_t,N>& _lims);
    
    NDVector(const NDVector<T,N>& rhs);
    NDVector& operator=(const NDVector<T,N>& rhs);
    
    NDVector(NDVector<T,N>&& rhs) noexcept;
    NDVector& operator=(NDVector<T,N>&& rhs) noexcept;

    void fill(const T& value);

    // Retur
    ssize_t lim0(const size_t dim) const;
    ssize_t lim1(const size_t dim) const;
    size_t  size(const size_t dim) const;
    
    void rebound(const std::array<size_t,N>& size);
    void reboundFirstIndex(const std::array<ssize_t,N>& i0);
    void rebound(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1);
    void rebound(const Box<ssize_t,N>& _lims);
    
    void resize(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1);
    void resize(const Box<ssize_t,N>& _lims);

    void push_back(const ElementType& value);
    template<class... Args> void emplace_back(Args&&... args);
    
    void push_front(const ElementType& value);

    void shrink_to_fit();
    
    ElementType& operator[](ssize_t n);
    const ElementType& operator[](ssize_t n) const;
    ElementType& at(ssize_t n);
    const ElementType& at(ssize_t n) const;

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

private:
    void movedPoint();
    void rebound1D(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1);

private:
    ElementType* moved_ptr = nullptr;
    Box<ssize_t, N> lims {};
};

/**
    Class template NDVector<T,N>.
    Provides a N-dimensional std::vector-like structure whose
    ranges may begin at any positive or negative index.
    The index in the dimension `n` vary between the limits
    `lim0(n)` and `lim1(n)-1` (both included).
*/
template <typename T>
class NDVector<T,1> : private std::vector<T> {
private:
public:
    using ValueType = T;
    using ElementType = T;
    using ContainerType = std::vector<ElementType>;

public:
    NDVector() = default;
    
    NDVector(const size_t size);
    NDVector(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1);
    NDVector(const ssize_t i0, const ssize_t i1);
    
    NDVector(const NDVector& rhs);
    NDVector& operator=(const NDVector& rhs);
    
    NDVector(NDVector&& rhs) noexcept;
    NDVector& operator=(NDVector&& rhs) noexcept;
    
    void fill(const T& value);

    ssize_t lim0() const;
    ssize_t lim1() const;
    using ContainerType::size;

    void rebound(const size_t size);
    void reboundFirstIndex(const ssize_t i0);
    void rebound(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1);
    void rebound(const ssize_t i0, const ssize_t i1);
    
    void resize(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1);
    void resize(const ssize_t i0, const ssize_t i1);

    void push_back(const T value);
    template<class... Args> void emplace_back(Args&&... args);

    void NDVector<T,1>::push_front(const T& value);
    
    void shrink_to_fit();

    ElementType& operator[](ssize_t n);
    const ElementType& operator[](ssize_t n) const;
    
    ElementType& at(ssize_t n);
    const ElementType& at(ssize_t n) const;

    using ContainerType::begin;
    using ContainerType::end;
    using ContainerType::rbegin;
    using ContainerType::rend;
    using ContainerType::cbegin;
    using ContainerType::cend;
    using ContainerType::crbegin;
    using ContainerType::crend;
    using ContainerType::shrink_to_fit;
    using ContainerType::back;
    using ContainerType::front;

//private:
    ContainerType& getVector();
    const ContainerType& getVector() const;

private:
    void movedPoint();

private:
    ElementType* moved_ptr = nullptr;
    std::array<ssize_t, 2> lims {};

};

#include "NDVector_impl.h"
