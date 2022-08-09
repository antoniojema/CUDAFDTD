#pragma once

#include "NDArray.h"

/*********************/
/*    Constructor    */
/*********************/
template <typename T, size_t N>
inline NDArray<T,N>::NDArray(const std::array<size_t,N>& size) {
    std::array<ssize_t, N> i0;
    std::array<ssize_t, N> i1;
    for (size_t i = 0; i < N; i++) {
        i0[i] = 0;
        i1[i] = (ssize_t) size[i];
    }
    this->reset(i0, i1);
}

template <typename T, size_t N>
inline NDArray<T,N>::NDArray(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1) {
    this->reset(i0, i1);
}

template <typename T, size_t N>
inline NDArray<T,N>::NDArray(const Box<ssize_t,N>& _lims) {
    this->reset(_lims[0], _lims[1]);
}

/**************/
/*    Copy    */
/**************/
template <typename T, size_t N>
inline NDArray<T,N>::NDArray(const NDArray& rhs) {
    *this = rhs;
    this->getVector().shrink_to_fit();
}

template <typename T, size_t N>
inline NDArray<T,N>& NDArray<T,N>::operator=(const NDArray& rhs) {
    this->getVector() = rhs.getVector();
    return *this;
}

/**************/
/*    Move    */
/**************/
template <typename T, size_t N>
inline NDArray<T,N>::NDArray(NDArray&& rhs) noexcept {
    *this = std::move(rhs);
    this->getVector().shrink_to_fit();
}

template <typename T, size_t N>
inline NDArray<T,N>& NDArray<T,N>::operator=(NDArray&& rhs) noexcept {
    this->getVector() = std::move(rhs.getVector());
    return *this;
}

/***************/
/*    Reset    */
/***************/
template <typename T, size_t N>
inline void NDArray<T,N>::reset(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1) {
    this->getVector().rebound(i0, i1);
    this->getVector().shrink_to_fit();
}

template <typename T, size_t N>
inline void NDArray<T,N>::reset(const Box<ssize_t, N>& _lims) {
    this->getVector().rebound(_lims[0], _lims[1]);
    this->getVector().shrink_to_fit();
}

/********************/
/*    Get Vector    */
/********************/
template <typename T, size_t N>
inline typename NDArray<T,N>::ContainerType& NDArray<T,N>::getVector() {
    return static_cast<ContainerType&>(*this);
}

template <typename T, size_t N>
inline const typename NDArray<T,N>::ContainerType& NDArray<T,N>::getVector() const {
    return static_cast<const ContainerType&>(*this);
}

/****************************************************************/
/**************        ^^^ NDArray<T,N> ^^^        **************/
/****************************************************************/
/**************        vvv NDArray<T,1> vvv        **************/
/****************************************************************/

/*********************/
/*    Constructor    */
/*********************/
template <typename T>
inline NDArray<T,1>::NDArray(const size_t size) {
    this->reset((ssize_t) 0, (ssize_t) size);
}

template <typename T>
inline NDArray<T,1>::NDArray(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1) {
    this->reset(i0[0], i1[0]);
}

template <typename T>
inline NDArray<T,1>::NDArray(const ssize_t i0, const ssize_t i1) {
    this->reset(i0, i1);
}

/**************/
/*    Copy    */
/**************/
template <typename T>
inline NDArray<T,1>::NDArray(const NDArray& rhs) {
    *this = rhs;
    this->getVector().shrink_to_fit();
}

template <typename T>
inline NDArray<T,1>& NDArray<T,1>::operator=(const NDArray& rhs) {
    this->getVector() = rhs.getVector();
    return *this;
}

/**************/
/*    Move    */
/**************/
template <typename T>
inline NDArray<T,1>::NDArray(NDArray&& rhs) noexcept {
    *this = std::move(rhs);
    this->getVector().shrink_to_fit();
}

template <typename T>
inline NDArray<T,1>& NDArray<T,1>::operator=(NDArray&& rhs) noexcept {
    this->getVector() = std::move(rhs.getVector());
    return *this;
}

/***************/
/*    Reset    */
/***************/
template <typename T>
inline void NDArray<T,1>::reset(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1) {
    this->reset(i0[0], i1[0]);
}

template <typename T>
inline void NDArray<T,1>::reset(const ssize_t i0, const ssize_t i1) {
    this->getVector().rebound(i0, i1);
    this->getVector().shrink_to_fit();
}

/********************/
/*    Get Vector    */
/********************/
template <typename T>
inline typename NDArray<T,1>::ContainerType& NDArray<T,1>::getVector() {
    return static_cast<ContainerType&>(*this);
}

template <typename T>
inline const typename NDArray<T,1>::ContainerType& NDArray<T,1>::getVector() const {
    return static_cast<const ContainerType&>(*this);
}
