#pragma once

#include "NDVector.h"

/*********************/
/*    Constructor    */
/*********************/
template <typename T, size_t N>
/*inline*/ NDVector<T,N>::NDVector(const std::array<size_t, N>& size) {
    this->rebound(size);
}

template <typename T, size_t N>
/*inline*/ NDVector<T,N>::NDVector(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1) {
    this->rebound(i0, i1);
}

template <typename T, size_t N>
/*inline*/ NDVector<T,N>::NDVector(const Box<ssize_t,N>& _lims) {
    this->rebound(_lims[0], _lims[1]);
}

/**************/
/*    Copy    */
/**************/
template <typename T, size_t N>
/*inline*/ NDVector<T,N>::NDVector(const NDVector<T,N>& arr) {
    *this = arr;
}

template <typename T, size_t N>
/*inline*/ NDVector<T,N>& NDVector<T,N>::operator=(const NDVector<T,N>& rhs) {
    this->getVector() = rhs.getVector();
    this->lims = rhs.lims;
    this->movedPoint();
    return *this;
}

/**************/
/*    Move    */
/**************/
template <typename T, size_t N>
/*inline*/ NDVector<T,N>::NDVector(NDVector<T,N>&& arr) noexcept {
    *this = std::move(arr);
}

template <typename T, size_t N>
/*inline*/ NDVector<T,N>& NDVector<T,N>::operator=(NDVector<T,N>&& rhs) noexcept {
    this->getVector() = std::move(rhs.getVector());
    this->lims = rhs.lims;
    this->movedPoint();
    return *this;
}

/**************/
/*    Fill    */
/**************/
template <typename T, size_t N>
/*inline*/void NDVector<T,N>::fill(const T& value) {
    for(auto& e: *this){
        e.fill(value);
    }
}

/*****************/
/*    Rebound    */
/*****************/
template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::rebound(const std::array<size_t,N>& size) {
    std::array<ssize_t,N> i0;
    std::array<ssize_t,N> i1;
    for (size_t i = 0; i < N; i++){
        i0[i] = 0;
        i1[i] = size[i];
    }
    this->rebound(i0, i1);
}

template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::reboundFirstIndex(const std::array<ssize_t,N>& i0_) {
    std::array<ssize_t,N> i0;
    std::array<ssize_t,N> i1;
    for (size_t i = 0; i < N; i++){
        i0[i] = i0_[i];
        i1[i] = i0_[i] + this->size(i);
    }
    this->rebound(i0, i1);
}

template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::rebound(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1) {
    #ifdef _DEBUG
        for (size_t i = 0; i < N; i++)
            Log::ASSERT(
                i1[i] >= i0[i],
                "Error: Bad boundaries in NDVector rebound."
            );
    #endif

    this->rebound1D(i0, i1);
        
    std::array<ssize_t,N-1> i0_down;
    std::array<ssize_t,N-1> i1_down;
    for (size_t i = 0; i < N-1; i++){
        i0_down[i] = i0[i+1];
        i1_down[i] = i1[i+1];
    }
    for (auto& e : *this)
        e.rebound(i0_down, i1_down);
}

template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::rebound(const Box<ssize_t,N>& _lims) {
    this->rebound(_lims[0], _lims[1]);
}

/****************/
/*    Resize    */
/****************/
template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::resize(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1) {
    #ifdef _DEBUG
        for (size_t i = 0; i < N; i++)
            Log::ASSERT(
                i1[i] >= i0[i],
                "Error: Bad boundaries in NDVector resize."
            );
    #endif

    ssize_t old_i0 = this->lims[0][0];
    ssize_t old_i1 = this->lims[0][1];
    NDVector old_arr = *this;
        
    this->rebound1D(i0, i1);

    for (ssize_t i = std::max(i0[0], old_i0); i < std::min(i1[0], old_i1); i++)
        (*this)[i] = old_arr[i];
        
    std::array<ssize_t,N-1> i0_down;
    std::array<ssize_t,N-1> i1_down;
    for (size_t i = 0; i < N-1; i++){
        i0_down[i] = i0[i+1];
        i1_down[i] = i1[i+1];
    }
    for (auto& e : *this)
        e.resize(i0_down, i1_down);
}

template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::resize(const Box<ssize_t,N>& _lims) {
    this->resize(_lims[0], _lims[1]);
}

/********************/
/*    Get limits    */
/********************/
template <typename T, size_t N>
/*inline*/ ssize_t NDVector<T,N>::lim0(const size_t dim) const {
    return lims[0][dim];
}

template <typename T, size_t N>
/*inline*/ ssize_t NDVector<T,N>::lim1(const size_t dim) const{
    return lims[1][dim];
}

template <typename T, size_t N>
/*inline*/ size_t NDVector<T,N>::size(const size_t dim) const{
    return (size_t) (lims[1][dim]-lims[0][dim]);
}

/**********************/
/*    Push/Emplace    */
/**********************/
template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::push_back(const ElementType& value) {
    this->getVector().push_back(value);
    this->lims[0][1] ++;
    this->movedPoint();
}

template <typename T, size_t N>
template<class... Args>
/*inline*/ void NDVector<T,N>::emplace_back(Args&&... args) {
    this->getVector().emplace_back(args...);
    this->lims[0][1] ++;
    this->movedPoint();
}

template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::push_front(const ElementType& value) {
    this->getVector().insert(this->getVector().begin(), value);
    this->lims[0][0] --;
    this->movedPoint();
}

/***********************/
/*    Shrink To Fit    */
/***********************/
template <typename T, size_t N>
/*inline*/void NDVector<T,N>::shrink_to_fit() {
    this->getVector().shrink_to_fit();
    for (auto& e : *this)
        e.shrink_to_fit();
}

/****************/
/*    Access    */
/****************/
template <typename T, size_t N>
/*inline*/ typename NDVector<T,N>::ElementType& NDVector<T,N>::operator[](ssize_t n) {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

template <typename T, size_t N>
/*inline*/ const typename NDVector<T,N>::ElementType& NDVector<T,N>::operator[](ssize_t n) const {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

template <typename T, size_t N>
/*inline*/typename NDVector<T,N>::ElementType& NDVector<T,N>::at(ssize_t n) {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

template <typename T, size_t N>
/*inline*/const typename NDVector<T,N>::ElementType& NDVector<T,N>::at(ssize_t n) const {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

/********************/
/*    Get Vector    */
/********************/
template <typename T, size_t N>
/*inline*/ typename NDVector<T,N>::ContainerType& NDVector<T,N>::getVector() {
    return static_cast<ContainerType&>(*this);
}

template <typename T, size_t N>
/*inline*/ const typename NDVector<T,N>::ContainerType& NDVector<T,N>::getVector() const {
    return static_cast<const ContainerType&>(*this);
}

/*****************/
/*    Private    */
/*****************/
template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::movedPoint() {
    if (this->size(0) > 0)
        this->moved_ptr = &this->getVector()[0] - this->lims[0][0];
    else
        this->moved_ptr = nullptr;
}

template <typename T, size_t N>
/*inline*/ void NDVector<T,N>::rebound1D(const std::array<ssize_t,N>& i0, const std::array<ssize_t,N>& i1) {
    this->lims[0] = i0;
    this->lims[1] = i1;

    this->getVector().resize((size_t) (i1[0] - i0[0]));
    this->movedPoint();
}

/*****************************************************************/
/**************        ^^^ NDVector<T,N> ^^^        **************/
/*****************************************************************/
/**************        vvv NDVector<T,1> vvv        **************/
/*****************************************************************/

/*********************/
/*    Constructor    */
/*********************/
template <typename T>
/*inline*/NDVector<T,1>::NDVector(const size_t size) {
    this->rebound(size);
}

template <typename T>
/*inline*/NDVector<T,1>::NDVector(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1) {
    this->rebound(i0, i1);
}

template <typename T>
/*inline*/NDVector<T,1>::NDVector(const ssize_t i0, const ssize_t i1) {
    this->rebound(i0, i1);
}

/**************/
/*    Copy    */
/**************/
template <typename T>
/*inline*/NDVector<T,1>::NDVector(const NDVector& arr) {
    *this = arr;
}

template <typename T>
/*inline*/NDVector<T,1>& NDVector<T,1>::operator=(const NDVector& rhs) {
    this->getVector() = rhs.getVector();
    this->lims = rhs.lims;
    this->movedPoint();
    return *this;
}

/**************/
/*    Move    */
/**************/
template <typename T>
/*inline*/ NDVector<T,1>::NDVector(NDVector<T,1>&& rhs) noexcept {
    *this = std::move(rhs);
}

template <typename T>
/*inline*/ NDVector<T,1>& NDVector<T,1>::operator=(NDVector<T,1>&& rhs) noexcept {
    this->getVector() = std::move(rhs.getVector());
    this->lims = rhs.lims;
    this->movedPoint();
    return *this;
}

/**************/
/*    Fill    */
/**************/
template <typename T>
/*inline*/void NDVector<T,1>::fill(const T& value) {
    for(auto& e: *this){
        e = value;
    }
}

/********************/
/*    Get limits    */
/********************/
template <typename T>
/*inline*/ssize_t NDVector<T,1>::lim0() const {
    return this->lims[0];
}

template <typename T>
/*inline*/ssize_t NDVector<T,1>::lim1() const {
    return this->lims[1];
}

/*****************/
/*    Rebound    */
/*****************/
template <typename T>
/*inline*/void NDVector<T,1>::rebound(const size_t size) {
    this->rebound((ssize_t) 0, (ssize_t) size);
}

template <typename T>
/*inline*/void NDVector<T,1>::reboundFirstIndex(const ssize_t i0) {
    this->rebound(i0, i0 + (ssize_t) this->size());
}

template <typename T>
/*inline*/void NDVector<T,1>::rebound(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1) {
    this->rebound(i0[0], i1[0]);
}

template <typename T>
/*inline*/void NDVector<T,1>::rebound(const ssize_t i0, const ssize_t i1) {
    if (i1 >= i0) {
        this->lims = {i0, i1};
        this->getVector().resize((size_t) i1 - i0);
        this->movedPoint();
    }
}

/****************/
/*    Resize    */
/****************/
template <typename T>
/*inline*/void NDVector<T,1>::resize(const std::array<ssize_t,1>& i0, const std::array<ssize_t,1>& i1) {
    this->resize(i0[0], i1[0]);
}

template <typename T>
/*inline*/void NDVector<T,1>::resize(const ssize_t i0, const ssize_t i1) {
    if (i1 >= i0) {
        ssize_t old_i0 = this->lim0();
        ssize_t old_i1 = this->lim1();
        NDVector old_arr = *this;
        
        this->rebound(i0, i1);
        for (ssize_t i = std::max(i0, old_i0); i < std::min(i1, old_i1); i++)
            (*this)[i] = old_arr[i];
    }
}

/**********************/
/*    Push/Emplace    */
/**********************/
template <typename T>
/*inline*/void NDVector<T,1>::push_back(const T value) {
    this->getVector().push_back(value);
    this->lims[1] ++;
    this->movedPoint();
}

template <typename T>
template<class... Args>
/*inline*/void NDVector<T,1>::emplace_back(Args&&... args) {
    this->getVector().emplace_back(args...);
    this->lims[1] ++;
    this->movedPoint();
}

template <typename T>
/*inline*/ void NDVector<T,1>::push_front(const ElementType& value) {
    this->getVector().insert(this->getVector().begin(), value);
    this->lims[0] --;
    this->movedPoint();
}

/***********************/
/*    Shrink To Fit    */
/***********************/
template <typename T>
/*inline*/void NDVector<T,1>::shrink_to_fit() {
    this->getVector().shrink_to_fit();
}

/****************/
/*    Access    */
/****************/
template <typename T>
/*inline*/typename NDVector<T,1>::ElementType& NDVector<T,1>::operator[](ssize_t n) {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

template <typename T>
/*inline*/const typename NDVector<T,1>::ElementType& NDVector<T,1>::operator[](ssize_t n) const {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

template <typename T>
/*inline*/typename NDVector<T,1>::ElementType& NDVector<T,1>::at(ssize_t n) {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

template <typename T>
/*inline*/const typename NDVector<T,1>::ElementType& NDVector<T,1>::at(ssize_t n) const {
    //return this->moved_ptr[n];
    return *(this->moved_ptr + n);
}

/********************/
/*    Get Vector    */
/********************/
template <typename T>
/*inline*/typename NDVector<T,1>::ContainerType& NDVector<T,1>::getVector() {
    return static_cast<ContainerType&>(*this);
}

template <typename T>
/*inline*/const typename NDVector<T,1>::ContainerType& NDVector<T,1>::getVector() const {
    return static_cast<const ContainerType&>(*this);
}

/*********************/
/*    Moved point    */
/*********************/
template <typename T>
/*inline*/void NDVector<T,1>::movedPoint() {
    if (this->size() > 0)
        this->moved_ptr = &(this->getVector()[0]) - this->lims[0];
    else
        this->moved_ptr = nullptr;
}