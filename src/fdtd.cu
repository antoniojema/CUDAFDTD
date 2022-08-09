#include "fdtd.h"

#include <vector>
#include <iostream>
#include <array>
#include "CudaVector.h"
#include "typedef.h"
#include "constants.h"


/*******************/
/*   CudaArray3D   */
/*******************/

template <typename T>
void CudaArray3D<T>::set(const std::array<size_t, CudaArray3D<T>::Dim> _dim) {
    this->dim = _dim;

    this->ptr_device_lvl0.resize(_dim[0]);
    this->ptr_device_lvl1.resize(_dim[0]);
    for (size_t i = 0; i < _dim[0]; i++) {
        this->ptr_device_lvl0[i].resize(_dim[1]);
        for (size_t j = 0; j < _dim[1]; j++) {
            this->ptr_device_lvl0[i][j] = newCUDA<T>(_dim[2]);
        }
        this->ptr_device_lvl1[i] = newCUDA<T*>(_dim[1]);
        copyToDeviceCUDA(this->ptr_device_lvl1[i], &this->ptr_device_lvl0[i][0], _dim[1]);
    }
    this->ptr_device_lvl2 = newCUDA<T*>(_dim[0]);
    copyToDeviceCUDA(this->ptr_device_lvl2, &this->ptr_device_lvl1[0], _dim[0]);
}

template <typename T>
void CudaArray3D<T>::del() {
    for (size_t i = 0; i < this->dim[0]; i++) {
        for (size_t j = 0; j < this->dim[1]; j++) {
            cudaFree(this->ptr_device_lvl0[i][j]);
        }
        cudaFree(this->ptr_device_lvl1[i]);
    }
    cudaFree(this->ptr_device_lvl2);
    this->ptr_device_lvl0 = {};
    this->ptr_device_lvl1 = {};
}

template <typename T>
template <typename VecType>
void CudaArray3D<T>::copy(const VecType& _src) {
    for (size_t i = 0; i < this->dim[0]; i++) {
    for (size_t j = 0; j < this->dim[1]; j++) {
        copyToDeviceCUDA(this->ptr_device_lvl0[i][j], &_src[i][j][0], this->dim[2]);
    }}
}
    
template <typename T>
template <typename VecType>
void CudaArray3D<T>::setAndCopy(const std::array<size_t, CudaArray3D<T>::Dim> _dim, VecType& _src) {
    this->set(_dim);
    this->copy(_src);
}
    
template <typename T>
template <typename VecType>
void CudaArray3D<T>::retrieve(VecType& _dst) {
    for (size_t i = 0; i < this->dim[0]; i++) {
    for (size_t j = 0; j < this->dim[1]; j++) {
        copyToHostCUDA(&_dst[i][j][0], this->ptr_device_lvl0[i][j], this->dim[2]);
    }}
}


/*****************/
/*   HostField   */
/*****************/

Field::Field(const FieldType _ft, const Orientation _or, const Box<ssize_t,3> _sim_bounds) :
    eh(_ft), xyz(_or)
{
    if (_ft == FieldType::None || _or == Orientation::None) {
        //TODO: ERROR
    }
    
    std::array<int,3> add {0,0,0};
    switch (_or) {
    case (Orientation::X):
        add[0] = (_ft == FieldType::E) ? 0 : 1;
        add[1] = (_ft == FieldType::E) ? 1 : 0;
        add[2] = (_ft == FieldType::E) ? 1 : 0;
        break;
    case (Orientation::Y):
        add[0] = (_ft == FieldType::E) ? 1 : 0;
        add[1] = (_ft == FieldType::E) ? 0 : 1;
        add[2] = (_ft == FieldType::E) ? 1 : 0;
        break;
    case (Orientation::Z):
        add[0] = (_ft == FieldType::E) ? 1 : 0;
        add[1] = (_ft == FieldType::E) ? 1 : 0;
        add[2] = (_ft == FieldType::E) ? 0 : 1;
        break;
    }

    for (size_t lu  = 0; lu  < 2; lu ++)
    for (size_t xyz = 0; xyz < 3; xyz++)
        this->sweep[lu][xyz] = _sim_bounds[lu][xyz];
    for (size_t i = 0; i < 3; i++) {
        this->sweep[1][i] += add[i];
    }
    
    for (size_t lu  = 0; lu  < 2; lu ++)
    for (size_t xyz = 0; xyz < 3; xyz++)
        this->alloc[lu][xyz] = this->sweep[lu][xyz];
    if (_ft == FieldType::H) {
        for (size_t i = 0; i < 3; i++) {
            this->sweep[0][i] -= (1-add[i]);
            this->sweep[1][i] += (1-add[i]);
        }
    }

    this->f.resize(this->getAllocSize(Orientation::X));
    for (auto& mat : this->f) {
        mat.resize(this->getAllocSize(Orientation::Y));
        for (auto& vec : mat)
            vec.resize(this->getAllocSize(Orientation::Z));
    }
}

real_t& Field::getElem(const ssize_t _i, const ssize_t _j, const ssize_t _k) {
    return this->f[_i - this->alloc[0][0]]
                  [_j - this->alloc[0][1]]
                  [_k - this->alloc[0][2]];
}

const real_t& Field::getElem(const ssize_t _i, const ssize_t _j, const ssize_t _k) const {
    return this->f[_i - this->alloc[0][0]]
                  [_j - this->alloc[0][1]]
                  [_k - this->alloc[0][2]];
}

size_t Field::getAllocSize(const Orientation _xyz) const {
    return this->alloc[1][(int)_xyz] - this->alloc[0][(int)_xyz];
}

/*******************/
/*   DeviceField   */
/*******************/

DeviceField::DeviceField(Field& _hf) :
    host_f(&_hf), eh(_hf.eh), xyz(_hf.xyz)
{
    for (size_t lu  = 0; lu  < 2; lu ++)
    for (size_t xyz = 0; xyz < 3; xyz++)
        this->alloc[lu][xyz] = _hf.alloc[lu][xyz];

    for (size_t lu  = 0; lu  < 2; lu ++)
    for (size_t xyz = 0; xyz < 3; xyz++)
        this->sweep[lu][xyz] = _hf.sweep[lu][xyz];

    //this->device_f = newCUDA<real_t>(_hf.getAllocSize());
}

__device__ real_t& DeviceField::getElem(const ssize_t _i, const ssize_t _j, const ssize_t _k) {
    size_t i = _i - this->alloc[0][0];
    size_t j = _j - this->alloc[0][1];
    size_t k = _k - this->alloc[0][2];
    size_t sz = this->getAllocSizeInDim(Orientation::Z);
    size_t sy = this->getAllocSizeInDim(Orientation::Y);
    
    return this->device_f[i * sy*sz + j * sz + k];
}

__device__ size_t DeviceField::getAllocSize() const {
    return this->getAllocSizeInDim(Orientation::X) *
           this->getAllocSizeInDim(Orientation::Y) *
           this->getAllocSizeInDim(Orientation::Z);
}

__device__ size_t DeviceField::getAllocSizeInDim(const Orientation _or) const {
    return this->alloc[1][(int)_or] - this->alloc[0][(int)_or];
}

__device__ bool DeviceField::inSweep(const ssize_t i, const ssize_t j, const ssize_t k) {
    return this->sweep[0][0] <= i && i < this->sweep[1][0] &&
           this->sweep[1][0] <= j && j < this->sweep[1][1] &&
           this->sweep[2][0] <= k && k < this->sweep[1][2];
}

void DeviceField::retrieveToHost() {
    //copyToHostCUDA(this->host_f->getPtr(), this->device_f, this->host_f->getAllocSize());
}


/***********************/
/*   OUTSIDE CLASSES   */
/***********************/

namespace {

size_t ceilInt(const size_t num, const size_t den) {
    return num/den + (num % den != 0);
}

__global__ void FDTDIterationEx(DeviceField *Ex, DeviceField *Hy, DeviceField *Hz, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + Ex->getSweep(Bound::L, Orientation::X);
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + Ex->getSweep(Bound::L, Orientation::Y);
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + Ex->getSweep(Bound::L, Orientation::Z);
    
    if (Ex->inSweep(i,j,k)) {
        Ex->getElem(i, j, k) = Ex->getElem(i, j, k) + C * (
            + Hy->getElem(i, j  , k-1)
            - Hy->getElem(i, j  , k  )
            - Hz->getElem(i, j-1, k  )
            + Hz->getElem(i, j  , k  )
        );
    }
}

__global__ void FDTDIterationEy(DeviceField *Ey, DeviceField *Hz, DeviceField *Hx, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + Ey->getSweep(Bound::L, Orientation::X);
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + Ey->getSweep(Bound::L, Orientation::Y);
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + Ey->getSweep(Bound::L, Orientation::Z);
    
    if (Ey->inSweep(i,j,k)) {
        Ey->getElem(i, j, k) = Ey->getElem(i, j, k) + C * (
            + Hz->getElem(i-1, j, k  )
            - Hz->getElem(i  , j, k  )
            - Hx->getElem(i  , j, k-1)
            + Hx->getElem(i  , j, k  )
        );
    }
}

__global__ void FDTDIterationEz(DeviceField *Ez, DeviceField *Hx, DeviceField *Hy, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + Ez->getSweep(Bound::L, Orientation::X);
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + Ez->getSweep(Bound::L, Orientation::Y);
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + Ez->getSweep(Bound::L, Orientation::Z);
    
    if (Ez->inSweep(i,j,k)) {
        Ez->getElem(i, j, k) = Ez->getElem(i, j, k) + C * (
            + Hx->getElem(i  , j-1, k)
            - Hx->getElem(i  , j  , k)
            - Hy->getElem(i-1, j  , k)
            + Hy->getElem(i  , j  , k)
        );
    }
}

__global__ void FDTDIterationHx(DeviceField *Hx, DeviceField *Ey, DeviceField *Ez, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + Hx->getSweep(Bound::L, Orientation::X);
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + Hx->getSweep(Bound::L, Orientation::Y);
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + Hx->getSweep(Bound::L, Orientation::Z);
    
    if (Hx->inSweep(i,j,k)) {
        Hx->getElem(i, j, k) = Hx->getElem(i, j, k) + C * (
            - Ey->getElem(i, j  , k  )
            + Ey->getElem(i, j  , k+1)
            + Ez->getElem(i, j  , k  )
            - Ez->getElem(i, j+1, k  )
        );
    }
}

__global__ void FDTDIterationHy(DeviceField *Hy, DeviceField *Ez, DeviceField *Ex, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + Hy->getSweep(Bound::L, Orientation::X);
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + Hy->getSweep(Bound::L, Orientation::Y);
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + Hy->getSweep(Bound::L, Orientation::Z);
    
    if (Hy->inSweep(i,j,k)) {
        Hy->getElem(i, j, k) = Hy->getElem(i, j, k) + C * (
            - Ez->getElem(i  , j, k  )
            + Ez->getElem(i+1, j, k  )
            + Ex->getElem(i  , j, k  )
            - Ex->getElem(i  , j, k+1)
        );
    }
}

__global__ void FDTDIterationHz(DeviceField *Hz, DeviceField *Ex, DeviceField *Ey, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + Hz->getSweep(Bound::L, Orientation::X);
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + Hz->getSweep(Bound::L, Orientation::Y);
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + Hz->getSweep(Bound::L, Orientation::Z);

    if (Hz->inSweep(i,j,k)) {
        Hz->getElem(i, j, k) = Hz->getElem(i, j, k) + C * (
            - Ex->getElem(k  , i  , j)
            + Ex->getElem(k  , i+1, j)
            + Ey->getElem(k  , i  , j)
            - Ey->getElem(k+1, i  , j)
        );
    }
}

//
//class FDTDRun_NOCUDA;
//
//class FDTDRun_CUDA {
//private:
//    HostField Ex_h, Ey_h, Ez_h, Hx_h, Hy_h, Hz_h;
//    DeviceField Ex_d, Ey_d, Ez_d, Hx_d, Hy_d, Hz_d;
//    DeviceField *Ex_d_ptr, *Ey_d_ptr, *Ez_d_ptr;
//    DeviceField *Hx_d_ptr, *Hy_d_ptr, *Hz_d_ptr;
//    size_t Nt;
//    real_t Ce;
//    real_t Ch;
//
//    void FDTDIterationE() {
//        dim3 block_size {10,10,10};
//        dim3 grid_size;
//        
//        // Ex
//        grid_size.x = ceilInt(Ex_h.getAllocSizeInDim(Orientation::X), block_size.x);
//        grid_size.y = ceilInt(Ex_h.getAllocSizeInDim(Orientation::Y), block_size.y);
//        grid_size.z = ceilInt(Ex_h.getAllocSizeInDim(Orientation::Z), block_size.z);
//        FDTDIterationEx _CK(grid_size, block_size)(Ex_d_ptr, Hy_d_ptr, Hz_d_ptr, Ce);
//        
//        // Ey
//        grid_size.x = ceilInt(Ey_h.getAllocSizeInDim(Orientation::X), block_size.x);
//        grid_size.y = ceilInt(Ey_h.getAllocSizeInDim(Orientation::Y), block_size.y);
//        grid_size.z = ceilInt(Ey_h.getAllocSizeInDim(Orientation::Z), block_size.z);
//        FDTDIterationEy _CK(grid_size, block_size)(Ey_d_ptr, Hz_d_ptr, Hx_d_ptr, Ce);
//
//        // Ez
//        grid_size.x = ceilInt(Ez_h.getAllocSizeInDim(Orientation::X), block_size.x);
//        grid_size.y = ceilInt(Ez_h.getAllocSizeInDim(Orientation::Y), block_size.y);
//        grid_size.z = ceilInt(Ez_h.getAllocSizeInDim(Orientation::Z), block_size.z);
//        FDTDIterationEz _CK(grid_size, block_size)(Ez_d_ptr, Hx_d_ptr, Hy_d_ptr, Ce);
//    }
//
//
//    void FDTDIterationH() {
//        dim3 block_size {10,10,10};
//        dim3 grid_size;
//        
//        // Hx
//        grid_size.x = ceilInt(Hx_h.getAllocSizeInDim(Orientation::X), block_size.x);
//        grid_size.y = ceilInt(Hx_h.getAllocSizeInDim(Orientation::Y), block_size.y);
//        grid_size.z = ceilInt(Hx_h.getAllocSizeInDim(Orientation::Z), block_size.z);
//        FDTDIterationHx _CK(grid_size, block_size)(Hx_d_ptr, Ey_d_ptr, Ez_d_ptr, Ce);
//        
//        // Hy
//        grid_size.x = ceilInt(Hy_h.getAllocSizeInDim(Orientation::X), block_size.x);
//        grid_size.y = ceilInt(Hy_h.getAllocSizeInDim(Orientation::Y), block_size.y);
//        grid_size.z = ceilInt(Hy_h.getAllocSizeInDim(Orientation::Z), block_size.z);
//        FDTDIterationHy _CK(grid_size, block_size)(Hy_d_ptr, Ez_d_ptr, Ex_d_ptr, Ce);
//
//        // Hz
//        grid_size.x = ceilInt(Hz_h.getAllocSizeInDim(Orientation::X), block_size.x);
//        grid_size.y = ceilInt(Hz_h.getAllocSizeInDim(Orientation::Y), block_size.y);
//        grid_size.z = ceilInt(Hz_h.getAllocSizeInDim(Orientation::Z), block_size.z);
//        FDTDIterationHz _CK(grid_size, block_size)(Hz_d_ptr, Ex_d_ptr, Ey_d_ptr, Ce);
//    }
//
//
//    void FDTDIteration() {
//        this->FDTDIterationE();
//        this->FDTDIterationH();
//    }
//
//public:
//    FDTDRun_CUDA(Box<ssize_t, 3> _sim_bounds, size_t _Nt, real_t Dt, real_t Ds) :
//        Ex_h (Field::E, Orientation::X, _sim_bounds),
//        Ey_h (Field::E, Orientation::Y, _sim_bounds),
//        Ez_h (Field::E, Orientation::Z, _sim_bounds),
//        Hx_h (Field::H, Orientation::X, _sim_bounds),
//        Hy_h (Field::H, Orientation::Y, _sim_bounds),
//        Hz_h (Field::H, Orientation::Z, _sim_bounds),
//        Ex_d(Ex_h), Ey_d(Ey_h), Ez_d(Ez_h),
//        Hx_d(Hx_h), Hy_d(Hy_h), Hz_d(Hz_h),
//        Ex_d_ptr(newCUDA<DeviceField>(1)),
//        Ey_d_ptr(newCUDA<DeviceField>(1)),
//        Ez_d_ptr(newCUDA<DeviceField>(1)),
//        Hx_d_ptr(newCUDA<DeviceField>(1)),
//        Hy_d_ptr(newCUDA<DeviceField>(1)),
//        Hz_d_ptr(newCUDA<DeviceField>(1)),
//        Nt(_Nt),
//        Ce(Dt/(EPS0*Ds)),
//        Ch(Dt/(MU0 *Ds))
//    {
//        copyToDeviceCUDA(Ex_d_ptr, &Ex_d, 1);
//        copyToDeviceCUDA(Ey_d_ptr, &Ey_d, 1);
//        copyToDeviceCUDA(Ez_d_ptr, &Ez_d, 1);
//        copyToDeviceCUDA(Hx_d_ptr, &Hx_d, 1);
//        copyToDeviceCUDA(Hy_d_ptr, &Hy_d, 1);
//        copyToDeviceCUDA(Hz_d_ptr, &Hz_d, 1);
//    }
//
//    ~FDTDRun_CUDA() {
//        cudaFree(Ex_d_ptr);
//        cudaFree(Ey_d_ptr);
//        cudaFree(Ez_d_ptr);
//        cudaFree(Hx_d_ptr);
//        cudaFree(Hy_d_ptr);
//        cudaFree(Hz_d_ptr);
//    }
//    
//    void run() {
//        for (size_t nt = 0; nt < this->Nt; nt++) {
//            this->FDTDIteration();
//        }
//        Ex_d.retrieveToHost();
//        Ey_d.retrieveToHost();
//        Ez_d.retrieveToHost();
//        Hx_d.retrieveToHost();
//        Hy_d.retrieveToHost();
//        Hz_d.retrieveToHost();
//    }
//
//    friend void compareValues(const FDTDRun_CUDA& cuda, const FDTDRun_NOCUDA& no_cuda);
//};
//
//class FDTDRun_NOCUDA {
//private:
//    HostField Ex, Ey, Ez, Hx, Hy, Hz;
//    real_t Ce, Ch;
//    size_t Nt;
//
//    void FDTDIterationE() {
//        // Ex
//        #pragma omp parallel
//        {
//        #pragma omp for collapse(2)
//        for (size_t i = Ex.getSweep(Bound::L, Orientation::X); i < Ex.getSweep(Bound::U, Orientation::X); i++) {
//        for (size_t j = Ex.getSweep(Bound::L, Orientation::Y); j < Ex.getSweep(Bound::U, Orientation::Y); j++) {
//        for (size_t k = Ex.getSweep(Bound::L, Orientation::Z); k < Ex.getSweep(Bound::U, Orientation::Z); k++) {
//            Ex.getElem(i, j, k) = Ex.getElem(i, j, k) + Ce * (
//                + Hy.getElem(i, j  , k-1)
//                - Hy.getElem(i, j  , k  )
//                - Hz.getElem(i, j-1, k  )
//                + Hz.getElem(i, j  , k  )
//            );
//        }}}
//        
//        // Ey
//        #pragma omp for collapse(2)
//        for (size_t i = Ey.getSweep(Bound::L, Orientation::X); i < Ey.getSweep(Bound::U, Orientation::X); i++) {
//        for (size_t j = Ey.getSweep(Bound::L, Orientation::Y); j < Ey.getSweep(Bound::U, Orientation::Y); j++) {
//        for (size_t k = Ey.getSweep(Bound::L, Orientation::Z); k < Ey.getSweep(Bound::U, Orientation::Z); k++) {
//            Ey.getElem(i, j, k) = Ey.getElem(i, j, k) + Ce * (
//                + Hz.getElem(i-1, j, k  )
//                - Hz.getElem(i  , j, k  )
//                - Hx.getElem(i  , j, k-1)
//                + Hx.getElem(i  , j, k  )
//            );
//        }}}
//        
//        // Ez
//        #pragma omp for collapse(2)
//        for (size_t i = Ez.getSweep(Bound::L, Orientation::X); i < Ez.getSweep(Bound::U, Orientation::X); i++) {
//        for (size_t j = Ez.getSweep(Bound::L, Orientation::Y); j < Ez.getSweep(Bound::U, Orientation::Y); j++) {
//        for (size_t k = Ez.getSweep(Bound::L, Orientation::Z); k < Ez.getSweep(Bound::U, Orientation::Z); k++) {
//            Ez.getElem(i, j, k) = Ez.getElem(i, j, k) + Ce * (
//                + Hx.getElem(i  , j-1, k)
//                - Hx.getElem(i  , j  , k)
//                - Hy.getElem(i-1, j  , k)
//                + Hy.getElem(i  , j  , k)
//            );
//        }}}
//        }
//    }
//
//    void FDTDIterationH() {
//        #pragma omp parallel
//        {
//        // Hx
//        #pragma omp for collapse(2)
//        for (size_t i = Hx.getSweep(Bound::L, Orientation::X); i < Hx.getSweep(Bound::U, Orientation::X); i++) {
//        for (size_t j = Hx.getSweep(Bound::L, Orientation::Y); j < Hx.getSweep(Bound::U, Orientation::Y); j++) {
//        for (size_t k = Hx.getSweep(Bound::L, Orientation::Z); k < Hx.getSweep(Bound::U, Orientation::Z); k++) {
//            Hx.getElem(i, j, k) = Hx.getElem(i, j, k) + Ch * (
//                - Ey.getElem(i, j  , k  )
//                + Ey.getElem(i, j  , k+1)
//                + Ez.getElem(i, j  , k  )
//                - Ez.getElem(i, j+1, k  )
//            );
//        }}}
//        
//        // Hy
//        #pragma omp for collapse(2)
//        for (size_t i = Hx.getSweep(Bound::L, Orientation::X); i < Hx.getSweep(Bound::U, Orientation::X); i++) {
//        for (size_t j = Hx.getSweep(Bound::L, Orientation::Y); j < Hx.getSweep(Bound::U, Orientation::Y); j++) {
//        for (size_t k = Hx.getSweep(Bound::L, Orientation::Z); k < Hx.getSweep(Bound::U, Orientation::Z); k++) {
//            Hy.getElem(i, j, k) = Hy.getElem(i, j, k) + Ch * (
//                - Ez.getElem(i  , j, k  )
//                + Ez.getElem(i+1, j, k  )
//                + Ex.getElem(i  , j, k  )
//                - Ex.getElem(i  , j, k+1)
//            );
//        }}}
//
//        // Hz
//        #pragma omp for collapse(2)
//        for (size_t i = Hx.getSweep(Bound::L, Orientation::X); i < Hx.getSweep(Bound::U, Orientation::X); i++) {
//        for (size_t j = Hx.getSweep(Bound::L, Orientation::Y); j < Hx.getSweep(Bound::U, Orientation::Y); j++) {
//        for (size_t k = Hx.getSweep(Bound::L, Orientation::Z); k < Hx.getSweep(Bound::U, Orientation::Z); k++) {
//            Hz.getElem(i, j, k) = Hz.getElem(i, j, k) + Ch * (
//                - Ex.getElem(k  , i  , j)
//                + Ex.getElem(k  , i+1, j)
//                + Ey.getElem(k  , i  , j)
//                - Ey.getElem(k+1, i  , j)
//            );
//        }}}
//        }
//    }
//
//
//    void FDTDIteration() {
//        this->FDTDIterationE();
//        this->FDTDIterationH();
//    }
//
//public:
//    FDTDRun_NOCUDA(Box<ssize_t, 3> _sim_bounds, size_t _Nt, real_t Dt, real_t Ds) :
//        Ex (Field::E, Orientation::X, _sim_bounds),
//        Ey (Field::E, Orientation::Y, _sim_bounds),
//        Ez (Field::E, Orientation::Z, _sim_bounds),
//        Hx (Field::H, Orientation::X, _sim_bounds),
//        Hy (Field::H, Orientation::Y, _sim_bounds),
//        Hz (Field::H, Orientation::Z, _sim_bounds),
//        Nt(_Nt),
//        Ce(Dt/(EPS0*Ds)),
//        Ch(Dt/(MU0 *Ds))
//    {}
//
//    void run() {
//        for (size_t nt = 0; nt < this->Nt; nt++) {
//            FDTDIteration();
//        }
//    }
//
//    friend void compareValues(const FDTDRun_CUDA& cuda, const FDTDRun_NOCUDA& no_cuda);
//};
//
//void compareValues(const FDTDRun_CUDA& cuda, const FDTDRun_NOCUDA& no_cuda) {
//    std::cout << cuda.Ex_h.getElem(3,4,5) << " " << no_cuda.Ex.getElem(3, 4, 5) << std::endl;
//}

} // namespace


//void runFDTDTest() {
//    // Params
//    Box<ssize_t, 3> sim_boundaries {0,0,0, 10,10,10};
//    size_t Nt = 10;
//    real_t Dt = 1e-7;
//    real_t Ds = 0.1;
//
//    std::cout << "Running cuda ..." << std::endl;
//
//    // CUDA
//    FDTDRun_CUDA fdtd_run_cuda(sim_boundaries, Nt, Dt, Ds);
//    fdtd_run_cuda.run();
//
//    std::cout << "Running no-cuda ..." << std::endl;
//    
//    // NO CUDA
//    FDTDRun_NOCUDA fdtd_run_nocuda (sim_boundaries, Nt, Dt, Ds);
//    fdtd_run_nocuda.run();
//
//    std::cout << "Done." << std::endl;
//
//    // Compare
//    compareValues(fdtd_run_cuda, fdtd_run_nocuda);
//}

void runFDTDTest() {
    // Params
    //Box<ssize_t, 3> sim_boundaries {0,0,0, 10,10,10};
    //size_t Nt = 10;
    //real_t Dt = 1e-7;
    //real_t Ds = 0.1;

    //Field Ex (FieldType::E, Orientation::X, sim_boundaries);
    //Field Ey (FieldType::E, Orientation::Y, sim_boundaries);
    //Field Ez (FieldType::E, Orientation::Z, sim_boundaries);
    //Field Hx (FieldType::H, Orientation::X, sim_boundaries);
    //Field Hy (FieldType::H, Orientation::Y, sim_boundaries);
    //Field Hz (FieldType::H, Orientation::Z, sim_boundaries);

    //CudaArray3D Ex_cuda (Ex)
}