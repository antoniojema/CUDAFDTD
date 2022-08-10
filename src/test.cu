#include "typedef.h"
#include "cuda_utils.h"
#include "DeviceNDArray.h"
#include "NDArray.h"
#include <iostream>
#include <vector>
#include <array>


/************************************/
/*   VERSION 1 - Flattened matrix   */
/************************************/

__global__ void matmul_cuda(real_t* m1, real_t* m2, real_t* m3, size_t Nx, size_t Nc, size_t Ny) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < Nx && j < Ny) {
        m3[i*Ny+j] = 0;
        for (size_t k = 0; k < Nc; k++) {
            m3[i*Ny+j] += m1[i*Nc+k]*m2[k*Ny+j];
        }
    }
}

size_t ceilDiv(const size_t num, const size_t den) {
    return num/den + (num % den != 0);
}

void matmul() {
    size_t Nx = 11;
    size_t Nc = 12;
    size_t Ny = 13;

    size_t total_size_1 = Nx*Nc;
    size_t total_size_2 = Nc*Ny;
    size_t total_size_3 = Nx*Ny;

    real_t* m1 = new real_t[total_size_1];
    real_t* m2 = new real_t[total_size_2];
    real_t* m3 = new real_t[total_size_3];

    for (size_t i = 0; i < Nx; i++) {
    for (size_t j = 0; j < Nc; j++) {
        m1[i*Nc+j] = i;
    }}

    for (size_t i = 0; i < Nc; i++) {
    for (size_t j = 0; j < Ny; j++) {
        m2[i*Ny+j] = j;
    }}

    real_t* m1_d = newCUDA<real_t>(total_size_1);
    real_t* m2_d = newCUDA<real_t>(total_size_2);
    real_t* m3_d = newCUDA<real_t>(total_size_3);

    copyToDeviceCUDA(m1_d, m1, total_size_1);
    copyToDeviceCUDA(m2_d, m2, total_size_2);

    dim3 block_size {10,10,1};
    dim3 grid_size;
    grid_size.x = ceilDiv(Nx, block_size.x);
    grid_size.y = ceilDiv(Ny, block_size.y);
    grid_size.z = 1;

    matmul_cuda _CK(grid_size, block_size) (m1_d, m2_d, m3_d, Nx, Nc, Ny);

    copyToHostCUDA(m3, m3_d, total_size_3);
    
    for (size_t i = 0; i < Nx; i++) {
    for (size_t j = 0; j < Ny; j++) {
        std::cout << m3[i*Ny+j] << ((j==Ny-1) ? "\n" : " ");
    }}
    std::cout << std::endl;

    cudaFree(m1_d);
    cudaFree(m2_d);
    cudaFree(m3_d);
}


/*****************************/
/*   VERSION 2 - 2D matrix   */
/*****************************/

template <typename T>
class CudaArray2D {
public:
    using value_type = T;

    CudaArray2D() = default;

    CudaArray2D(const std::array<size_t,2> _dim) {this->set(_dim);}

    template <typename VecType>
    CudaArray2D(const std::array<size_t,2> _dim, VecType& _src) {this->setAndCopy(_dim, _src);}

    ~CudaArray2D() {this->del();}

    void set(const std::array<size_t,2> _dim) {
        this->dim = _dim;

        this->ptr_host   = new T*[_dim[0]];
        for (size_t i = 0; i < _dim[0]; i++) {
            this->ptr_host[i] = newCUDA<T>(_dim[1]);
        }
        
        this->ptr_device = newCUDA<T*>(_dim[0]);
        copyToDeviceCUDA(this->ptr_device, this->ptr_host, _dim[0]);
    }

    template <typename VecType>
    void copy(const VecType& _src) {
        for (size_t i = 0; i < this->dim[0]; i++) {
            copyToDeviceCUDA(this->ptr_host[i], &_src[i][0], this->dim[1]);
        }
    }
    
    template <typename VecType>
    void setAndCopy(const std::array<size_t,2> _dim, VecType& _src) {
        this->set(_dim);
        this->copy(_src);
    }
    
    template <typename VecType>
    void retrieve(VecType& _dst) {
        for (size_t i = 0; i < this->dim[0]; i++) {
            copyToHostCUDA(&_dst[i][0], this->ptr_host[i], this->dim[1]);
        }
    }

    void del() {
        if (this->ptr_host != nullptr) {
            for (size_t i = 0; i < this->dim[0]; i++) {
                cudaFree(this->ptr_host[i]);
            }
            delete[] ptr_host;
            this->ptr_host = nullptr;
        }
        if (this->ptr_device != nullptr) {
            cudaFree(this->ptr_device);
            this->ptr_device = nullptr;
        }
    }

    T** getDevicePtr() {return this->ptr_device;}

private:
    T** ptr_device {nullptr};
    T** ptr_host {nullptr};
    std::array<size_t, 2> dim {0,0};
};

__global__ void matmul2D_cuda(real_t** m1, real_t** m2, real_t** m3, size_t Nx, size_t Nc, size_t Ny) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < Nx && j < Ny) {
        m3[i][j] = 0;
        for (size_t k = 0; k < Nc; k++) {
            m3[i][j] += m1[i][k]*m2[k][j];
        }
    }
}

using Vec = std::vector<real_t>;
using Mat = std::vector<Vec>;

void matmul2D() {
    size_t Nx = 11;
    size_t Nc = 12;
    size_t Ny = 13;

    Mat m1(Nx, Vec(Nc)),
        m2(Nc, Vec(Ny)),
        m3(Nx, Vec(Ny));

    for (size_t i = 0; i < Nx; i++) {
    for (size_t j = 0; j < Nc; j++) {
        m1[i][j] = i;
    }}

    for (size_t i = 0; i < Nc; i++) {
    for (size_t j = 0; j < Ny; j++) {
        m2[i][j] = j;
    }}

    // Device

    CudaArray2D<real_t> m1_d {{Nx, Nc}, m1};
    CudaArray2D<real_t> m2_d {{Nc, Ny}, m2};
    CudaArray2D<real_t> m3_d {{Nx, Ny}};

    // Multiply
    dim3 block_size {10,10};
    dim3 grid_size;
    grid_size.x = ceilDiv(Nx, block_size.x);
    grid_size.y = ceilDiv(Ny, block_size.y);
    grid_size.z = 1;

    matmul2D_cuda _CK(grid_size, block_size) (m1_d.getDevicePtr(), m2_d.getDevicePtr(), m3_d.getDevicePtr(), Nx, Nc, Ny);

    // Retrieve result
    m3_d.retrieve(m3);
    
    for (size_t i = 0; i < Nx; i++) {
    for (size_t j = 0; j < Ny; j++) {
        std::cout << m3[i][j] << ((j==Ny-1) ? "\n" : " ");
    }}
    std::cout << std::endl;
}


/*********************************/
/*   VERSION 3 - DeviceNDArray   */
/*********************************/

__global__ void matmul2D_DeviceNDArray_cuda(
    real_t **m1, real_t **m2, real_t**m3,
    ssize_t x0, ssize_t c0, ssize_t y0,
    ssize_t x1, ssize_t c1, ssize_t y1
) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + x0;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + y0;

    if (i < x1 && j < y1) {
        m3[i][j] = 0;
        for (ssize_t k = c0; k < c1; k++) {
            m3[i][j] += m1[i][k] * m2[k][j];
        }
    }
}

void matmul2D_DeviceNDArray() {
    ssize_t x0 = -5, x1 = 6;
    ssize_t c0 = -3, c1 = 9;
    ssize_t y0 = -8, y1 = 5;
    size_t Nx = x1-x0;
    size_t Nc = c1-c0;
    size_t Ny = y1-y0;

    Mat m1(Nx, Vec(Nc)),
        m2(Nc, Vec(Ny)),
        m3(Nx, Vec(Ny));

    for (size_t i = 0; i < Nx; i++) {
    for (size_t j = 0; j < Nc; j++) {
        m1[i][j] = i;
    }}

    for (size_t i = 0; i < Nc; i++) {
    for (size_t j = 0; j < Ny; j++) {
        m2[i][j] = j;
    }}

    // Device
    DeviceNDArray<real_t, 2> m1_d {{{x0,c0},{x1,c1}}};
    DeviceNDArray<real_t, 2> m2_d {{{c0,y0},{c1,y1}}};
    DeviceNDArray<real_t, 2> m3_d {{{x0,y0},{x1,y1}}};
    m1_d.copyFromZero(m1);
    m2_d.copyFromZero(m2);
    
    // Multiply
    dim3 block_size {10,10};
    dim3 grid_size {
        (unsigned int) ceilDiv(Nx, block_size.x),
        (unsigned int) ceilDiv(Ny, block_size.y),
        (unsigned int) 1
    };

    matmul2D_DeviceNDArray_cuda _CK(grid_size, block_size) (
        m1_d.getDeviceMoved(),
        m2_d.getDeviceMoved(),
        m3_d.getDeviceMoved(),
        x0, c0, y0, x1, c1, y1
    );

    // Retrieve result
    m3_d.retrieveToZero(m3);
    
    for (size_t i = 0; i < Nx; i++) {
    for (size_t j = 0; j < Ny; j++) {
        std::cout << m3[i][j] << ((j==Ny-1) ? "\n" : " ");
    }}
    std::cout << std::endl;
}


/********************************************/
/*   VERSION 3 - DeviceNDArray w/ NDArray   */
/********************************************/

void matmul2D_DeviceNDArray_NDArray() {
    ssize_t x0 = -5, x1 = 6;
    ssize_t c0 = -3, c1 = 9;
    ssize_t y0 = -8, y1 = 5;
    size_t Nx = x1-x0;
    size_t Nc = c1-c0;
    size_t Ny = y1-y0;

    NDArray<real_t,2> m1 {{x0,c0}, {x1,c1}};
    NDArray<real_t,2> m2 {{c0,y0}, {c1,y1}};
    NDArray<real_t,2> m3 {{x0,y0}, {x1,y1}};

    for (ssize_t i = x0; i < x1; i++) {
    for (ssize_t j = c0; j < c1; j++) {
        m1[i][j] = i-x0;
    }}

    for (ssize_t i = c0; i < c1; i++) {
    for (ssize_t j = y0; j < y1; j++) {
        m2[i][j] = j-y0;
    }}

    // Device
    DeviceNDArray<real_t, 2> m1_d {{{x0,c0},{x1,c1}}};
    DeviceNDArray<real_t, 2> m2_d {{{c0,y0},{c1,y1}}};
    DeviceNDArray<real_t, 2> m3_d {{{x0,y0},{x1,y1}}};
    m1_d.copy(m1);
    m2_d.copy(m2);
    
    // Multiply
    dim3 block_size {10,10};
    dim3 grid_size {
        (unsigned int) ceilDiv(Nx, block_size.x),
        (unsigned int) ceilDiv(Ny, block_size.y),
        (unsigned int) 1
    };

    matmul2D_DeviceNDArray_cuda _CK(grid_size, block_size) (
        m1_d.getDeviceMoved(),
        m2_d.getDeviceMoved(),
        m3_d.getDeviceMoved(),
        x0, c0, y0, x1, c1, y1
    );

    // Retrieve result
    m3_d.retrieve(m3);
    
    for (ssize_t i = x0; i < x1; i++) {
    for (ssize_t j = y0; j < y1; j++) {
        std::cout << m3[i][j] << ((j==y1-1) ? "\n" : " ");
    }}
    std::cout << std::endl;
}
