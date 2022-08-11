#include "fdtd.h"

#include "DeviceNDArray.h"
#include "constants.h"
#include <iostream>

namespace {

size_t ceilDiv(const size_t num, const size_t den) {
    return num/den + (num % den != 0);
}

enum class FieldType {
    None = -1, E, H
};

enum class Orientation {
    None = -1, X, Y, Z
};

struct FieldValue {
    FieldType eh {FieldType::None};
    Orientation xyz {Orientation::None};
    ssize_t i {0}, j {0}, k {0};
    real_t value {0.};

    FieldValue() = default;
    FieldValue(
        const FieldType _eh,const Orientation _xyz,
        const ssize_t _i, const ssize_t _j, const ssize_t _k,
        const real_t _value
    ) :
        eh(_eh), xyz(_xyz), i(_i), j(_j), k(_k), value(_value)
    {}
};

struct Field {
    NDArray<real_t, 3> field {};
    Box<ssize_t, 3> sweep {{0,0,0}, {0,0,0}};
    Box<ssize_t, 3> alloc {{0,0,0}, {0,0,0}};
    real_t C {0.};

    size_t getSweepSize(const Orientation xyz) {
        return sweep[1][(int)xyz] - sweep[0][(int)xyz];
    }

    size_t getAllocSize(const Orientation xyz) {
        return alloc[1][(int)xyz] - alloc[0][(int)xyz];
    }

    void set(const Box<ssize_t, 3>& _sweep, const size_t _eh, const size_t _xyz, const real_t _C) {
        this->sweep = _sweep;
        this->alloc = _sweep;
        if (_eh == 1) {
            this->alloc[0][(_xyz+1)%3] -= 1;
            this->alloc[0][(_xyz+2)%3] -= 1;
            this->alloc[1][(_xyz+1)%3] += 1;
            this->alloc[1][(_xyz+2)%3] += 1;
        }
        field.reset(this->alloc);
        C = _C;
    }

    void set(const Array<ssize_t,3>& _sw0, const Array<ssize_t,3>& _sw1, const size_t _eh, const size_t _xyz, const real_t _C) {
        this->set({_sw0, _sw1}, _eh, _xyz, _C);
    }
};

struct DeviceFullField {
    real_t ***field {nullptr};
    Box<ssize_t, 3> sweep {{0,0,0}, {0,0,0}};
    real_t C {0.};

    DeviceFullField() = default;

    DeviceFullField(real_t*** _f, const Box<ssize_t, 3>& _sw, const real_t _C) :
        field(_f), sweep(_sw), C(_C)
    {}

    void set(real_t*** _f, const Box<ssize_t, 3>& _sw, const real_t _C){
        field = _f;
        sweep = _sw;
        C = _C;
    }

    __hostdev__ bool inSweep(const ssize_t i, const ssize_t j, const ssize_t k) {
        return sweep[0][0] <= i && i < sweep[1][0] &&
               sweep[1][0] <= j && j < sweep[1][1] &&
               sweep[2][0] <= k && k < sweep[1][2];
    }
};


__device__ bool inSweep(
    const ssize_t i, const ssize_t j, const ssize_t k,
    const ssize_t x0, const ssize_t y0, const ssize_t z0,
    const ssize_t x1, const ssize_t y1, const ssize_t z1
) {
        return x0 <= i && i < x1 &&
               y0 <= j && j < y1 &&
               z0 <= k && k < z1;
}

__global__ void FDTDIterationExCuda_alt(
    real_t ***E0, real_t ***H1, real_t ***H2,
    const ssize_t x0, const ssize_t y0, const ssize_t z0,
    const ssize_t x1, const ssize_t y1, const ssize_t z1,
    const real_t C
) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + x0;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + y0;
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + z0;
    
    if (inSweep(i,j,k, x0, y0, z0, x1, y1, z1)) {
        E0[i][j][k] = E0[i][j][k] + C * (
            + H1[i][j  ][k-1]
            - H1[i][j  ][k  ]
            - H2[i][j-1][k  ]
            + H2[i][j  ][k  ]
        );
    }
}

__global__ void FDTDIterationEyCuda_alt(
    real_t ***E0, real_t ***H1, real_t ***H2,
    const ssize_t x0, const ssize_t y0, const ssize_t z0,
    const ssize_t x1, const ssize_t y1, const ssize_t z1,
    const real_t C
) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + x0;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + y0;
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + z0;
    
    if (inSweep(i,j,k, x0, y0, z0, x1, y1, z1)) {
        E0[i][j][k] = E0[i][j][k] + C * (
            + H1[i-1][j][k  ]
            - H1[i  ][j][k  ]
            - H2[i  ][j][k-1]
            + H2[i  ][j][k  ]
        );
    }
}

__global__ void FDTDIterationEzCuda_alt(
    real_t ***E0, real_t ***H1, real_t ***H2,
    const ssize_t x0, const ssize_t y0, const ssize_t z0,
    const ssize_t x1, const ssize_t y1, const ssize_t z1,
    const real_t C
) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + x0;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + y0;
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + z0;
    
    if (inSweep(i,j,k, x0, y0, z0, x1, y1, z1)) {
        E0[i][j][k] = E0[i][j][k] + C * (
            + H1[i  ][j-1][k]
            - H1[i  ][j  ][k]
            - H2[i-1][j  ][k]
            + H2[i  ][j  ][k]
        );
    }
}

__global__ void FDTDIterationHxCuda_alt(
    real_t ***H0, real_t ***E1, real_t ***E2,
    const ssize_t x0, const ssize_t y0, const ssize_t z0,
    const ssize_t x1, const ssize_t y1, const ssize_t z1,
    const real_t C
) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + x0;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + y0;
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + z0;
    
    if (inSweep(i,j,k, x0, y0, z0, x1, y1, z1)) {
        H0[i][j][k] = H0[i][j][k] + C * (
            - E1[i][j  ][k  ]
            + E1[i][j  ][k+1]
            + E2[i][j  ][k  ]
            - E2[i][j+1][k  ]
        );
    }
}

__global__ void FDTDIterationHyCuda_alt(
    real_t ***H0, real_t ***E1, real_t ***E2,
    const ssize_t x0, const ssize_t y0, const ssize_t z0,
    const ssize_t x1, const ssize_t y1, const ssize_t z1,
    const real_t C
) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + x0;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + y0;
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + z0;
    
    if (inSweep(i,j,k, x0, y0, z0, x1, y1, z1)) {
        H0[i][j][k] = H0[i][j][k] + C * (
            - E1[i  ][j][k  ]
            + E1[i+1][j][k  ]
            + E2[i  ][j][k  ]
            - E2[i  ][j][k+1]
        );
    }
}

__global__ void FDTDIterationHzCuda_alt(
    real_t ***H0, real_t ***E1, real_t ***E2,
    const ssize_t x0, const ssize_t y0, const ssize_t z0,
    const ssize_t x1, const ssize_t y1, const ssize_t z1,
    const real_t C
) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + x0;
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + y0;
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + z0;
    
    if (inSweep(i,j,k, x0, y0, z0, x1, y1, z1)) {
        H0[i][j][k] = H0[i][j][k] + C * (
            - E1[i  ][j  ][k]
            + E1[i  ][j+1][k]
            + E2[i  ][j  ][k]
            - E2[i+1][j  ][k]
        );
    }
}


__global__ void FDTDIterationExCuda(DeviceFullField *E0, DeviceFullField *H1, DeviceFullField *H2) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + E0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + E0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + E0->sweep[0][2];
    real_t ***E0_f {E0->field}, ***H1_f{H1->field}, ***H2_f{H2->field};
    real_t C = E0->C;
    
    if (E0->inSweep(i,j,k)) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i][j  ][k-1]
            - H1_f[i][j  ][k  ]
            - H2_f[i][j-1][k  ]
            + H2_f[i][j  ][k  ]
        );
    }
}

__global__ void FDTDIterationEyCuda(DeviceFullField *E0, DeviceFullField *H1, DeviceFullField *H2) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + E0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + E0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + E0->sweep[0][2];
    real_t ***E0_f {E0->field}, ***H1_f{H1->field}, ***H2_f{H2->field};
    real_t C = E0->C;
    
    if (E0->inSweep(i,j,k)) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i-1][j][k  ]
            - H1_f[i  ][j][k  ]
            - H2_f[i  ][j][k-1]
            + H2_f[i  ][j][k  ]
        );
    }
}

__global__ void FDTDIterationEzCuda(DeviceFullField *E0, DeviceFullField *H1, DeviceFullField *H2) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + E0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + E0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + E0->sweep[0][2];
    real_t ***E0_f {E0->field}, ***H1_f{H1->field}, ***H2_f{H2->field};
    real_t C = E0->C;
    
    if (E0->inSweep(i,j,k)) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i  ][j-1][k]
            - H1_f[i  ][j  ][k]
            - H2_f[i-1][j  ][k]
            + H2_f[i  ][j  ][k]
        );
    }
}

__global__ void FDTDIterationHxCuda(DeviceFullField *H0, DeviceFullField *E1, DeviceFullField *E2) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + H0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + H0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + H0->sweep[0][2];
    real_t ***H0_f {H0->field}, ***E1_f{E1->field}, ***E2_f{E2->field};
    real_t C = H0->C;
    
    if (H0->inSweep(i,j,k)) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i][j  ][k  ]
            + E1_f[i][j  ][k+1]
            + E2_f[i][j  ][k  ]
            - E2_f[i][j+1][k  ]
        );
    }
}

__global__ void FDTDIterationHyCuda(DeviceFullField *H0, DeviceFullField *E1, DeviceFullField *E2) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + H0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + H0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + H0->sweep[0][2];
    real_t ***H0_f {H0->field}, ***E1_f{E1->field}, ***E2_f{E2->field};
    real_t C = H0->C;
    
    if (H0->inSweep(i,j,k)) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i  ][j][k  ]
            + E1_f[i+1][j][k  ]
            + E2_f[i  ][j][k  ]
            - E2_f[i  ][j][k+1]
        );
    }
}

__global__ void FDTDIterationHzCuda(DeviceFullField *H0, DeviceFullField *E1, DeviceFullField *E2) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + H0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + H0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + H0->sweep[0][2];
    real_t ***H0_f {H0->field}, ***E1_f{E1->field}, ***E2_f{E2->field};
    real_t C = H0->C;
    
    if (H0->inSweep(i,j,k)) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i  ][j  ][k]
            + E1_f[i  ][j+1][k]
            + E2_f[i  ][j  ][k]
            - E2_f[i+1][j  ][k]
        );
    }
}

void FDTDIterationEx(Field &E0, Field &H1, Field &H2) {
    NDArray<real_t,3> &E0_f {E0.field}, &H1_f{H1.field}, &H2_f{H2.field};
    real_t C = E0.C;
    
    for (ssize_t i = E0.sweep[0][0]; i < E0.sweep[1][0]; i++)
    for (ssize_t j = E0.sweep[0][1]; j < E0.sweep[1][1]; j++)
    for (ssize_t k = E0.sweep[0][2]; k < E0.sweep[1][2]; k++) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i][j  ][k-1]
            - H1_f[i][j  ][k  ]
            - H2_f[i][j-1][k  ]
            + H2_f[i][j  ][k  ]
        );
    }
}

void FDTDIterationEy(Field &E0, Field &H1, Field &H2) {
    NDArray<real_t,3> &E0_f {E0.field}, &H1_f{H1.field}, &H2_f{H2.field};
    real_t C = E0.C;
    
    for (ssize_t i = E0.sweep[0][0]; i < E0.sweep[1][0]; i++)
    for (ssize_t j = E0.sweep[0][1]; j < E0.sweep[1][1]; j++)
    for (ssize_t k = E0.sweep[0][2]; k < E0.sweep[1][2]; k++) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i-1][j][k  ]
            - H1_f[i  ][j][k  ]
            - H2_f[i  ][j][k-1]
            + H2_f[i  ][j][k  ]
        );
    }
}

void FDTDIterationEz(Field &E0, Field &H1, Field &H2) {
    NDArray<real_t,3> &E0_f {E0.field}, &H1_f{H1.field}, &H2_f{H2.field};
    real_t C = E0.C;
    
    for (ssize_t i = E0.sweep[0][0]; i < E0.sweep[1][0]; i++)
    for (ssize_t j = E0.sweep[0][1]; j < E0.sweep[1][1]; j++)
    for (ssize_t k = E0.sweep[0][2]; k < E0.sweep[1][2]; k++) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i  ][j-1][k]
            - H1_f[i  ][j  ][k]
            - H2_f[i-1][j  ][k]
            + H2_f[i  ][j  ][k]
        );
    }
}

void FDTDIterationHx(Field &H0, Field &E1, Field &E2) {
    NDArray<real_t,3> &H0_f {H0.field}, &E1_f{E1.field}, &E2_f{E2.field};
    real_t C = H0.C;
    
    for (ssize_t i = H0.sweep[0][0]; i < H0.sweep[1][0]; i++)
    for (ssize_t j = H0.sweep[0][1]; j < H0.sweep[1][1]; j++)
    for (ssize_t k = H0.sweep[0][2]; k < H0.sweep[1][2]; k++) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i][j  ][k  ]
            + E1_f[i][j  ][k+1]
            + E2_f[i][j  ][k  ]
            - E2_f[i][j+1][k  ]
        );
    }
}

void FDTDIterationHy(Field &H0, Field &E1, Field &E2) {
    NDArray<real_t,3> &H0_f {H0.field}, &E1_f{E1.field}, &E2_f{E2.field};
    real_t C = H0.C;
    
    for (ssize_t i = H0.sweep[0][0]; i < H0.sweep[1][0]; i++)
    for (ssize_t j = H0.sweep[0][1]; j < H0.sweep[1][1]; j++)
    for (ssize_t k = H0.sweep[0][2]; k < H0.sweep[1][2]; k++) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i  ][j][k  ]
            + E1_f[i+1][j][k  ]
            + E2_f[i  ][j][k  ]
            - E2_f[i  ][j][k+1]
        );
    }
}

void FDTDIterationHz(Field &H0, Field &E1, Field &E2) {
    NDArray<real_t,3> &H0_f {H0.field}, &E1_f{E1.field}, &E2_f{E2.field};
    real_t C = H0.C;
    
    for (ssize_t i = H0.sweep[0][0]; i < H0.sweep[1][0]; i++)
    for (ssize_t j = H0.sweep[0][1]; j < H0.sweep[1][1]; j++)
    for (ssize_t k = H0.sweep[0][2]; k < H0.sweep[1][2]; k++) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i  ][j  ][k]
            + E1_f[i  ][j+1][k]
            + E2_f[i  ][j  ][k]
            - E2_f[i+1][j  ][k]
        );
    }
}

class FDTDFields {
public:
    using DeviceField = DeviceNDArray<real_t, 3>;

    FDTDFields() = default;
    ~FDTDFields() {
        for (size_t eh  = 0; eh  < 2; eh ++)
        for (size_t xyz = 0; xyz < 3; xyz++)
            if (fields_full_device[eh][xyz] != nullptr)
                cudaFree(fields_full_device[eh][xyz]);
    }

    void set(const Box<ssize_t, 3>& _bounds, const size_t _Nt, const real_t _Dt, const real_t _Ds, const std::vector<FieldValue> _init = {}) {
        Nt = _Nt;
        C[0] = _Dt / (EPS0*_Ds);
        C[1] = _Dt / (MU0 *_Ds);
        ssize_t x0 = _bounds[0][0];
        ssize_t y0 = _bounds[0][1];
        ssize_t z0 = _bounds[0][2];
        ssize_t x1 = _bounds[1][0];
        ssize_t y1 = _bounds[1][1];
        ssize_t z1 = _bounds[1][2];

        // Host
        Ex.set({x0,y0,z0},{x1  ,y1+1,z1+1}, 0, 0, C[0]);
        Ey.set({x0,y0,z0},{x1+1,y1  ,z1+1}, 0, 1, C[0]);
        Ez.set({x0,y0,z0},{x1+1,y1+1,z1  }, 0, 2, C[0]);
        Hx.set({x0,y0,z0},{x1+1,y1  ,z1  }, 1, 0, C[1]);
        Hy.set({x0,y0,z0},{x1  ,y1+1,z1  }, 1, 1, C[1]);
        Hz.set({x0,y0,z0},{x1  ,y1  ,z1+1}, 1, 2, C[1]);

        for (auto& i : _init)
            fields[(int)i.eh][(int)i.xyz].field[i.i][i.j][i.k] = i.value;

        for (size_t eh  = 0; eh  < 2; eh ++)
        for (size_t xyz = 0; xyz < 3; xyz++) {
            // Set fields in device
            fields_dev[eh][xyz].set (fields[eh][xyz].alloc);
            fields_dev[eh][xyz].copy(fields[eh][xyz].field);
            
            // Set fields w/ alloc (fields full) in device and get pointer
            fields_full[eh][xyz].set(fields_dev[eh][xyz].getDeviceMoved(), fields[eh][xyz].sweep, C[eh]);
            fields_full_device[eh][xyz] = newCUDA<DeviceFullField>(1);
            copyToDeviceCUDA(fields_full_device[eh][xyz], &fields_full[eh][xyz], 1);
        }
    }

    void retrieve() {
        for (size_t eh  = 0; eh  < 2; eh ++)
        for (size_t xyz = 0; xyz < 3; xyz++) {
            fields_dev[eh][xyz].retrieve(fields[eh][xyz].field);
        }
    }

    void free() {
        for (size_t eh  = 0; eh  < 2; eh ++)
        for (size_t xyz = 0; xyz < 3; xyz++) {
            fields_dev[eh][xyz].free();
            cudaFree(&fields_full_device[eh][xyz]);
        }
    }

    void FDTDIterationECuda() {
        dim3 block_size {10,10,10};
        dim3 grid_size;
        
        // Ex
        grid_size.x = ceilDiv(Ex.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ex.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ex.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationExCuda _CK(grid_size, block_size)(Ex_full_device, Hy_full_device, Hz_full_device);
        
        // Ey
        grid_size.x = ceilDiv(Ey.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ey.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ey.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationEyCuda _CK(grid_size, block_size)(Ey_full_device, Hz_full_device, Hx_full_device);

        // Ez
        grid_size.x = ceilDiv(Ez.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ez.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ez.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationEzCuda _CK(grid_size, block_size)(Ez_full_device, Hx_full_device, Hy_full_device);
    }


    void FDTDIterationHCuda() {
        dim3 block_size {10,10,10};
        dim3 grid_size;
        
        // Hx
        grid_size.x = ceilDiv(Hx.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hx.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hx.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHxCuda _CK(grid_size, block_size)(Hx_full_device, Ey_full_device, Ez_full_device);
        
        // Hy
        grid_size.x = ceilDiv(Hy.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hy.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hy.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHyCuda _CK(grid_size, block_size)(Hy_full_device, Ez_full_device, Ex_full_device);

        // Hz
        grid_size.x = ceilDiv(Hz.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hz.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hz.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHzCuda _CK(grid_size, block_size)(Hz_full_device, Ex_full_device, Ey_full_device);
    }

    void FDTDIterationECuda_alt() {
        dim3 block_size {10,10,10};
        dim3 grid_size;
        
        // Ex
        grid_size.x = ceilDiv(Ex.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ex.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ex.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationExCuda_alt _CK(grid_size, block_size)(
            Ex_dev.getDeviceMoved(), Hy_dev.getDeviceMoved(), Hz_dev.getDeviceMoved(),
            Ex.sweep[0][0], Ex.sweep[0][1], Ex.sweep[0][2],
            Ex.sweep[1][0], Ex.sweep[1][1], Ex.sweep[1][2],
            C[0]
        );
        
        // Ey
        grid_size.x = ceilDiv(Ey.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ey.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ey.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationEyCuda_alt _CK(grid_size, block_size)(
            Ey_dev.getDeviceMoved(), Hz_dev.getDeviceMoved(), Hx_dev.getDeviceMoved(),
            Ey.sweep[0][0], Ey.sweep[0][1], Ey.sweep[0][2],
            Ey.sweep[1][0], Ey.sweep[1][1], Ey.sweep[1][2],
            C[0]
        );

        // Ez
        grid_size.x = ceilDiv(Ez.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ez.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ez.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationEzCuda_alt _CK(grid_size, block_size)(
            Ez_dev.getDeviceMoved(), Hx_dev.getDeviceMoved(), Hy_dev.getDeviceMoved(),
            Ez.sweep[0][0], Ez.sweep[0][1], Ez.sweep[0][2],
            Ez.sweep[1][0], Ez.sweep[1][1], Ez.sweep[1][2],
            C[0]
        );
    }


    void FDTDIterationHCuda_alt() {
        dim3 block_size {10,10,10};
        dim3 grid_size;
        
        // Hx
        grid_size.x = ceilDiv(Hx.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hx.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hx.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHxCuda_alt _CK(grid_size, block_size)(
            Hx_dev.getDeviceMoved(), Ey_dev.getDeviceMoved(), Ez_dev.getDeviceMoved(),
            Hx.sweep[0][0], Hx.sweep[0][1], Hx.sweep[0][2],
            Hx.sweep[1][0], Hx.sweep[1][1], Hx.sweep[1][2],
            C[1]
        );
        
        // Hy
        grid_size.x = ceilDiv(Hy.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hy.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hy.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHyCuda_alt _CK(grid_size, block_size)(
            Hy_dev.getDeviceMoved(), Ez_dev.getDeviceMoved(), Ex_dev.getDeviceMoved(),
            Hy.sweep[0][0], Hy.sweep[0][1], Hy.sweep[0][2],
            Hy.sweep[1][0], Hy.sweep[1][1], Hy.sweep[1][2],
            C[1]
        );

        // Hz
        grid_size.x = ceilDiv(Hz.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hz.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hz.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHzCuda_alt _CK(grid_size, block_size)(
            Hz_dev.getDeviceMoved(), Ex_dev.getDeviceMoved(), Ey_dev.getDeviceMoved(),
            Hz.sweep[0][0], Hz.sweep[0][1], Hz.sweep[0][2],
            Hz.sweep[1][0], Hz.sweep[1][1], Hz.sweep[1][2],
            C[1]
        );
    }

    void FDTDIterationE() {
        FDTDIterationEx(Ex, Hy, Hz);
        FDTDIterationEy(Ey, Hz, Hx);
        FDTDIterationEz(Ez, Hx, Hy);
    }

    void FDTDIterationH() {
        FDTDIterationHx(Hx, Ey, Ez);
        FDTDIterationHy(Hy, Ez, Ex);
        FDTDIterationHz(Hz, Ex, Ey);
    }

    void runCuda() {
        for (size_t nt = 0; nt < Nt; nt++) {
            FDTDIterationECuda();
            FDTDIterationHCuda();
        }
        retrieve();
    }

    void runCuda_alt() {
        for (size_t nt = 0; nt < Nt; nt++) {
            FDTDIterationECuda_alt();
            FDTDIterationHCuda_alt();
        }
        retrieve();
    }

    void run() {
        for (size_t nt = 0; nt < Nt; nt++) {
            FDTDIterationE();
            FDTDIterationH();
        }
    }

    Field&       getField(FieldType _eh, Orientation _xyz)       {return fields[(int)_eh][(int)_xyz];}
    const Field& getField(FieldType _eh, Orientation _xyz) const {return fields[(int)_eh][(int)_xyz];}

    DeviceFullField*       getDeviceField(FieldType _eh, Orientation _xyz)       {return fields_full_device[(int)_eh][(int)_xyz];}
    const DeviceFullField* getDeviceField(FieldType _eh, Orientation _xyz) const {return fields_full_device[(int)_eh][(int)_xyz];}
    
//private:
    Box<Field, 3> fields {};
    Box<DeviceField, 3> fields_dev {};
    Box<DeviceFullField, 3> fields_full {};
    Box<DeviceFullField*, 3> fields_full_device {};
    Array<real_t,2> C{0.,0.};
    size_t Nt {0};

    // References
    Array<Field,3>& E {fields[0]};
    Array<Field,3>& H {fields[1]};
    Field& Ex {fields[0][0]};
    Field& Ey {fields[0][1]};
    Field& Ez {fields[0][2]};
    Field& Hx {fields[1][0]};
    Field& Hy {fields[1][1]};
    Field& Hz {fields[1][2]};

    Array<DeviceField,3>& E_dev {fields_dev[0]};
    Array<DeviceField,3>& H_dev {fields_dev[1]};
    DeviceField& Ex_dev {fields_dev[0][0]};
    DeviceField& Ey_dev {fields_dev[0][1]};
    DeviceField& Ez_dev {fields_dev[0][2]};
    DeviceField& Hx_dev {fields_dev[1][0]};
    DeviceField& Hy_dev {fields_dev[1][1]};
    DeviceField& Hz_dev {fields_dev[1][2]};

    Array<DeviceFullField,3>& E_full {fields_full[0]};
    Array<DeviceFullField,3>& H_full {fields_full[1]};
    DeviceFullField& Ex_full {fields_full[0][0]};
    DeviceFullField& Ey_full {fields_full[0][1]};
    DeviceFullField& Ez_full {fields_full[0][2]};
    DeviceFullField& Hx_full {fields_full[1][0]};
    DeviceFullField& Hy_full {fields_full[1][1]};
    DeviceFullField& Hz_full {fields_full[1][2]};

    Array<DeviceFullField*,3>& E_full_device {fields_full_device[0]};
    Array<DeviceFullField*,3>& H_full_device {fields_full_device[1]};
    DeviceFullField*& Ex_full_device {fields_full_device[0][0]};
    DeviceFullField*& Ey_full_device {fields_full_device[0][1]};
    DeviceFullField*& Ez_full_device {fields_full_device[0][2]};
    DeviceFullField*& Hx_full_device {fields_full_device[1][0]};
    DeviceFullField*& Hy_full_device {fields_full_device[1][1]};
    DeviceFullField*& Hz_full_device {fields_full_device[1][2]};
};

void compare(const FDTDFields& fdtd_nocuda, const FDTDFields& fdtd_cuda, const FDTDFields& fdtd_cuda_alt) {
    std::cout << fdtd_nocuda  .Ex.field[1][2][3] << " ";
    std::cout << fdtd_cuda    .Ex.field[1][2][3] << " ";
    std::cout << fdtd_cuda_alt.Ex.field[1][2][3] << std::endl;

    std::cout << fdtd_nocuda  .Hy.field[1][2][3] << " ";
    std::cout << fdtd_cuda    .Hy.field[1][2][3] << " ";
    std::cout << fdtd_cuda_alt.Hy.field[1][2][3] << std::endl;
}

} // namespace

void runFDTDTest() {
    // Params
    Box<ssize_t, 3> sim_boundaries {{0,0,0}, {10,10,10}};
    size_t Nt = 100;
    real_t Dt = 1e-10;
    real_t Ds = 0.1;
    std::vector<FieldValue> init_fields {
        {FieldType::E, Orientation::X, 1, 2, 3,  2.50},
        {FieldType::H, Orientation::Y, 1, 2, 3, -0.33}
    };

    FDTDFields fdtd_nocuda, fdtd_cuda, fdtd_cuda_alt;

    fdtd_nocuda  .set(sim_boundaries, Nt, Dt, Ds, init_fields);
    fdtd_cuda    .set(sim_boundaries, Nt, Dt, Ds, init_fields);
    fdtd_cuda_alt.set(sim_boundaries, Nt, Dt, Ds, init_fields);
    
    fdtd_nocuda  .run();
    fdtd_cuda    .runCuda();
    fdtd_cuda_alt.runCuda_alt();

    compare(fdtd_nocuda, fdtd_cuda, fdtd_cuda_alt);
}