#include "fdtd.h"

#include "DeviceNDArray.h"
#include "constants.h"

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

    size_t getSweepSize(const Orientation xyz) {
        return sweep[1][(int)xyz] - sweep[0][(int)xyz];
    }

    size_t getAllocSize(const Orientation xyz) {
        return alloc[1][(int)xyz] - alloc[0][(int)xyz];
    }

    void set(const Box<ssize_t, 3>& _sweep, const size_t eh, const size_t _xyz) {
        this->sweep = _sweep;
        this->alloc = _sweep;
        if (eh == 1) {
            this->alloc[0][(_xyz+1)%3] -= 1;
            this->alloc[0][(_xyz+2)%3] -= 1;
            this->alloc[1][(_xyz+1)%3] += 1;
            this->alloc[1][(_xyz+2)%3] += 1;
        }
    }
};

struct DeviceFullField {
    real_t ***field;
    Box<ssize_t, 3> sweep {{0,0,0}, {0,0,0}};

    DeviceFullField() = default;

    DeviceFullField(real_t*** _f, const Box<ssize_t, 3>& _sw) :
        field(_f), sweep(_sw)
    {}

    void set(real_t*** _f, const Box<ssize_t, 3>& _sw){
        field = _f;
        sweep = _sw;
    }

    __hostdev__ bool inSweep(const ssize_t i, const ssize_t j, const ssize_t k) {
        return sweep[0][0] <= i && i < sweep[1][0] &&
               sweep[1][0] <= j && j < sweep[1][1] &&
               sweep[2][0] <= k && k < sweep[1][2];
    }
};


__global__ void FDTDIterationEx(DeviceFullField *E0, DeviceFullField *H1, DeviceFullField *H2, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + E0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + E0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + E0->sweep[0][2];
    real_t ***E0_f {E0->field}, ***H1_f{H1->field}, ***H2_f{H2->field};
    
    if (E0->inSweep(i,j,k)) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i][j  ][k-1]
            - H1_f[i][j  ][k  ]
            - H2_f[i][j-1][k  ]
            + H2_f[i][j  ][k  ]
        );
    }
}

__global__ void FDTDIterationEy(DeviceFullField *E0, DeviceFullField *H1, DeviceFullField *H2, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + E0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + E0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + E0->sweep[0][2];
    real_t ***E0_f {E0->field}, ***H1_f{H1->field}, ***H2_f{H2->field};
    
    if (E0->inSweep(i,j,k)) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i-1][j][k  ]
            - H1_f[i  ][j][k  ]
            - H2_f[i  ][j][k-1]
            + H2_f[i  ][j][k  ]
        );
    }
}

__global__ void FDTDIterationEz(DeviceFullField *E0, DeviceFullField *H1, DeviceFullField *H2, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + E0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + E0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + E0->sweep[0][2];
    real_t ***E0_f {E0->field}, ***H1_f{H1->field}, ***H2_f{H2->field};
    
    if (E0->inSweep(i,j,k)) {
        E0_f[i][j][k] = E0_f[i][j][k] + C * (
            + H1_f[i  ][j-1][k]
            - H1_f[i  ][j  ][k]
            - H2_f[i-1][j  ][k]
            + H2_f[i  ][j  ][k]
        );
    }
}

__global__ void FDTDIterationHx(DeviceFullField *H0, DeviceFullField *E1, DeviceFullField *E2, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + H0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + H0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + H0->sweep[0][2];
    real_t ***H0_f {H0->field}, ***E1_f{E1->field}, ***E2_f{E2->field};
    
    if (H0->inSweep(i,j,k)) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i][j  ][k  ]
            + E1_f[i][j  ][k+1]
            + E2_f[i][j  ][k  ]
            - E2_f[i][j+1][k  ]
        );
    }
}

__global__ void FDTDIterationHy(DeviceFullField *H0, DeviceFullField *E1, DeviceFullField *E2, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + H0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + H0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + H0->sweep[0][2];
    real_t ***H0_f {H0->field}, ***E1_f{E1->field}, ***E2_f{E2->field};
    
    if (H0->inSweep(i,j,k)) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[i  ][j][k  ]
            + E1_f[i+1][j][k  ]
            + E2_f[i  ][j][k  ]
            - E2_f[i  ][j][k+1]
        );
    }
}

__global__ void FDTDIterationHz(DeviceFullField *H0, DeviceFullField *E1, DeviceFullField *E2, real_t C) {
    ssize_t i = blockDim.x * blockIdx.x + threadIdx.x + H0->sweep[0][0];
    ssize_t j = blockDim.y * blockIdx.y + threadIdx.y + H0->sweep[0][1];
    ssize_t k = blockDim.z * blockIdx.z + threadIdx.z + H0->sweep[0][2];
    real_t ***H0_f {H0->field}, ***E1_f{E1->field}, ***E2_f{E2->field};
    
    if (H0->inSweep(i,j,k)) {
        H0_f[i][j][k] = H0_f[i][j][k] + C * (
            - E1_f[k  ][i  ][j]
            + E1_f[k  ][i+1][j]
            + E2_f[k  ][i  ][j]
            - E2_f[k+1][i  ][j]
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
        Ce = _Dt / (EPS0*_Ds);
        Ch = _Dt / (MU0 *_Ds);
        ssize_t x0 = _bounds[0][0];
        ssize_t y0 = _bounds[0][1];
        ssize_t z0 = _bounds[0][2];
        ssize_t x1 = _bounds[1][0];
        ssize_t y1 = _bounds[1][1];
        ssize_t z1 = _bounds[1][2];

        // Host
        Ex.set({{x0,y0,z0},{x1  ,y1+1,z1+1}}, 0, 0);
        Ey.set({{x0,y0,z0},{x1+1,y1  ,z1+1}}, 0, 1);
        Ez.set({{x0,y0,z0},{x1+1,y1+1,z1  }}, 0, 2);
        Hx.set({{x0,y0,z0},{x1+1,y1  ,z1  }}, 1, 0);
        Hy.set({{x0,y0,z0},{x1  ,y1+1,z1  }}, 1, 1);
        Hz.set({{x0,y0,z0},{x1  ,y1  ,z1+1}}, 1, 2);

        for (auto& i : _init)
            fields[(int)i.eh][(int)i.xyz].field[i.i][i.j][i.k] = i.value;

        // Device
        Ex_dev.set({{x0,y0,z0},{x1  ,y1+1,z1+1}});
        Ey_dev.set({{x0,y0,z0},{x1+1,y1  ,z1+1}});
        Ez_dev.set({{x0,y0,z0},{x1+1,y1+1,z1  }});
        Hx_dev.set({{x0,y0,z0},{x1+1,y1  ,z1  }});
        Hy_dev.set({{x0,y0,z0},{x1  ,y1+1,z1  }});
        Hz_dev.set({{x0,y0,z0},{x1  ,y1  ,z1+1}});

        for (size_t eh  = 0; eh  < 2; eh ++)
        for (size_t xyz = 0; xyz < 3; xyz++)
            fields_dev[eh][xyz].copy(fields[eh][xyz].field);

        // Full
        for (size_t eh  = 0; eh  < 2; eh ++)
        for (size_t xyz = 0; xyz < 3; xyz++) {
            fields_full[eh][xyz].set(fields_dev[eh][xyz].getDeviceMoved(), fields[eh][xyz].sweep);
            fields_full_device[eh][xyz] = newCUDA<DeviceFullField>(1);
            copyToDeviceCUDA(fields_full_device[eh][xyz], &fields_full[eh][xyz], 1);
        }

    }

    void FDTDIterationE() {
        dim3 block_size {10,10,10};
        dim3 grid_size;
        
        // Ex
        grid_size.x = ceilDiv(Ex.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ex.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ex.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationEx _CK(grid_size, block_size)(Ex_full_device, Hy_full_device, Hz_full_device, Ce);
        
        // Ey
        grid_size.x = ceilDiv(Ey.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ey.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ey.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationEy _CK(grid_size, block_size)(Ey_full_device, Hz_full_device, Hx_full_device, Ce);

        // Ez
        grid_size.x = ceilDiv(Ez.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Ez.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Ez.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationEz _CK(grid_size, block_size)(Ez_full_device, Hx_full_device, Hy_full_device, Ce);
    }


    void FDTDIterationH() {
        dim3 block_size {10,10,10};
        dim3 grid_size;
        
        // Hx
        grid_size.x = ceilDiv(Hx.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hx.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hx.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHx _CK(grid_size, block_size)(Hx_full_device, Ey_full_device, Ez_full_device, Ce);
        
        // Hy
        grid_size.x = ceilDiv(Hy.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hy.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hy.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHy _CK(grid_size, block_size)(Hy_full_device, Ez_full_device, Ex_full_device, Ce);

        // Hz
        grid_size.x = ceilDiv(Hz.getSweepSize(Orientation::X), block_size.x);
        grid_size.y = ceilDiv(Hz.getSweepSize(Orientation::Y), block_size.y);
        grid_size.z = ceilDiv(Hz.getSweepSize(Orientation::Z), block_size.z);
        FDTDIterationHz _CK(grid_size, block_size)(Hz_full_device, Ex_full_device, Ey_full_device, Ce);
    }

    void runCuda() {
        for (size_t nt = 0; nt < Nt; nt++) {
            
        }
    }

    Field&       getField(FieldType _eh, Orientation _xyz)       {return fields[(int)_eh][(int)_xyz];}
    const Field& getField(FieldType _eh, Orientation _xyz) const {return fields[(int)_eh][(int)_xyz];}

    DeviceFullField*       getDeviceField(FieldType _eh, Orientation _xyz)       {return fields_full_device[(int)_eh][(int)_xyz];}
    const DeviceFullField* getDeviceField(FieldType _eh, Orientation _xyz) const {return fields_full_device[(int)_eh][(int)_xyz];}
    
private:
    Box<Field, 3> fields {};
    Box<DeviceField, 3> fields_dev {};
    Box<DeviceFullField, 3> fields_full {};
    Box<DeviceFullField*, 3> fields_full_device {};
    real_t Ce{0.}, Ch {0.};
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

} // namespace

void runFDTDTest() {
    // Params
    Box<ssize_t, 3> sim_boundaries {{0,0,0}, {10,10,10}};
    size_t Nt = 100;
    real_t Dt = 1e-7;
    real_t Ds = 0.1;
    std::vector<FieldValue> init_fields {
        {FieldType::E, Orientation::X, 1, 2, 3,  2.50},
        {FieldType::H, Orientation::Y, 1, 2, 3, -0.33}
    };

    FDTDFields fdtd_fields;
    fdtd_fields.set(sim_boundaries, Nt, Dt, Ds, init_fields);

    fdtd_fields.runCuda();

    //CudaArray3D Ex_cuda (Ex)
}