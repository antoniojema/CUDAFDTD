#include "fdtd.h"

void matmul2D();
void matmul();
void matmul2D_DeviceNDArray();

int main() {
    matmul2D_DeviceNDArray();
    matmul();
    matmul2D();
    matmul2D_DeviceNDArray();
    //runFDTDTest();
}
