#include "fdtd.h"

void matmul2D();
void matmul();
void matmul2D_negative();

int main() {
    matmul();
    matmul2D();
    matmul2D_negative();
    runFDTDTest();
}
