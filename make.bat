@echo off

cmake^
    -B./build/ -S./
    rem -T "Intel C++ Compiler 19.2" ^
    rem -DCUDAToolkit_ROOT="D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
    rem -DCMAKE_CXX_COMPILER="icl"
