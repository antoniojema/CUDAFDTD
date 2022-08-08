#pragma once

#ifdef __INTELLISENSE__
    #include <cuda_runtime.h>
    #include "device_launch_parameters.h"
    #define _CK(...)
#else
    #define _CK(...) <<<__VA_ARGS__>>>
#endif

