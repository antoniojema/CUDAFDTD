set(HeaderList
    "common.h"
    "typedef.h"
    "constants.h"
    "cuda_utils.h"
    "CudaVector.h"
    "CudaVector_impl.h"
    "fdtd.h"
)

set(SRCList
    "main.cu"
    "fdtd.cu"
)


# Add executable target
set(TargetName ${ExeName})
add_executable(
    ${TargetName}
    ${HeaderList}
    ${SRCList}
    "CMakeLists.txt"
    "../CMakeLists.txt"
)

# (Windows) Gestionamos los codigos dentro del proyecto
IF (WIN32)
    source_group("" FILES ${SRCList} ${HeaderList})
    source_group("CMakeLists/" FILES "CMakeLists.txt" "../CMakeLists.txt")
ENDIF()
