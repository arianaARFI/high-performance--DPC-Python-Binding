cmake_minimum_required(VERSION 3.18)
project(minilb_step_pybind11 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MINILB_ENABLE_CUDA "Build miniLB SYCL extension for NVIDIA CUDA backend" OFF)
set(MINILB_CUDA_ARCH "sm_80" CACHE STRING "NVIDIA CUDA architecture, e.g. sm_80 for A100")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import dpctl, sys; sys.stdout.write(dpctl.get_include())"
    OUTPUT_VARIABLE DPCTL_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "dpctl include dir: ${DPCTL_INCLUDE_DIR}")
message(STATUS "MINILB_ENABLE_CUDA: ${MINILB_ENABLE_CUDA}")
message(STATUS "MINILB_CUDA_ARCH: ${MINILB_CUDA_ARCH}")

pybind11_add_module(_minilb_step _minilb_step.cpp)

target_include_directories(_minilb_step PRIVATE ${DPCTL_INCLUDE_DIR})

set(MINILB_SYCL_COMPILE_OPTIONS
    -fsycl
)

set(MINILB_SYCL_LINK_OPTIONS
    -fsycl
)

if(MINILB_ENABLE_CUDA)
    message(STATUS "Building with NVIDIA CUDA SYCL target")

    list(APPEND MINILB_SYCL_COMPILE_OPTIONS
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend=nvptx64-nvidia-cuda
        --offload-arch=${MINILB_CUDA_ARCH}
    )

    list(APPEND MINILB_SYCL_LINK_OPTIONS
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend=nvptx64-nvidia-cuda
        --offload-arch=${MINILB_CUDA_ARCH}
    )
else()
    message(STATUS "Building default SYCL target, suitable for current CPU/OpenCL setup")
endif()

target_compile_options(_minilb_step PRIVATE ${MINILB_SYCL_COMPILE_OPTIONS})
target_link_options(_minilb_step PRIVATE ${MINILB_SYCL_LINK_OPTIONS})
