cmake_minimum_required(VERSION 3.21)
project(StrassenCUDA CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
FIND_PACKAGE(OpenMP REQUIRED)
IF (OPENMP_FOUND)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
add_executable(StrassenCUDA main.cu Utils.h Matrix.h lib/VariadicTable.h)

#set_target_properties(StrassenCUDA PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
