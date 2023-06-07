cmake_minimum_required(VERSION 3.26)

project(test_cuda_kernel LANGUAGES CXX CUDA)

# Set CMAKE_EXPORT_COMPILE_COMMANDS to ON to generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set these so nvcc does not use response file for includes
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# Set CUDA Flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

set(CMAKE_ENABLE_EXPORTS ON)

# Find CCache

# If you have ccache installed, you can speed up recompilation a lot.
message(STATUS "Looking for ccache...")
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Found ccache: ${CCACHE_FOUND}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
else()
    message(STATUS "Could not find ccache")
endif()

# Find Python3
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find PythonLibs
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find Torch (This is a more reliable way to find Torch than using find_package(Torch) alone)
if(Python_Interpreter_FOUND)
  message(STATUS "Python Interpreter: ${Python3_EXECUTABLE}")

  execute_process(COMMAND ${Python_EXECUTABLE} -c 
    "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # Print output of torch.utils.cmake_prefix_path
  message(STATUS "TORCH_CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")

  set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Adds main.cpp as an executable target
add_executable(test_cuda_kernel kernel/main.cpp)
target_sources(test_cuda_kernel PUBLIC kernel/kernel.cu kernel/old_kernel/quant_cuda_kernel.cu)
target_link_libraries(test_cuda_kernel ${TORCH_LIBRARIES})

# Add pybind11 module for import from Python
pybind11_add_module(kernel_pybind kernel/kernel_pybind.cpp kernel/kernel.cu)
target_link_libraries(kernel_pybind PRIVATE ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})


# Disable CMake Automatic CUDA Architecture Detection
set_property(TARGET test_cuda_kernel PROPERTY CUDA_ARCHITECTURES OFF)