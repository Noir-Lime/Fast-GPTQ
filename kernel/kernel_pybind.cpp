
#include <pybind11/pybind11.h>

#include "kernel.cuh"

void execute_kernel_bridge(uint64_t input_vector, uint64_t quantized_weight,
                           uint64_t output_tensor, uint64_t scales,
                           uint64_t zero_points, int64_t width, int64_t height,
                           int64_t groupsize) {

  execute_kernel((half *)input_vector, (uint32_t *)quantized_weight,
                 (half *)output_tensor, (half *)scales, (uint32_t *)zero_points,
                 width, height, groupsize);
}

PYBIND11_MODULE(kernel_pybind, module_) {
  module_.def(
      "execute_kernel_bridge", &execute_kernel_bridge,
      "Execute kernel with given input, weight, scales, zero_points, dim, "
      "groupsize");
}