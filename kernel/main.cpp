#include <torch/types.h>

#include "kernel.cuh"
#include "old_kernel/quant_cuda_kernel.cuh"

int main(int argc, char **argv) {

  // Ensure argv[1] is a integer.
  if (argc != 2) {
    std::cout << "Usage: ./main <dim>\n";
    return 1;
  }
  int dim = std::stoi(argv[1]);
  int groupsize = 128;

  torch::manual_seed(0);

  torch::Tensor quantized_weight = torch::randint(
      0, 8388608, {dim / 8, dim},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  torch::Tensor scales = torch::randn(
      {dim / groupsize, dim},
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

  torch::Tensor zero_points = torch::randint(
      0, 8388608, {dim / groupsize, dim / 8},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  torch::Tensor input_vector = torch::randn(
      {1, dim},
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

  torch::Tensor output_tensor = torch::empty(
      {1, dim},
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

  torch::Tensor output_tensor_2 = torch::empty(
      {1, dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  execute_kernel((half *)input_vector.data_ptr(),
                 (uint32_t *)quantized_weight.data_ptr(),
                 (half *)output_tensor.data_ptr(), (half *)scales.data_ptr(),
                 (uint32_t *)zero_points.data_ptr(), dim, groupsize);

  vecquant4matmul_cuda((half2 *)input_vector.data_ptr(),
                       quantized_weight.data_ptr<int>(),
                       output_tensor_2.data_ptr<float>(),
                       scales.to(torch::kFloat32).data_ptr<float>(),
                       zero_points.data_ptr<int>(), groupsize, dim, 1);

  // Print out first 5 elements of output tensor.
  std::cout << "Output tensor 1: " << output_tensor.slice(1, 0, 5) << "\n";

  // Print out first 5 elements of output tensor 2.
  std::cout << "Output tensor 2: "
            << output_tensor_2.to(torch::kFloat16).slice(1, 0, 5) << "\n";

  return 0;
}