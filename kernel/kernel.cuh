#include <cstdint>
#include <cuda_fp16.h>

void execute_kernel(const half *input_tensor,
                    const uint32_t *quantized_weight_tensor,
                    half *output_tensor, const half *scales,
                    const uint32_t *zeros, const int width, const int height,
                    const int groupsize);