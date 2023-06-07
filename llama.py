from build import kernel_pybind
import torch

print("Hello from Python!")

groupsize = 128
dim = 5120
# Cast to int
quantized_dim = int(dim / 8)
groupsize_dim = int(dim / groupsize)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



kernel_pybind.execute_kernel_bridge(
    input_vector.data_ptr(),
    quantized_weight.data_ptr(),
    output_tensor.data_ptr(),
    scales.data_ptr(),
    zero_points.data_ptr(),
    dim,
    groupsize,
)

print(output_tensor)