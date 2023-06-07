from build import kernel_pybind
import torch

from safetensors.torch import load_file

loaded = load_file(
    "/home/zhaoj/Work/Fast-GPTQ/wizard-vicuna-13B-GPTQ/wizard-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors"
)

# print(loaded.keys())

groupsize = 128
width = 8192
height = 8192
# Cast to int
quantized_height = int(height / 8)
quantized_width = int(width / 8)
groupsize_height = int(height / groupsize)

torch.manual_seed(0)

quantized_weight = loaded["model.layers.0.mlp.down_proj.qweight"]
other = loaded["model.layers.0.mlp.gate_proj.qweight"]

print(quantized_weight.shape)
print(quantized_weight.dtype)

quantized_weight = torch.randint(
    0, 8388608, (quantized_height, width), dtype=torch.int32, device="cuda"
)
scales = torch.randn((groupsize_height, width), dtype=torch.float16, device="cuda")
zero_points = torch.randint(
    0, 8388608, (groupsize_height, width), dtype=torch.int32, device="cuda"
)
input_vector = torch.randn((1, height), dtype=torch.float16, device="cuda")
output_tensor = torch.empty((1, width), dtype=torch.float16, device="cuda")

run_times = 1
for i in range(run_times):
    kernel_pybind.execute_kernel_bridge(
        input_vector.data_ptr(),
        quantized_weight.data_ptr(),
        output_tensor.data_ptr(),
        scales.data_ptr(),
        zero_points.data_ptr(),
        width,
        height,
        groupsize,
    )

# print(output_tensor)
