import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
import json
from enum import Enum

optimal_switch_thd = 6


class ParsedEnum(Enum):
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @classmethod
    def argparse(cls, s):
        try:
            return cls[s.upper()]
        except KeyError:
            return s


class ExLlamaConfig:
    class MatmulMethod(ParsedEnum):
        QUANT_ONLY = 1  # Use the quantized matmul
        SWITCHED = 2  # Switch between quantized matmul and FP16 reconstruction (best)
        PYTORCH_ONLY = 3  # Always reconstruct and perform FP16 matmul

    # Load config from Llama config.json

    def __init__(self, model_config_path):
        with open(model_config_path) as f:
            read_config = json.load(f)

        # Loaded/automatic settings

        self.bos_token_id = read_config[
            "bos_token_id"
        ]  # Note that the HF LlamaTokenizer doesn't seem to recognize these automatically
        self.eos_token_id = read_config["eos_token_id"]
        self.pad_token_id = read_config["pad_token_id"]

        self.hidden_size = read_config["hidden_size"]
        self.initializer_range = read_config["initializer_range"]
        self.intermediate_size = read_config["intermediate_size"]
        self.num_attention_heads = read_config["num_attention_heads"]
        self.num_hidden_layers = read_config["num_hidden_layers"]
        self.num_attention_heads = read_config["num_attention_heads"]
        self.rms_norm_eps = read_config["rms_norm_eps"]
        self.vocab_size = read_config["vocab_size"]

        self.rotary_embedding_base = (
            10000  # Constant used for pretrained models, leave as is unless retraining
        )
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.groupsize = None  # Autodetected
        self.act_order = False  # Autodetected

        # Required settings

        self.model_path: str | None = None

        # Optional settings

        self.stream_layer_interval = 0  # Store every nth layer in system RAM and
        self.max_seq_len = 2048  # Reduce to save memory. Can also be increased, but the pretrained models produce degenerate output after 2048 tokens in any case. Should be possible to finetune for longer sequence lengths.
        self.matmul_method = self.MatmulMethod.SWITCHED
        self.device_map = ExLlamaDeviceMap(self.num_hidden_layers)
        self.auto_map = None  # List of ints with memory allocation in GB, per CUDA device, overrides device_map


# Switching
def _matmul_switch(config, x):
    if config.matmul_method == ExLlamaConfig.MatmulMethod.QUANT_ONLY:
        return False
    if config.matmul_method == ExLlamaConfig.MatmulMethod.PYTORCH_ONLY:
        return True

    xdp = 1
    for y in x.shape[:-1]:
        xdp *= y
    return xdp > optimal_switch_thd


# 4-bit linear layer implementation


# class Ex4bitLinear:
class Ex4bitLinear(nn.Module):
    def __init__(self, config, in_features, out_features, has_bias, tensors, key):
        super().__init__()

        self.config = config
        self.key = key

        self.in_features = in_features
        self.out_features = out_features
        self.bits = 4  # Only support 4 bits for now

        self.maxq = 2**self.bits - 1
        self.bias = None
        self.x_map = None
        self.seq_g_idx = None

        self.qweight = tensors[key + ".qweight"]

        self.qzeros = tensors[key + ".qzeros"]
        self.scales = tensors[key + ".scales"]

        # Infer groupsize from height of qzeros

        self.groupsize = None
        if self.qzeros.shape[0] > 1:
            self.groupsize = (self.qweight.shape[0] * 8) // self.qzeros.shape[0]

            if self.config.groupsize is None:
                self.config.groupsize = self.groupsize
            else:
                if self.config.groupsize != self.groupsize:
                    self.config.no_groupsize = True

        # Handle act-order matrix

        # if key + ".g_idx" in tensors:
        #     if self.groupsize is None:
        #         raise ValueError("Found group index but no groupsize. What do?")

        #     self.config.act_order = True

        #     # Rearrange groups sequentially for act-order matrices

        #     g_idx = tensors[key + ".g_idx"]
        #     num_groups = self.qzeros.shape[0]
        #     seq_g_idx, self.x_map = cuda_ext.sequential_q4v2(
        #         self.qweight, g_idx, num_groups
        #     )

        #     # Discard group index if sequential groups all have the same groupsize. Treat as regular groupsize
        #     # matrix but keep the x_map

        #     i = 0
        #     j = 0
        #     discard = True
        #     while i < seq_g_idx.shape[-1]:
        #         if (
        #             seq_g_idx[i].item() != j
        #             or seq_g_idx[i + 1].item() != self.groupsize
        #         ):
        #             discard = False
        #             break
        #         i += self.groupsize * 2
        #         j += 1

        #     if not discard:
        #         self.seq_g_idx = seq_g_idx

        # Bias

        if has_bias:
            self.bias = tensors[key + ".bias"]

    def quant_args(self):
        return {
            "qweight": self.qweight,
            "scales": self.scales,
            "zeros": self.qzeros,
            "seq_g_idx": self.seq_g_idx,
            "x_map": self.x_map,
        }

    def forward(self, x):
        # out = cuda_ext.matmul_q4v2(x, self.quant_args(), _matmul_switch(self.config, x))

        return x


# Llama MLP


# class ExLlamaMLP:
class ExLlamaMLP(nn.Module):
    def __init__(self, config, tensors, key):
        super().__init__()

        self.config = config

        self.gate_proj = Ex4bitLinear(
            config,
            self.config.hidden_size,
            self.config.intermediate_size,
            False,
            tensors,
            key + ".gate_proj",
        )
        self.up_proj = Ex4bitLinear(
            config,
            self.config.hidden_size,
            self.config.intermediate_size,
            False,
            tensors,
            key + ".up_proj",
        )
        self.down_proj = Ex4bitLinear(
            config,
            self.config.intermediate_size,
            self.config.hidden_size,
            False,
            tensors,
            key + ".down_proj",
        )

        self.act_fn = nn.SiLU()

    def forward(self, x):
        y = self.gate_proj.forward(x)
        y = self.act_fn(y)
        y *= self.up_proj.forward(x)
        y = self.down_proj.forward(y)


# RMS Layer norm.


# class ExLlamaRMSNorm:
class ExLlamaRMSNorm(nn.Module):
    def __init__(self, config, tensors, key):
        super().__init__()

        self.config = config
        self.variance_epsilon = self.config.rms_norm_eps
        self.weight = tensors[key]

    def forward(self, hidden_states, buffer):
        # hidden_states = cuda_ext.llama_rms_norm(
        #     hidden_states, self.weight, self.variance_epsilon
        # )
        return hidden_states


# Llama attention


# class ExLlamaAttention:
class ExLlamaAttention(nn.Module):
    def __init__(self, config, tensors, key, sin, cos, index):
        super().__init__()

        self.config = config
        self.sin = sin
        self.cos = cos
        self.index = index

        self.q_proj = Ex4bitLinear(
            config,
            self.config.hidden_size,
            self.config.num_attention_heads * self.config.head_dim,
            False,
            tensors,
            key + ".q_proj",
        )
        self.k_proj = Ex4bitLinear(
            config,
            self.config.hidden_size,
            self.config.num_attention_heads * self.config.head_dim,
            False,
            tensors,
            key + ".k_proj",
        )
        self.v_proj = Ex4bitLinear(
            config,
            self.config.hidden_size,
            self.config.num_attention_heads * self.config.head_dim,
            False,
            tensors,
            key + ".v_proj",
        )
        self.o_proj = Ex4bitLinear(
            config,
            self.config.num_attention_heads * self.config.head_dim,
            self.config.hidden_size,
            False,
            tensors,
            key + ".o_proj",
        )

    def forward(self, hidden_states, cache, buffer):
        bsz, q_len, _ = hidden_states.size()
        past_len = cache.current_seq_len

        # Project q, k, v, apply position embeddings to k and v

        query_states = self.q_proj.forward(hidden_states)
        key_states = self.k_proj.forward(hidden_states)

        # cuda_ext.rope_(
        #     query_states,
        #     self.sin,
        #     self.cos,
        #     past_len,
        #     self.config.num_attention_heads,
        #     self.config.head_dim,
        # )
        # cuda_ext.rope_(
        #     key_states,
        #     self.sin,
        #     self.cos,
        #     past_len,
        #     self.config.num_attention_heads,
        #     self.config.head_dim,
        # )

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)
        value_states = (
            self.v_proj.forward(hidden_states)
            .view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim)
            .transpose(1, 2)
        )

        # Add keys and values to cache
        new_keys = cache.key_states[self.index].narrow(2, past_len, q_len)
        new_values = cache.value_states[self.index].narrow(2, past_len, q_len)
        new_keys.copy_(key_states)
        new_values.copy_(value_states)

        # Key/value tensors with past

        key_states = cache.key_states[self.index].narrow(2, 0, past_len + q_len)
        value_states = cache.value_states[self.index].narrow(2, 0, past_len + q_len)

        # Attention

        # -- Scaled dot-product attention from PyTorch 2, should be comparable to xformers (?)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=buffer.attn_mask,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2)

        # Output projection

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj.forward(attn_output)

        return attn_output


# class ExLlamaDecoderLayer:
class ExLlamaDecoderLayer(nn.Module):
    def __init__(self, config, tensors, key, index, sin, cos):
        super().__init__()

        self.config = config
        self.index = index

        self.self_attn = ExLlamaAttention(
            self.config,
            tensors,
            key + ".self_attn",
            sin,
            cos,
            self.index,
        )
        self.mlp = ExLlamaMLP(self.config, tensors, key + ".mlp")

        self.input_layernorm = ExLlamaRMSNorm(
            self.config, tensors, key + ".input_layernorm.weight"
        )
        self.post_attention_layernorm = ExLlamaRMSNorm(
            self.config, tensors, key + ".post_attention_layernorm.weight"
        )

    def forward(self, hidden_states, cache, buffer):
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states, buffer)
        hidden_states = self.self_attn.forward(hidden_states, cache, buffer)
        hidden_states = residual + hidden_states

        # hidden_states += cuda_ext.mlp_q4v2(
        #     hidden_states,
        #     self.post_attention_layernorm.weight,
        #     self.config.rms_norm_eps,
        #     self.mlp.gate_proj.quant_args(),
        #     self.mlp.up_proj.quant_args(),
        #     self.mlp.down_proj.quant_args(),
        #     self.config.intermediate_size,
        # )

        return hidden_states


# Persistent cache for inference. Allocate the whole thing up front.


class ExLlamaCache:
    def __init__(self, model, batch_size=1, max_seq_len=-1, copy_from=None):
        self.model = model
        self.config = self.model.config
        self.max_seq_len = max_seq_len if max_seq_len != -1 else self.config.max_seq_len
        self.batch_size = batch_size

        self.key_states = []
        self.value_states = []
        self.current_seq_len = 0

        # Preallocate full-length cache

        for i in range(self.config.num_hidden_layers):
            if copy_from is None:
                p_key_states = torch.zeros(
                    self.batch_size,
                    self.config.num_attention_heads,
                    self.max_seq_len,
                    self.config.head_dim,
                    dtype=torch.float16,
                    device=self.model.config.device_map.layers[i],
                )
                p_value_states = torch.zeros(
                    self.batch_size,
                    self.config.num_attention_heads,
                    self.max_seq_len,
                    self.config.head_dim,
                    dtype=torch.float16,
                    device=self.model.config.device_map.layers[i],
                )

            else:
                p_key_states = copy_from.key_states[i].clone()
                p_value_states = copy_from.value_states[i].clone()

            self.key_states.append(p_key_states)
            self.value_states.append(p_value_states)

    def clone(self):
        new = ExLlamaCache(
            self.model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            copy_from=self,
        )
        return new

    def roll_left(self):
        for i in range(self.config.num_hidden_layers):
            self.key_states[i] = torch.roll(self.key_states[i], shifts=-1, dims=2)
            self.value_states[i] = torch.roll(self.value_states[i], shifts=-1, dims=2)

        self.current_seq_len -= 1

    def copy_states(
        self,
        target,
        from_column,
        from_columns,
        to_column,
        to_columns,
        from_row,
        from_rows,
        to_row,
        to_rows,
    ):
        assert from_rows == 1
        assert from_columns == to_columns
        assert to_column + to_columns <= target.max_seq_len
        assert from_column + from_columns <= self.max_seq_len

        for i in range(self.config.num_hidden_layers):
            source_view_k = (
                self.key_states[i]
                .narrow(0, from_row, from_rows)
                .narrow(2, from_column, from_columns)
            )
            source_view_v = (
                self.value_states[i]
                .narrow(0, from_row, from_rows)
                .narrow(2, from_column, from_columns)
            )
            target_view_k = (
                target.key_states[i]
                .narrow(0, to_row, to_rows)
                .narrow(2, to_column, to_columns)
            )
            target_view_v = (
                target.value_states[i]
                .narrow(0, to_row, to_rows)
                .narrow(2, to_column, to_columns)
            )

            if to_rows > 1:
                source_view_k = source_view_k.expand_as(target_view_k)
                source_view_v = source_view_v.expand_as(target_view_v)

            target_view_k.copy_(source_view_k)
            target_view_v.copy_(source_view_v)


# Device map for the model.
class ExLlamaDeviceMap:
    def __init__(self, num_layers):
        self.num_layers = num_layers

        self.embed_tokens = "cpu"  # Embedding table on CPU saves 400 MB on the 30B model with no measurable impact on performance
        self.lm_head = "cuda:0"
        self.norm = "cuda:0"
        self.layers = ["cuda:0"] * self.num_layers
        self.stream_layer_interval = 0

    def get_layers_devs(self):
        return sorted(list(set(self.layers)))

    def map(self, key, loading=False):
        if key.startswith("lm_head."):
            return self.lm_head
        if key.startswith("model.embed_tokens."):
            return self.embed_tokens
        if key.startswith("model.norm."):
            return self.norm

        if key.startswith("model.layers."):
            num = int(key.split(".")[2])
            if (
                loading
                and self.stream_layer_interval > 0
                and (num + 1) % self.stream_layer_interval == 0
            ):
                if key.startswith(f"model.layers.{num}.mlp."):
                    return "cpu"
                if key.startswith(f"model.layers.{num}.self_attn."):
                    return "cpu"
            return self.layers[num]

        raise ValueError("Unknown key: " + key)


class ExLlamaBuffer:
    config: ExLlamaConfig

    def __init__(self, config):
        self.config = config

    # Attention mask

    attn_mask: torch.Tensor | None = None

    # Move to device

    def to(self, device):
        new = ExLlamaBuffer(self.config)
        new.attn_mask = _move_tensor(self.attn_mask, device, "attn_mask", self.config)
        return new


def _skip_key(key):
    if key.endswith("_proj.bias"):
        return True
    if key.endswith(".rotary_emb.inv_freq"):
        return True
    return False


def _move_tensor(tensor, new_device, name, config):
    device = str(tensor.device)
    if device == new_device:
        return tensor
    return tensor.to(new_device)


# class ExLlama:
class ExLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eval()

        self.config = config
        self.stream_buffer = None

        # Forward streaming config to device map so we only load the first layer on GPU

        self.config.device_map.stream_layer_interval = self.config.stream_layer_interval

        # Load model weights
        tensors = {}
        with safe_open(self.config.model_path, framework="pt", device="cpu") as f:
            # Begin auto mapping if enabled

            decoder_size = 0
            decoder_dq_size = 0
            norm_size = 0
            head_size = 0
            half_element_size = torch.tensor([], dtype=torch.float16).element_size()

            if self.config.auto_map is not None:
                self.config.device_map.embed_tokens = "cpu"
                self.config.device_map.layers = ["cuda:0"] + ["?"] * (
                    self.config.num_hidden_layers - 1
                )

                for key in f.keys():
                    if _skip_key(key):
                        continue

                    if key.startswith("model.layers.0."):
                        tensor = f.get_tensor(key)
                        decoder_size += tensor.numel() * tensor.element_size()
                        if key.endswith(".weight"):
                            decoder_dq_size += tensor.numel() * tensor.element_size()
                        if key.endswith(".qweight"):
                            decoder_dq_size += tensor.numel() * 8 * half_element_size

                    if key.startswith("model.norm."):
                        tensor = f.get_tensor(key)
                        norm_size += tensor.numel() * tensor.element_size()

                    if key.startswith("lm_head."):
                        tensor = f.get_tensor(key)
                        head_size += tensor.numel() * tensor.element_size()

                # Assign layers automatically

                device_usage = 0
                device_index = 0
                layer_index_device = 0
                max_usage = self.config.auto_map[device_index] * (1024**3)

                for layer in range(self.config.num_hidden_layers + 2):
                    this_layer_size = decoder_size
                    if layer == self.config.num_hidden_layers + 0:
                        this_layer_size = norm_size
                    elif layer == self.config.num_hidden_layers + 1:
                        this_layer_size = head_size
                    while device_usage + this_layer_size > max_usage:
                        device_index += 1
                        device_usage = 0
                        layer_index_device = 0
                        max_usage = self.config.auto_map[device_index] * (1024**3)
                        if device_index >= len(self.config.auto_map):
                            raise ValueError(
                                "Model too large for device allocation scheme."
                            )

                    target = f"cuda:{device_index}"
                    if layer == self.config.num_hidden_layers + 0:
                        self.config.device_map.norm = target
                    elif layer == self.config.num_hidden_layers + 1:
                        self.config.device_map.lm_head = target
                    else:
                        self.config.device_map.layers[layer] = f"cuda:{device_index}"

                    device_usage += this_layer_size
                    layer_index_device += 1

            # Load tensors, move to device(s)
            for key in f.keys():
                if _skip_key(key):
                    continue

                device = self.config.device_map.map(key, loading=True)
                tensor = f.get_tensor(key)

                if key.endswith(".scales"):
                    tensor = tensor.half()
                if key == "lm_head.weight":
                    tensor = tensor.float() if device == "cpu" else tensor.half()
                if key == "model.norm.weight":
                    tensor = tensor.half()
                if key.endswith(".embed_tokens.weight"):
                    tensor = tensor.half()
                if key.endswith(".input_layernorm.weight"):
                    tensor = tensor.half()
                if key.endswith(".post_attention_layernorm.weight"):
                    tensor = tensor.half()

                tensor = tensor.to(device, non_blocking=True)

                tensors[key] = tensor
                # print(key + " -> " + device)

        # Head

        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False, device="meta"
        )
        self.lm_head.weight = nn.Parameter(tensors["lm_head.weight"])
        # self.lm_head_data = tensors["lm_head.weight"].transpose(0, 1).contiguous()

        # Token embeddings

        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.pad_token_id,
            device="meta",
        )
        self.embed_tokens.weight = nn.Parameter(tensors["model.embed_tokens.weight"])

        # Norm

        self.norm = ExLlamaRMSNorm(self.config, tensors, "model.norm.weight")

        # Prepare position embeddings for max seq length

        devs = self.config.device_map.get_layers_devs()

        self.sincos = {}
        for device in devs:
            inv_freq = 1.0 / (
                self.config.rotary_embedding_base
                ** (
                    torch.arange(0, self.config.head_dim, 2, device=device).float()
                    / self.config.head_dim
                )
            )
            t = torch.arange(
                self.config.max_seq_len, device=device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            sin = emb.sin()[None, None, :, :].half()
            cos = emb.cos()[None, None, :, :].half()

            self.sincos[device] = (sin, cos)

        # Layers

        modules = []

        for i in range(self.config.num_hidden_layers):
            device = self.config.device_map.layers[i]
            sin, cos = self.sincos[device]

            layer = ExLlamaDecoderLayer(
                self.config, tensors, f"model.layers.{i}", i, sin, cos
            )

            modules.append(layer)

        # self.layers = modules
        self.layers = nn.ModuleList(modules)

    def forward(self, input_ids, cache, last_id_only=True, preprocess_only=False):
        batch_size, seq_len = input_ids.shape
        past_len = cache.current_seq_len

        buffer = ExLlamaBuffer(self.config)

        # Build attention mask on first device, copy to others if necessary

        devs = self.config.device_map.get_layers_devs()

        if seq_len > 1:
            attn_mask = torch.zeros(
                batch_size,
                1,
                seq_len,
                past_len + seq_len,
                dtype=torch.float16,
                device=devs[0],
            )
            attn_mask_triu = torch.triu(
                torch.full((seq_len - 1, seq_len - 1), torch.finfo(torch.float16).min)
            )
            attn_mask[
                :, :, : seq_len - 1, past_len + 1 : past_len + seq_len
            ] = attn_mask_triu

        else:
            attn_mask = torch.zeros(
                batch_size,
                1,
                seq_len,
                seq_len + past_len,
                dtype=torch.float16,
                device=devs[0],
            )

        buffer.attn_mask = attn_mask

        # Embeddings
        # TODO: Allow passing input embeddings instead of IDs
        input_ids = _move_tensor(input_ids, "cpu", "input_ids", self.config)
        hidden_states = self.embed_tokens(input_ids)

        # Split buffers to devices
        buffers = {devs[0]: buffer}
        for device in devs[1:]:
            buffers[device] = buffer.to(device)

        # Decoder layers
        for i, decoder_layer in enumerate(self.layers):
            device = self.config.device_map.layers[i]
            hidden_states = _move_tensor(
                hidden_states, device, "hidden_states", self.config
            )
            hidden_states = decoder_layer.forward(hidden_states, cache, buffers[device])

        cache.current_seq_len += seq_len

        # Early exit when we don't need logits
        if preprocess_only:
            return None

        # Norm
        hidden_states = _move_tensor(
            hidden_states, self.config.device_map.norm, "hidden_states", self.config
        )

        hidden_states = self.norm.forward(hidden_states, buffer)

        # Head
        if last_id_only:
            hidden_states = hidden_states[:, -1:, :].contiguous()
        if self.config.device_map.lm_head == "cpu":
            hidden_states = hidden_states.float()

        hidden_states = _move_tensor(
            hidden_states, self.config.device_map.lm_head, "hidden_states", self.config
        )

        logits = self.lm_head(hidden_states)
        # logits = cuda_ext.matmul_half(hidden_states, self.lm_head_data, cublas = False)

        logits = logits.float()
        logits = _move_tensor(
            logits, self.config.device_map.embed_tokens, "logits", self.config
        )
        return logits
