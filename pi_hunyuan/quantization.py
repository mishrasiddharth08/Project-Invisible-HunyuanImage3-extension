"""Bounded checkpoint inspection and strict native HunyuanImage3 preflight.

No torch/Comfy/GGUF imports at module import time, no downloads, no global patches.
describe_checkpoint reads safetensors headers and small comfy_quant payloads only.
validate_checkpoint must run before the native loader (integration belongs to caller).
load_gguf is an experimental native-name loader with lazy dense operations,
row-gathered embeddings and selected-expert banks. No real-model validation.
It uses the standard ModelPatcher, bypassing aimdo weight/prefetch ownership.
Audited sources: bundled hunyuan loader.py/ops.py, comfy ops.py/quant_ops.py;
city96/ComfyUI-GGUF loader.py/ops.py; ggml-org/llama.cpp gguf_reader.py.
"""
import importlib
import json
import math
import struct
import uuid
from contextlib import contextmanager
from collections import Counter, OrderedDict
from pathlib import Path


class QuantizationError(ValueError):
    """Unsupported format, family, layout or execution hardware."""


FORMATS = {
    "asym_w4a8_int8": ("W4A8", "int8", ("weight_s_rel",)),
    "w6a8_int8": ("W6A8", "int8", ("weight_s_rel",)),
    "int8_tensorwise": ("INT8", "int8", ("weight_scale",)),
    "convrot_w4a4": ("ConvRot W4A4", "int8", ("weight_scale",)),
    "float8_e4m3fn": ("FP8 E4M3", "fp8", ()),
    "float8_e5m2": ("FP8 E5M2", "fp8", ()),
    "mxfp8": ("MXFP8", "mxfp8", ("weight_scale",)),
    "nvfp4": ("NVFP4", "nvfp4", ("weight_scale", "weight_scale_2")),
}
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1,
                "F8_E4M3": 1, "F8_E4M3FN": 1, "F8_E5M2": 1, "F8_E8M0": 1}
GGUF_REASON = (
    "GGUF requires the experimental load_gguf route (real generation unvalidated). "
    "Required: exact native model.wte/self_attn.qkv_proj keys and image projections; "
    "MoE experts_gate_up_proj [64,6144,4096] and experts_down_proj [64,4096,3072] "
    "with verified expert order and GGML block alignment; custom Linear, Embedding, "
    "Conv2d, RMSNorm and MoEExperts operations; selected-expert-only dequantization "
    "and bounded caching with the standard ModelPatcher; aimdo and bank weight merges are excluded. "
    "Generic llama/Flux/HunyuanVideo GGUF mappings are incompatible. "
    "Whole-model or whole-bank BF16 conversion is intentionally prohibited."
)


def _json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise QuantizationError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QuantizationError(f"Invalid checkpoint JSON: {exc}") from exc


def _family(tensors):
    required = ("model.wte.weight", "model.layers.0.input_layernorm.weight",
                "model.layers.0.self_attn.qkv_proj.weight",
                "model.layers.0.mlp.experts_gate_up_proj.weight",
                "model.layers.0.mlp.experts_down_proj.weight")
    missing = [key for key in required if key not in tensors]
    if missing or not all(any(k.startswith(p) for k in tensors)
                          for p in ("patch_embed.", "final_layer.", "vision_aligner.")):
        raise QuantizationError("Not a native HunyuanImage3 image checkpoint; missing native "
                                f"image/MoE signatures: {missing[:3]}")
    if list(tensors[required[1]]["shape"]) != [4096]:
        raise QuantizationError("HunyuanImage3 hidden dimension must be 4096")
    roots = ("model.layers.", "model.wte.", "model.ln_f.", "lm_head.",
             "patch_embed.", "final_layer.", "vision_aligner.", "time_embed.",
             "time_embed_2.", "timestep_emb.", "guidance_emb.", "timestep_r_emb.")
    foreign = [k for k in tensors if not k.startswith(roots)]
    if foreign:
        raise QuantizationError(f"Foreign/unmapped tensor keys: {foreign[:3]}")
    for key, value in tensors.items():
        if key.endswith(("experts_gate_up_proj.weight", "experts_down_proj.weight")):
            shape = value["shape"]
            rows = 6144 if "experts_gate_up_proj" in key else 4096
            if len(shape) != 3 or list(shape[:2]) != [64, rows]:
                raise QuantizationError(f"Invalid native expert-bank shape: {key}: {shape}")


def _safetensors(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise QuantizationError("Truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(64 << 20, size - 8):
            raise QuantizationError("Invalid/oversized safetensors header")
        header = _json(stream.read(length))
        if not isinstance(header, dict):
            raise QuantizationError("Safetensors header must be an object")
        metadata = header.pop("__metadata__", {})
        base = 8 + length
        spans = []
        for key, tensor in header.items():
            if not isinstance(tensor, dict):
                raise QuantizationError(f"Invalid descriptor: {key}")
            shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
            if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= size - base):
                raise QuantizationError(f"Invalid shape/offsets: {key}")
            if offsets[1] > offsets[0]:
                spans.append((*offsets, key))
            width = _DTYPE_BYTES.get(tensor.get("dtype"))
            if width is None or math.prod(shape) * width != offsets[1] - offsets[0]:
                raise QuantizationError(f"Invalid dtype/payload size: {key}")
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise QuantizationError("Overlapping safetensors payloads")
        _family(header)
        layers = {}
        for key, tensor in header.items():
            if not key.endswith(".comfy_quant"):
                continue
            first, last = tensor["data_offsets"]
            if (tensor.get("dtype") != "U8" or last - first > 65536
                    or tensor["shape"] != [last - first]):
                raise QuantizationError(f"Invalid comfy_quant payload: {key}")
            stream.seek(base + first)
            conf = _json(stream.read(last - first))
            if not isinstance(conf, dict) or conf.get("format") not in FORMATS:
                raise QuantizationError(f"Unknown Comfy quantization: {key}: {conf}")
            prefix = key[:-len("comfy_quant")]
            fmt = conf["format"]
            if not isinstance(conf.get("params", {}), dict):
                raise QuantizationError(f"Invalid quantization params: {key}")
            for suffix in ("weight",) + FORMATS[fmt][2]:
                if prefix + suffix not in header:
                    raise QuantizationError(f"Missing {fmt} tensor: {prefix + suffix}")
            weight = header[prefix + "weight"]
            shape = weight["shape"]
            dtype = weight.get("dtype")
            allowed = ({"I8"} if FORMATS[fmt][1] == "int8" else
                       {"U8"} if fmt == "nvfp4" else
                       {"U8", "F8_E4M3", "F8_E4M3FN"} if fmt in ("mxfp8", "float8_e4m3fn") else
                       {"U8", "F8_E5M2"})
            if dtype not in allowed:
                raise QuantizationError(f"{fmt} has incompatible storage dtype {dtype}: {prefix}")
            if "experts_" in prefix:
                logical_k = 4096 if "experts_gate_up_proj" in prefix else 3072
                bits = {"asym_w4a8_int8": 4, "w6a8_int8": 6,
                        "nvfp4": 4, "convrot_w4a4": 4}.get(fmt, 8)
                if shape[-1] * 8 != logical_k * bits or conf.get("num_experts", 64) != 64:
                    raise QuantizationError(f"{fmt} expert width/count mismatch: {prefix}")
            layers[prefix[:-1]] = conf
        unmarked = [k for k, v in header.items() if k.endswith(".weight")
                    and v.get("dtype") not in ("BF16", "F16", "F32")
                    and k[:-len(".weight")] not in layers]
        if unmarked:
            raise QuantizationError("Quantized weights lack native comfy_quant metadata: "
                                    f"{unmarked[:3]}; convert legacy metadata/scaled_fp8 explicitly")
        for key, value in header.items():
            if key.endswith(("experts_gate_up_proj.weight", "experts_down_proj.weight")):
                if key[:-len(".weight")] not in layers:
                    expected = 4096 if "experts_gate_up_proj" in key else 3072
                    if value["shape"][-1] != expected:
                        raise QuantizationError(f"Invalid plain expert width: {key}")
        formats = sorted({conf["format"] for conf in layers.values()})
        labels = sorted({("INT8 ConvRot" if conf["format"] == "int8_tensorwise"
                          and (conf.get("convrot") or conf.get("params", {}).get("convrot"))
                          else FORMATS[conf["format"]][0]) for conf in layers.values()})
        dtypes = dict(Counter(v.get("dtype") for v in header.values()))
        if not formats:
            labels = ["BF16" if "BF16" in dtypes else "FP16" if "F16" in dtypes else "FP32"]
        format_name = formats[0] if len(formats) == 1 else "mixed" if formats else labels[0].lower()
        return dict(container="safetensors", format=format_name, family="hunyuan_image_3", supported=True,
                    formats=formats, quantization=labels, quant_config={"mixed_ops": True} if layers else None,
                    layers=layers, metadata=metadata, tensor_count=len(header), dtypes=dtypes,
                    validation="structural inspection only; no generation validation")


def _gguf_module():
    try:
        return importlib.import_module("gguf")
    except ImportError as exc:
        raise QuantizationError("Optional gguf package is absent; install it explicitly in the isolated "
                                "worker for experimental GGUF loading. " + GGUF_REASON) from exc


def _gguf_shape(reader, tensor):
    field = reader.get_field("comfy.gguf.orig_shape." + tensor.name)
    shape = tuple(int(n) for n in (field.contents() if field else reversed(tensor.shape)))
    if not shape or any(n <= 0 for n in shape) or math.prod(shape) != tensor.n_elements:
        raise QuantizationError(f"Invalid GGUF original shape: {tensor.name}: {shape}")
    return shape


def describe_checkpoint(path, *, inspect_gguf=True):
    """Return JSON-safe structural information; never materialize model weights.

    Native safetensors raise on wrong families/layouts. GGUF returns supported=False
    even when its native names match. Hardware is checked separately below.
    inspect_gguf=False permits frontend discovery from four magic bytes alone.
    """
    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic == b"GGUF":
        report = dict(container="gguf", format="gguf", dtypes={}, supported=False, family=None,
                      experimental=True, experimental_loader=False, magic_verified=True, reason=GGUF_REASON)
        if not inspect_gguf:
            return report  # frontend discovery: magic only, no optional dependency import
        try:
            gguf = _gguf_module()
        except QuantizationError as exc:
            report["inspection_error"] = str(exc)
            return report
        reader = gguf.GGUFReader(str(path), mode="r")
        tensors = {t.name: dict(shape=list(_gguf_shape(reader, t)),
                               dtype=t.tensor_type.name, bytes=int(t.n_bytes)) for t in reader.tensors}
        field = reader.get_field("general.architecture")
        report.update(architecture=field.contents() if field else None,
                      tensors=tensors, tensor_count=len(tensors),
                      dtypes=dict(Counter(t["dtype"] for t in tensors.values())),
                      formats=sorted({t["dtype"] for t in tensors.values()}))
        try:
            _family(tensors)
            report["family"] = "hunyuan_image_3"
            report["experimental_candidate"] = True
            report["validation"] = "experimental CPU-tested operations; real generation unvalidated"
        except QuantizationError as exc:
            report["mapping_error"] = str(exc)
        return report
    if path.suffix.lower() == ".gguf":
        raise QuantizationError("File named GGUF has invalid GGUF magic")
    if path.suffix.lower() != ".safetensors":
        raise QuantizationError("Only native safetensors or inspectable GGUF containers are accepted")
    return _safetensors(path)


def validate_checkpoint(path, device=None, *, ops=None, model_management=None,
                        allow_experimental_gguf=False, original_loader=None, model_type=None):
    """Strict native preflight; refuse fallback dequantization on incompatible hardware.

    Caller may inject the worker's comfy.ops and comfy.model_management. This
    validates formats/capability predicates, not VRAM fit or installed kernel execution.
    GGUF needs allow_experimental_gguf=True and a complete native meta/config
    preflight. original_loader accepts the backend's loader module; model_type
    explicitly resolves undistilled checkpoints without an unambiguous filename.
    """
    report = describe_checkpoint(path)
    if report.get("container") == "gguf":
        if not allow_experimental_gguf:
            raise QuantizationError("Experimental GGUF requires allow_experimental_gguf=True. " + report["reason"])
        if report.get("family") != "hunyuan_image_3":
            raise QuantizationError(report.get("mapping_error", report.get("inspection_error", GGUF_REASON)))
        if device is not None and str(device).split(":", 1)[0] not in ("cpu", "cuda"):
            raise QuantizationError(f"Experimental GGUF execution device is unsupported: {device}")
        contract = _preflight_gguf(path, original_loader, model_type)
        return {**report, **contract, "supported": True, "experimental_loader": True,
                "validation": "native config/key/shape/decoder preflight; real generation unvalidated"}
    if not report["supported"]:
        raise QuantizationError(report["reason"])
    if not report["formats"]:
        return report  # QuantlessMoEOps is selected by the native loader.
    ops = ops if ops is not None else importlib.import_module("comfy.ops")
    mm = model_management if model_management is not None else importlib.import_module("comfy.model_management")
    device = mm.get_torch_device() if device is None else device
    for layer, conf in report["layers"].items():
        if "experts_" in layer and conf.get("full_precision_matrix_mult"):
            raise QuantizationError(f"{layer} requests whole-bank full-precision dequantization; refused")
    for fmt in report["formats"]:
        if fmt not in ops.QUANT_ALGOS:
            raise QuantizationError(f"Worker lacks the {fmt} layout/kernel registration")
        capability = FORMATS[fmt][1]
        predicate = getattr(mm, f"supports_{capability}_compute", None)
        if predicate is None or not predicate(device):
            raise QuantizationError(f"{fmt} is incompatible with hardware {device}; native "
                                    f"{capability} compute required; whole-bank fallback refused")
    return report


class GGUFExpertReader:
    """CPU mmap accessor with a byte-bounded expert LRU and chunked decoder.

    Dense matrices dequantize on explicit request; 3D banks ONLY by expert index.
    Cache budget excludes the caller's returned array and decoder temporaries.
    """
    def __init__(self, path, cache_bytes=64 << 20, max_matrix_bytes=256 << 20):
        if cache_bytes < 0 or max_matrix_bytes <= 0:
            raise ValueError("Invalid GGUF memory budget")
        self.gguf = _gguf_module()
        self.reader = self.gguf.GGUFReader(str(path), mode="r")
        self.tensors = {t.name: t for t in self.reader.tensors}
        _family({k: {"shape": _gguf_shape(self.reader, t)} for k, t in self.tensors.items()})
        self.cache_bytes, self.max_matrix_bytes = cache_bytes, max_matrix_bytes
        self.cache, self.cached_bytes = OrderedDict(), 0

    def matrix(self, name, expert=None):
        tensor = self.tensors[name]
        shape = _gguf_shape(self.reader, tensor)
        if len(shape) == 3:
            if type(expert) is not int or not 0 <= expert < shape[0]:
                raise QuantizationError("A bank requires exactly one valid expert index")
            # Orig-shape metadata must not reshape bytes across expert/block boundaries.
            if tuple(reversed(tensor.shape)) != shape or tensor.data.shape[0] != shape[0]:
                raise QuantizationError("GGUF bank storage is not independently expert-aligned")
            data, target = tensor.data[expert], shape[1:]
        elif len(shape) == 2 and expert is None:
            data, target = tensor.data, shape
        else:
            raise QuantizationError("Only dense matrices or selected 3D experts are accessible")
        if math.prod(target) * 4 > self.max_matrix_bytes:
            raise QuantizationError("Requested matrix exceeds the dequantization memory budget")
        key = (name, expert)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        # gguf.quants CPU reference decoder; slice the mapped bytes BEFORE dequantizing.
        result = self._decode(data, tensor, target)
        result.setflags(write=False)
        if expert is not None and result.nbytes <= self.cache_bytes:
            while self.cached_bytes + result.nbytes > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cached_bytes -= old.nbytes
            self.cache[key] = result
            self.cached_bytes += result.nbytes
        return result

    def _decode(self, data, tensor, shape):
        """Limit reference-decoder temporaries to a few rows; never decode a bank."""
        import numpy as np
        if math.prod(shape) * 4 > self.max_matrix_bytes:
            raise QuantizationError(f"{tensor.name}: dequantization output exceeds memory budget")
        quants = importlib.import_module("gguf.quants")
        rows = data.reshape(-1, data.shape[-1])
        logical_width = math.prod(shape) // rows.shape[0]
        step = max(1, (4 << 20) // (logical_width * 4))
        result = np.empty((rows.shape[0], logical_width), dtype=np.float32)
        for start in range(0, len(rows), step):
            stop = min(start + step, len(rows))
            result[start:stop] = quants.dequantize(rows[start:stop], tensor.tensor_type).reshape(stop - start, logical_width)
        return result.reshape(shape)

    def tensor(self, name):
        """Decode a non-bank parameter, bounded by max_matrix_bytes."""
        if name.endswith(("experts_gate_up_proj.weight", "experts_down_proj.weight")):
            raise QuantizationError("Whole-bank dequantization is prohibited")
        tensor = self.tensors[name]
        return self._decode(tensor.data, tensor, _gguf_shape(self.reader, tensor))

    def embedding_rows(self, name, indices):
        """Decode only requested vocabulary rows, in bounded row chunks."""
        np = importlib.import_module("numpy")
        tensor = self.tensors[name]
        shape = _gguf_shape(self.reader, tensor)
        indices = np.asarray(indices)
        if (len(shape) != 2 or tuple(reversed(tensor.shape)) != shape
                or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= shape[0])):
            raise QuantizationError("Invalid GGUF embedding row layout/indices")
        target = (*indices.shape, shape[1])
        if math.prod(target) * 4 > self.max_matrix_bytes:
            raise QuantizationError("Embedding lookup output exceeds memory budget")
        result = np.empty((indices.size, shape[1]), dtype=np.float32)
        flat = indices.reshape(-1)
        step = max(1, (4 << 20) // (shape[1] * 4))
        for start in range(0, len(flat), step):
            stop = min(start + step, len(flat))
            result[start:stop] = self._decode(tensor.data[flat[start:stop]], tensor, (stop - start, shape[1]))
        return result.reshape(target)

    def clear_cache(self):
        self.cache.clear()
        self.cached_bytes = 0

    def validate_storage(self):
        """Reject unknown decoders and cross-expert reshapes before constructing a model."""
        quants = importlib.import_module("gguf.quants")
        field = self.reader.get_field("general.architecture")
        arch = field.contents() if field else None
        if arch not in (None, "hunyuan_image_3", "hunyuan_image_3_moe", "hunyuanimage3"):
            raise QuantizationError(f"Incompatible GGUF architecture: {arch}; native HunyuanImage3 required")
        for tensor in self.tensors.values():
            shape = _gguf_shape(self.reader, tensor)
            kind = tensor.tensor_type.name
            if kind not in ("F32", "F16") and not callable(getattr(getattr(quants, kind, None), "dequantize", None)):
                raise QuantizationError(f"GGUF CPU decoder unavailable: {tensor.name}: {kind}")
            if tensor.name.endswith(("experts_gate_up_proj.weight", "experts_down_proj.weight")):
                if tuple(reversed(tensor.shape)) != shape or tensor.data.shape[0] != shape[0]:
                    raise QuantizationError(f"GGUF expert/block alignment mismatch: {tensor.name}")
                if math.prod(shape[1:]) * 4 > self.max_matrix_bytes:
                    raise QuantizationError(f"One expert exceeds memory budget: {tensor.name}")


def make_gguf_operations(store, *, torch_module=None):
    """Native custom_operations factory; all parameter construction stays on meta.

    bind_gguf_model validates every native key/shape before removing meta Parameters.
    Afterwards only zero-sized dtype markers remain; .to() never moves mapped banks.
    """
    torch = torch_module or importlib.import_module("torch")
    nn, F = torch.nn, torch.nn.functional

    class Mapped:
        def _bind(self, prefix):
            self._gguf_prefix = prefix
            self._gguf_logical_shape = tuple(self.weight.shape) if getattr(self, "weight", None) is not None else None
            for name, parameter in list(self._parameters.items()):
                if parameter is not None:
                    del self._parameters[name]
                    setattr(self, name, torch.empty(0, dtype=parameter.dtype, device="cpu"))
            self._gguf_bound = True

        def _value(self, name, input, expert=None):
            if not getattr(self, "_gguf_bound", False):
                raise QuantizationError("GGUF operation must be bound before forward")
            marker = getattr(self, name, None)
            if marker is None:
                return None
            key = self._gguf_prefix + "." + name
            array = store.matrix(key, expert) if expert is not None else store.tensor(key)
            weight = torch.tensor(array, device=input.device, dtype=input.dtype)
            patches = getattr(self, "_gguf_dense_patches", {}).get(name, ())
            if patches:
                if expert is not None or weight.numel() * 4 > store.max_matrix_bytes:
                    raise QuantizationError("GGUF adapter matrix exceeds its bounded dense scope")
                shape = weight.shape
                weight = importlib.import_module("comfy.lora").calculate_weight(
                    patches, weight.float(), "diffusion_model." + key, intermediate_dtype=torch.float32)
                if weight.shape != shape:
                    raise QuantizationError("GGUF adapter changed the native matrix shape")
                weight = weight.to(input.dtype)
            return weight

        def _apply(self, fn, recurse=True):
            if not getattr(self, "_gguf_bound", False):
                # Constructed meta Parameters must never be materialized by .to().
                return self
            for name in ("weight", "bias"):
                marker = getattr(self, name, None)
                if marker is not None:
                    setattr(self, name, fn(marker))
            return self

        def reset_parameters(self):
            pass

    class Linear(Mapped, nn.Linear):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, device="meta", **kwargs)
        def forward(self, input):
            return F.linear(input, self._value("weight", input), self._value("bias", input))

    class Conv2d(Mapped, nn.Conv2d):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, device="meta", **kwargs)
        def forward(self, input):
            return self._conv_forward(input, self._value("weight", input), self._value("bias", input))

    class Embedding(Mapped, nn.Embedding):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, device="meta", **kwargs)
        def forward(self, input):
            if not getattr(self, "_gguf_bound", False):
                raise QuantizationError("GGUF embedding is not bound")
            if self.max_norm is not None:
                raise QuantizationError("GGUF embedding max_norm mutation is unsupported")
            rows = store.embedding_rows(self._gguf_prefix + ".weight", input.detach().cpu().numpy())
            return torch.tensor(rows, device=input.device, dtype=self.weight.dtype)

    class RMSNorm(Mapped, nn.RMSNorm):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, device="meta", **kwargs)
        def forward(self, input):
            return F.rms_norm(input, self.normalized_shape, self._value("weight", input), self.eps)

    class GroupNorm(Mapped, nn.GroupNorm):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, device="meta", **kwargs)
        def forward(self, input):
            return F.group_norm(input, self.num_groups, self._value("weight", input), self._value("bias", input), self.eps)

    class LayerNorm(Mapped, nn.LayerNorm):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, device="meta", **kwargs)
        def forward(self, input):
            return F.layer_norm(input, self.normalized_shape, self._value("weight", input), self._value("bias", input), self.eps)

    class MoEExperts(Mapped, nn.Module):
        def __init__(self, num_experts, in_features, out_features, bias=False, device=None, dtype=None):
            super().__init__()
            self.num_experts, self.in_features, self.out_features = num_experts, in_features, out_features
            self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features, device="meta", dtype=dtype), requires_grad=False)
            # Published native Hunyuan banks are biasless. Fail rather than decoding a whole bias bank.
            if bias:
                raise QuantizationError("GGUF native expert-bank biases are unsupported")
            self.bias = None
            self._resident_bank = None
        @contextmanager
        def bank_resident(self, input):
            # Even the native full-sequence path streams individual experts.
            yield self
        def expert_linear(self, input, i):
            return F.linear(input, self._value("weight", input, expert=i))
        def forward(self, input):
            raise QuantizationError("An expert index is required; full-bank forward prohibited")

    from types import SimpleNamespace
    return SimpleNamespace(Linear=Linear, Conv2d=Conv2d, Embedding=Embedding,
                           RMSNorm=RMSNorm, GroupNorm=GroupNorm, LayerNorm=LayerNorm,
                           MoEExperts=MoEExperts, LinearExpertBank=MoEExperts, _Mapped=Mapped)


def bind_gguf_model(model, store, operations):
    """Compare the entire meta state against GGUF, then bind without load_state_dict."""
    expected = model.state_dict()
    heads = ("lm_head.", "model.ln_f.")
    actual = {name for name in store.tensors if not name.startswith(heads)}
    missing, extra = set(expected) - actual, actual - set(expected)
    if missing or extra:
        raise QuantizationError(f"GGUF native key mismatch; missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    for name, parameter in expected.items():
        shape = _gguf_shape(store.reader, store.tensors[name])
        if tuple(parameter.shape) != shape:
            raise QuantizationError(f"GGUF shape mismatch: {name}: {shape} != {tuple(parameter.shape)}")
    for name, module in model.named_modules():
        if isinstance(module, operations._Mapped):
            module._bind(name)
        elif list(module.parameters(recurse=False)) or list(module.buffers(recurse=False)):
            raise QuantizationError(f"Unmapped native parameter/buffer module: {name}")
    if any(p.is_meta for p in model.parameters()):
        raise QuantizationError("Unbound meta parameters remain")


class _DenseGGUFInjection:
    """Reversible per-forward dense weight patches; clone-local injection registration."""
    def __init__(self, patches):
        self.patches, self.active = patches, {}

    def inject(self, patcher):
        if id(patcher) in self.active:
            return
        modules = dict(patcher.model.named_modules())
        restored = []
        try:
            for key, patches in self.patches.items():
                module_name, parameter = key.rsplit(".", 1)
                module = modules[module_name]
                if getattr(module, "_gguf_dense_owner", None) is not None:
                    raise QuantizationError("Concurrent GGUF dense adapter injections are unsupported")
                previous = getattr(module, "_gguf_dense_patches", None)
                module._gguf_dense_patches = {**(previous or {}), parameter: patches}
                module._gguf_dense_owner = self
                restored.append((module, previous))
            self.active[id(patcher)] = restored
        except BaseException:
            for module, previous in reversed(restored):
                module._gguf_dense_patches = previous or {}
                module._gguf_dense_owner = None
            raise

    def eject(self, patcher):
        for module, previous in reversed(self.active.pop(id(patcher), [])):
            module._gguf_dense_patches = previous or {}
            module._gguf_dense_owner = None


def _mapped_patcher_class(base):
    class MappedModelPatcher(base):
        def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
            if not math.isfinite(strength_patch) or strength_model != 1.0:
                raise QuantizationError("GGUF dense adapters require finite strength and strength_model=1")
            modules = dict(self.model.named_modules())
            pending = {}
            for key, adapter in patches.items():
                if not isinstance(key, str) or not key.endswith(".weight"):
                    raise QuantizationError("GGUF dense adapters accept exact weight keys only; no offsets")
                module = modules.get(key[:-len(".weight")])
                if module is None or not getattr(module, "_gguf_bound", False):
                    raise QuantizationError(f"GGUF dense adapter target is not mapped: {key}")
                if callable(getattr(module, "expert_linear", None)):
                    raise QuantizationError("GGUF bank weight merges are prohibited; use selected-expert factor injections")
                if (getattr(adapter, "name", None) not in {"lora", "lokr", "loha", "oft", "boft"}
                        or not callable(getattr(adapter, "calculate_weight", None))
                        or not callable(getattr(adapter, "calculate_shape", None))):
                    raise QuantizationError("GGUF dense patches require a Comfy LoRA/LoKr/LoHa/OFT/BOFT registry adapter")
                shape = _gguf_shape(self.model.gguf_store.reader,
                                    self.model.gguf_store.tensors[module._gguf_prefix + ".weight"])
                proposed = adapter.calculate_shape(key)
                if (len(shape) < 2 or proposed is not None and tuple(proposed) != shape
                        or math.prod(shape) * 4 > self.model.gguf_store.max_matrix_bytes):
                    raise QuantizationError(f"GGUF adapter shape/budget mismatch: {key}")
                pending[key] = [(strength_patch, adapter, 1.0, None, None)]
            if not pending:
                return []
            with self.use_ejected():
                combined = {}
                for name, entries in list(self.injections.items()):
                    if name.startswith("pi_hunyuan_gguf_dense_"):
                        for entry in entries:
                            for key, values in entry.inject.__self__.patches.items():
                                combined.setdefault(key, []).extend(values)
                        del self.injections[name]
                for key, values in pending.items():
                    combined.setdefault(key, []).extend(values)
                injection = _DenseGGUFInjection(combined)
                extension = importlib.import_module("comfy.patcher_extension")
                self.set_injections("pi_hunyuan_gguf_dense_" + uuid.uuid4().hex,
                                    [extension.PatcherInjection(injection.inject, injection.eject)])
                self.patches_uuid = uuid.uuid4()
            return list(pending)
        def memory_required(self, input_shape):
            return super().memory_required(input_shape) + self.model.gguf_workspace_bytes
        def model_state_dict_for_saving(self, *args, **kwargs):
            raise QuantizationError("Mapped GGUF checkpoint export is unsupported; retain the original file")
        def add_hook_patches(self, patches, *args, **kwargs):
            if patches:
                raise QuantizationError("Experimental GGUF weight hooks are unsupported")
            return super().add_hook_patches(patches, *args, **kwargs)
        def loaded_ram_size(self):
            return self.model.gguf_store.cached_bytes
        def partially_unload_ram(self, ram_to_unload):
            previous = self.loaded_ram_size()
            self.model.gguf_store.clear_cache()
            return previous
        def unpatch_model(self, *args, **kwargs):
            self.model.gguf_store.clear_cache()
            return super().unpatch_model(*args, **kwargs)
    return MappedModelPatcher


def _gguf_contract(store, path, original_loader=None, model_type=None):
    if original_loader is None:
        loader = importlib.import_module("pi_hunyuan_upstream.hunyuan_image_3.loader")
    elif callable(original_loader):
        loader = importlib.import_module(original_loader.__module__)
    else:
        loader = original_loader
    native = importlib.import_module(loader.__package__ + ".model")
    with open(loader.CONFIG_PATH, encoding="utf-8") as stream:
        config = json.load(stream)
    distilled = "guidance_emb.mlp.0.weight" in store.tensors
    if model_type is None:
        if not distilled and not any(token in Path(path).name.lower() for token in ("instruct", "base")):
            raise QuantizationError("Undistilled GGUF sampling contract is ambiguous; provide model_type='base' or 'instruct'")
        signature = {"guidance_emb.mlp.0.weight": None} if distilled else {}
        model_type = loader.detect_model_type(signature, str(path))
    if model_type not in loader.MODEL_TYPES:
        raise QuantizationError(f"Unknown HunyuanImage3 sampling contract: {model_type}")
    if bool(loader.MODEL_TYPES[model_type]["cfg_distilled"]) != distilled:
        raise QuantizationError("GGUF distilled embedders disagree with the sampling contract")
    config.update(loader.MODEL_TYPES[model_type])
    config["model_type"] = model_type
    params = native.params_from_config(config)
    return loader, native, params, model_type


def _preflight_gguf(path, original_loader=None, model_type=None):
    """Validate complete native meta architecture; no decoder or GPU execution."""
    store = GGUFExpertReader(path)
    store.validate_storage()
    torch = importlib.import_module("torch")
    _, native, params, model_type = _gguf_contract(store, path, original_loader, model_type)
    operations = make_gguf_operations(store, torch_module=torch)
    model = native.HunyuanImage3(params, dtype=torch.float32, device=torch.device("cpu"), operations=operations)
    bind_gguf_model(model, store, operations)
    return {"model_type": model_type, "native_contract_validated": True,
            "adapters_supported": True, "adapter_scope": "bounded dense LoRA/LoKr/LoHa/OFT/BOFT incl DoRA; external expert factors",
            "bank_weight_merges_supported": False, "dynamic_vram_supported": False}


def load_gguf(path, original_loader=None, *, model_type=None, cache_bytes=64 << 20,
              max_matrix_bytes=256 << 20, disable_dynamic=True):
    """Return (standard_model_patcher, tokenizer) for EXPERIMENTAL native GGUF.

    original_loader: bundled loader module, or its load_hunyuan_image_3 function;
    omitted uses the backend's pi_hunyuan_upstream alias. Never invokes or replaces
    that loader. Native tensor names/order only; no llama.cpp aliases or conversion.
    disable_dynamic is accepted for rebuild compatibility; aimdo is always bypassed.
    Dense registry adapters incl DoRA via reversible injections; bank merges, training, real-model
    validation and automatic dependency installation are unsupported.
    """
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise QuantizationError("load_gguf requires a genuine GGUF container")
    store = GGUFExpertReader(path, cache_bytes, max_matrix_bytes)
    store.validate_storage()
    torch = importlib.import_module("torch")
    mm = importlib.import_module("comfy.model_management")
    patchers = importlib.import_module("comfy.model_patcher")
    loader, native, params, model_type = _gguf_contract(store, path, original_loader, model_type)
    model_base = importlib.import_module(loader.__package__ + ".model_base")
    latent = importlib.import_module(loader.__package__.rsplit(".", 1)[0] + ".latent_formats")
    dtype = mm.unet_dtype(supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])
    operations = make_gguf_operations(store, torch_module=torch)
    model_config = model_base.HunyuanImage3ModelConfig(
        params, latent.HunyuanImage3(), {"shift": loader.SHIFT},
        quant_config=None, dtype=dtype, custom_operations=operations)
    # All large operation Parameters are meta; BaseModel's sampling buffers stay CPU.
    model = model_base.HunyuanImage3Model(model_config, device=torch.device("cpu"))
    bind_gguf_model(model.diffusion_model, store, operations)
    model.gguf_store = store  # retain the reader/mmap for the complete patcher lifetime
    model.gguf_experimental = True
    model.gguf_mapped_bytes = sum(int(t.n_bytes) for t in store.tensors.values())
    # Conservative additional CPU/GPU decode workspace, separately from mapped file size.
    model.gguf_workspace_bytes = max_matrix_bytes * 4  # includes temporary float32 LoRA deltas
    patcher = _mapped_patcher_class(patchers.ModelPatcher)(
        model, load_device=mm.get_torch_device(), offload_device=torch.device("cpu"))
    patcher.model_options.setdefault("transformer_options", {})["prefetch_dynamic_vbars"] = False
    tokenizer = importlib.import_module("tokenizers").Tokenizer.from_file(loader.TOKENIZER_PATH)
    model.tokenizer = tokenizer
    def rebuild(checkpoint_path, disable_dynamic=False):
        return load_gguf(checkpoint_path, loader, model_type=model_type,
                         cache_bytes=cache_bytes, max_matrix_bytes=max_matrix_bytes)[0]
    patcher.cached_patcher_init = (rebuild, (str(path),))
    return patcher, tokenizer
