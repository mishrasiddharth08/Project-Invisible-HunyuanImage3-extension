"""Synthetic CPU contracts only: no weights, downloads, Comfy imports or GPU."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hunyuan_quant_test", ROOT / "pi_hunyuan/quantization.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "misleading_bf16.safetensors"

    def checkpoint(self, fmt=None, dtype="BF16", convrot=False, remove=None, conf_extra=None):
        # Small matrices exercise real header/payload handling. Family checked separately.
        tensors = {"dense.weight": (dtype, [2, 2], bytes(4 * q._DTYPE_BYTES[dtype]))}
        if fmt:
            conf = {"format": fmt, "convrot": convrot, **(conf_extra or {})}
            raw = json.dumps(conf).encode()
            tensors["dense.comfy_quant"] = ("U8", [len(raw)], raw)
            for suffix in q.FORMATS.get(fmt, (None, None, ()))[2]:
                tensors["dense." + suffix] = ("F32", [1], bytes(4))
        if remove:
            del tensors[remove]
        header, payload = {}, bytearray()
        for name, (dt, shape, raw) in tensors.items():
            first = len(payload)
            payload.extend(raw)
            header[name] = dict(dtype=dt, shape=shape, data_offsets=[first, len(payload)])
        header["__metadata__"] = {"format": "pt"}
        raw = json.dumps(header).encode()
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        return self.path

    def describe(self, *args, **kwargs):
        path = self.checkpoint(*args, **kwargs)
        with patch.object(q, "_family"):
            return q.describe_checkpoint(path)

    def test_all_native_formats_and_convrot(self):
        for fmt in q.FORMATS:
            with self.subTest(fmt=fmt):
                dtype = "I8" if q.FORMATS[fmt][1] == "int8" else "U8"
                result = self.describe(fmt, dtype)
                self.assertEqual(result["formats"], [fmt])
                self.assertEqual(result["format"], fmt)
                self.assertEqual(result["quant_config"], {"mixed_ops": True})
        self.assertEqual(self.describe("int8_tensorwise", "I8", True)["quantization"], ["INT8 ConvRot"])
        self.assertEqual(self.describe()["quantization"], ["BF16"])
        self.assertEqual(self.describe(dtype="F16")["quantization"], ["FP16"])

    def test_filename_never_decides_format(self):
        self.assertEqual(self.describe("asym_w4a8_int8", "I8")["quantization"], ["W4A8"])

    def test_inspection_never_reads_weight_payload(self):
        self.checkpoint("int8_tensorwise", "I8")
        raw = self.path.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8+n])
        first, last = header["dense.weight"]["data_offsets"]
        first, last = first + 8 + n, last + 8 + n
        actual_open = Path.open
        reads = []

        class RecordingStream:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def seek(self, offset):
                return self.stream.seek(offset)
            def read(self, count=-1):
                pos = self.stream.tell()
                value = self.stream.read(count)
                reads.append((pos, pos + len(value)))
                return value

        with patch.object(Path, "open", lambda path, *a, **kw: RecordingStream(actual_open(path, *a, **kw))), \
                patch.object(q, "_family"):
            q.describe_checkpoint(self.path)
        self.assertTrue(all(end <= first or start >= last for start, end in reads))

    def test_wrong_dtype_unknown_format_missing_scales_and_unmarked(self):
        for args, kwargs in [(("nvfp4", "I8"), {}), (("bogus", "I8"), {}),
                             (("int8_tensorwise", "I8"), {"remove": "dense.weight_scale"}),
                             ((None, "I8"), {}),
                             (("int8_tensorwise", "I8"), {"conf_extra": {"params": []}})]:
            with self.subTest(args=args), self.assertRaises(q.QuantizationError):
                self.describe(*args, **kwargs)

    def test_corrupt_and_oversized_headers(self):
        for raw in (b"", struct.pack("<Q", 1 << 40), struct.pack("<Q", 4) + b"null"):
            self.path.write_bytes(raw)
            with self.assertRaises(q.QuantizationError):
                q.describe_checkpoint(self.path)
        with self.assertRaisesRegex(q.QuantizationError, "Duplicate"):
            q._json(b'{"a":1,"a":2}')

    def test_payload_size_and_overlap(self):
        self.checkpoint("int8_tensorwise", "I8")
        raw = self.path.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8+n])
        header["dense.weight"]["shape"] = [999]
        altered = json.dumps(header).encode()
        self.path.write_bytes(struct.pack("<Q", len(altered)) + altered + raw[8+n:])
        with self.assertRaisesRegex(q.QuantizationError, "payload size"):
            q.describe_checkpoint(self.path)

    @staticmethod
    def native_keys():
        return {"model.wte.weight": {"shape": [128000, 4096]},
                "model.layers.0.input_layernorm.weight": {"shape": [4096]},
                "model.layers.0.self_attn.qkv_proj.weight": {"shape": [6144, 4096]},
                "model.layers.0.mlp.experts_gate_up_proj.weight": {"shape": [64, 6144, 4096]},
                "model.layers.0.mlp.experts_down_proj.weight": {"shape": [64, 4096, 3072]},
                "patch_embed.model.0.weight": {"shape": [1]},
                "final_layer.model.0.weight": {"shape": [1]},
                "vision_aligner.layers.0.weight": {"shape": [1]}}

    def test_family_requires_native_image_and_moe(self):
        keys = self.native_keys()
        q._family(keys)
        for foreign in ({"transformer.blocks.0.weight": {"shape": [1]}},
                        {**keys, "diffusion_model.flux.weight": {"shape": [1]}}):
            with self.assertRaises(q.QuantizationError):
                q._family(foreign)
        del keys["final_layer.model.0.weight"]
        with self.assertRaises(q.QuantizationError):
            q._family(keys)

    def test_hardware_and_kernel_gates(self):
        report = self.describe("asym_w4a8_int8", "I8")
        ops = SimpleNamespace(QUANT_ALGOS={"asym_w4a8_int8": {}})
        mm = SimpleNamespace(supports_int8_compute=Mock(return_value=False))
        with patch.object(q, "describe_checkpoint", return_value=report):
            with self.assertRaisesRegex(q.QuantizationError, "hardware mps"):
                q.validate_checkpoint(self.path, "mps", ops=ops, model_management=mm)
            mm.supports_int8_compute.return_value = True
            self.assertTrue(q.validate_checkpoint(self.path, "cpu", ops=ops, model_management=mm)["supported"])
            ops.QUANT_ALGOS.clear()
            with self.assertRaisesRegex(q.QuantizationError, "registration"):
                q.validate_checkpoint(self.path, "cpu", ops=ops, model_management=mm)

    def test_each_format_checks_its_hardware_predicate(self):
        for fmt, (_, capability, _) in q.FORMATS.items():
            with self.subTest(fmt=fmt):
                report = self.describe(fmt, "I8" if capability == "int8" else "U8")
                predicate = Mock(return_value=False)
                mm = SimpleNamespace(**{f"supports_{capability}_compute": predicate})
                with patch.object(q, "describe_checkpoint", return_value=report):
                    with self.assertRaisesRegex(q.QuantizationError, "incompatible"):
                        q.validate_checkpoint(self.path, "test-device", ops=SimpleNamespace(QUANT_ALGOS={fmt: {}}),
                                              model_management=mm)
                predicate.assert_called_once_with("test-device")

    def test_whole_bank_fallback_rejected(self):
        report = {"supported": True, "formats": ["asym_w4a8_int8"],
                  "layers": {"model.layers.0.mlp.experts_gate_up_proj":
                             {"full_precision_matrix_mult": True}}}
        with patch.object(q, "describe_checkpoint", return_value=report):
            with self.assertRaisesRegex(q.QuantizationError, "whole-bank"):
                q.validate_checkpoint(self.path, "cpu", ops=SimpleNamespace(), model_management=SimpleNamespace())

    def test_gguf_magic_and_honest_rejection_without_dependency(self):
        self.path.write_bytes(b"GGUF" + bytes(20))
        with patch.object(q, "_gguf_module", side_effect=q.QuantizationError("optional dependency absent")):
            self.assertFalse(q.describe_checkpoint(self.path)["supported"])
            original = Mock()
            with self.assertRaisesRegex(q.QuantizationError, "dependency absent"):
                q.load_gguf(self.path, original)
            original.assert_not_called()
        wrong = self.path.with_suffix(".gguf")
        wrong.write_bytes(b"nope")
        with self.assertRaisesRegex(q.QuantizationError, "magic"):
            q.describe_checkpoint(wrong)

    def test_gguf_frontend_discovery_is_magic_only(self):
        self.path.write_bytes(b"GGUF" + bytes(20))
        with patch.object(q, "_gguf_module") as dependency:
            report = q.describe_checkpoint(self.path, inspect_gguf=False)
        dependency.assert_not_called()
        self.assertTrue(report["magic_verified"])
        self.assertFalse(report["supported"])
        self.assertFalse(report["experimental_loader"])

    def test_experimental_gguf_requires_opt_in_and_valid_contract(self):
        report = {"container": "gguf", "supported": False, "family": "hunyuan_image_3", "reason": q.GGUF_REASON}
        with patch.object(q, "describe_checkpoint", return_value=report), \
                patch.object(q, "_preflight_gguf", return_value={"native_contract_validated": True}) as preflight:
            with self.assertRaisesRegex(q.QuantizationError, "allow_experimental_gguf=True"):
                q.validate_checkpoint(self.path)
            preflight.assert_not_called()
            accepted = q.validate_checkpoint(self.path, "cpu", allow_experimental_gguf=True)
            self.assertTrue(accepted["supported"])
            self.assertTrue(accepted["experimental_loader"])
            with self.assertRaisesRegex(q.QuantizationError, "unsupported"):
                q.validate_checkpoint(self.path, "mps", allow_experimental_gguf=True)
            preflight.side_effect = q.QuantizationError("native shape mismatch")
            with self.assertRaisesRegex(q.QuantizationError, "shape mismatch"):
                q.validate_checkpoint(self.path, allow_experimental_gguf=True)
            report["family"] = None
            report["mapping_error"] = "foreign model family"
            with self.assertRaisesRegex(q.QuantizationError, "foreign model"):
                q.validate_checkpoint(self.path, allow_experimental_gguf=True)

    def test_gguf_selected_expert_and_bounded_cache(self):
        import numpy as np
        data = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        tensor = SimpleNamespace(name="bank", shape=[4, 2, 3], n_elements=24,
                                 data=data, tensor_type="F32")
        reader = SimpleNamespace(tensors=[tensor], get_field=lambda _: None)
        gguf = SimpleNamespace(GGUFReader=Mock(return_value=reader))
        decode = Mock(side_effect=lambda values, _: values.copy())
        with patch.object(q, "_gguf_module", return_value=gguf), patch.object(q, "_family"), \
                patch.object(q.importlib, "import_module", return_value=SimpleNamespace(dequantize=decode)):
            accessor = q.GGUFExpertReader(self.path, cache_bytes=32)
            with self.assertRaisesRegex(q.QuantizationError, "expert index"):
                accessor.matrix("bank")
            np.testing.assert_array_equal(accessor.matrix("bank", 0), data[0])
            accessor.matrix("bank", 0)
            self.assertEqual(decode.call_count, 1)
            accessor.matrix("bank", 1)
            self.assertEqual(list(accessor.cache), [("bank", 1)])
            self.assertEqual(accessor.cached_bytes, 32)
            self.assertEqual(decode.call_args.args[0].shape, (2, 4))
            self.assertTrue(np.shares_memory(decode.call_args.args[0], data))
            accessor.max_matrix_bytes = 1
            with self.assertRaisesRegex(q.QuantizationError, "budget"):
                accessor.matrix("bank", 2)

    def test_gguf_shape_metadata_and_storage_mismatch(self):
        tensor = SimpleNamespace(name="bank", shape=[4, 2, 3], n_elements=24)
        reader = SimpleNamespace(get_field=lambda _: SimpleNamespace(contents=lambda: [3, 2, 4]))
        self.assertEqual(q._gguf_shape(reader, tensor), (3, 2, 4))
        tensor.n_elements = 25
        with self.assertRaisesRegex(q.QuantizationError, "original shape"):
            q._gguf_shape(reader, tensor)


@unittest.skipUnless(importlib.util.find_spec("gguf") and importlib.util.find_spec("torch"),
                     "Optional gguf/torch unavailable; mapped GGUF CPU tests require gguf==0.19.0")
class MappedGGUFTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        import gguf
        import gguf.quants
        self.np, self.torch, self.gguf = np, torch, gguf
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "tiny.gguf"
        self.values = {
            "dense.weight": np.arange(64, dtype=np.float32).reshape(2, 32) / 64,
            "dense.bias": np.array([.1, -.2], dtype=np.float32),
            "embed.weight": np.arange(128, dtype=np.float32).reshape(4, 32) / 128,
            "bank.weight": np.arange(192, dtype=np.float32).reshape(3, 2, 32) / 192,
            "conv.weight": np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2),
            "conv.bias": np.array([.25], dtype=np.float32),
            "rms.weight": np.ones(32, dtype=np.float32),
            "group.weight": np.array([1., 2.], dtype=np.float32),
            "group.bias": np.array([.1, .2], dtype=np.float32),
            "layer.weight": np.ones(32, dtype=np.float32),
            "layer.bias": np.zeros(32, dtype=np.float32),
        }
        writer = gguf.GGUFWriter(str(self.path), "hunyuan_image_3")
        for name, values in self.values.items():
            kind = gguf.GGMLQuantizationType.Q4_0 if name in ("bank.weight", "embed.weight") else gguf.GGMLQuantizationType.F32
            encoded = gguf.quants.quantize(values, kind)
            writer.add_tensor(name, encoded, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with patch.object(q, "_family"):
            self.store = q.GGUFExpertReader(self.path, cache_bytes=256, max_matrix_bytes=4096)
        self.store.validate_storage()
        self.ops = q.make_gguf_operations(self.store)
        nn = torch.nn
        operations = self.ops

        class Tiny(nn.Module):
            def __init__(self, ops=operations):
                super().__init__()
                self.dense = ops.Linear(32, 2)
                self.embed = ops.Embedding(4, 32)
                self.bank = ops.MoEExperts(3, 32, 2)
                self.conv = ops.Conv2d(1, 1, 2)
                self.rms = ops.RMSNorm(32, eps=1e-6)
                self.group = ops.GroupNorm(1, 2)
                self.layer = ops.LayerNorm(32)
                self.dtype = torch.float32

        self.Tiny = Tiny
        self.model = Tiny()
        self.assertTrue(all(p.is_meta for p in self.model.parameters()))
        q.bind_gguf_model(self.model, self.store, self.ops)

    def tearDown(self):
        import gc
        self.model, self.store, self.ops, self.Tiny = None, None, None, None
        gc.collect()  # release all Windows mmap handles before temporary-file cleanup

    def reference(self, name):
        tensor = self.store.tensors[name]
        return self.torch.tensor(self.gguf.quants.dequantize(tensor.data, tensor.tensor_type))

    def test_meta_bind_and_device_transfer_never_materialize_weights(self):
        self.assertEqual(list(self.model.parameters()), [])
        self.model.to("cpu")
        self.assertEqual(self.model.bank.weight.numel(), 0)
        self.assertEqual(self.store.cached_bytes, 0)
        self.assertFalse(any(name.endswith("weight") for name in self.model.state_dict()))

    def test_dense_conv_and_norms_match_torch(self):
        torch = self.torch
        F = torch.nn.functional
        x = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 32
        torch.testing.assert_close(self.model.dense(x), F.linear(x, self.reference("dense.weight"), self.reference("dense.bias")))
        torch.testing.assert_close(self.model.rms(x), F.rms_norm(x, (32,), self.reference("rms.weight"), 1e-6))
        torch.testing.assert_close(self.model.layer(x), F.layer_norm(x, (32,), self.reference("layer.weight"), self.reference("layer.bias")))
        image = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        torch.testing.assert_close(self.model.conv(image), F.conv2d(image, self.reference("conv.weight"), self.reference("conv.bias")))
        groups = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        torch.testing.assert_close(self.model.group(groups), F.group_norm(groups, 1, self.reference("group.weight"), self.reference("group.bias")))

    def test_quantized_bank_decodes_only_selected_expert_even_in_context(self):
        torch = self.torch
        reference = self.reference("bank.weight")
        x = torch.ones(2, 32)
        decode = self.gguf.quants.dequantize
        with patch.object(self.gguf.quants, "dequantize", wraps=decode) as spy:
            with self.model.bank.bank_resident(x):
                actual = self.model.bank.expert_linear(x, 2)
            torch.testing.assert_close(actual, torch.nn.functional.linear(x, reference[2]))
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(spy.call_args.args[0].shape, (2, 18))  # Q4_0 packed rows, not 3 experts
            self.model.bank.expert_linear(x, 2)
            self.assertEqual(spy.call_count, 1)
            self.model.bank.expert_linear(x, 1)
        self.assertLessEqual(self.store.cached_bytes, 256)
        self.assertEqual(list(self.store.cache), [("bank.weight", 1)])
        with self.assertRaises(q.QuantizationError):
            self.model.bank(x)

    def test_embedding_only_decodes_requested_rows(self):
        torch = self.torch
        ids = torch.tensor([[3, 1, 3]])
        reference = self.reference("embed.weight")
        decode = self.gguf.quants.dequantize
        with patch.object(self.gguf.quants, "dequantize", wraps=decode) as spy:
            actual = self.model.embed(ids)
            self.assertEqual(spy.call_args.args[0].shape[0], 3)
        torch.testing.assert_close(actual, reference[ids])
        self.assertEqual(self.store.cached_bytes, 0)

    def test_binding_rejects_missing_extra_and_wrong_shapes(self):
        tensor = self.store.tensors.pop("dense.bias")
        with self.assertRaisesRegex(q.QuantizationError, "key mismatch"):
            q.bind_gguf_model(self.Tiny(), self.store, self.ops)
        self.store.tensors["dense.bias"] = tensor
        self.store.tensors["foreign.weight"] = tensor
        with self.assertRaisesRegex(q.QuantizationError, "extra"):
            q.bind_gguf_model(self.Tiny(), self.store, self.ops)
        del self.store.tensors["foreign.weight"]
        tiny = self.Tiny()
        tiny.dense = self.ops.Linear(32, 3)
        with self.assertRaisesRegex(q.QuantizationError, "shape mismatch"):
            q.bind_gguf_model(tiny, self.store, self.ops)

    def test_loader_returns_patcher_tokenizer_and_preserves_mmap(self):
        import sys
        torch = self.torch
        config_path = Path(self.temp.name) / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        loader = SimpleNamespace(__package__="test_native.hunyuan_image_3", CONFIG_PATH=config_path,
                                 TOKENIZER_PATH="synthetic_tokenizer", SHIFT=3.0,
                                 MODEL_TYPES={"base": {"cfg_distilled": False, "use_meanflow": False}},
                                 detect_model_type=Mock(return_value="base"))
        operations_model = self.Tiny
        class Config:
            def __init__(self, params, latent, settings, **kwargs):
                self.operations = kwargs["custom_operations"]
        # The full native config is production-only; this tiny wrapper tests loader plumbing.
        class Model(torch.nn.Module):
            def __init__(self, config, device):
                super().__init__()
                self.diffusion_model = operations_model(config.operations)
                self.model_sampling = torch.nn.Module()
                self.model_sampling.register_buffer("sigma", torch.ones(1))
        class Patcher:
            def __init__(self, model, load_device, offload_device):
                self.model = model
                self.model_options = {}
            def add_patches(self, patches):
                return []
            def memory_required(self, shape):
                return 1
            def unpatch_model(self, *args, **kwargs):
                self.model.to("cpu")
        tokenizer = object()
        mm = SimpleNamespace(unet_dtype=lambda **_: torch.float32, get_torch_device=lambda: torch.device("cpu"))
        modules = {"comfy.model_management": mm, "comfy.model_patcher": SimpleNamespace(ModelPatcher=Patcher),
                   "test_native.hunyuan_image_3.model": SimpleNamespace(params_from_config=lambda c: c,
                       HunyuanImage3=lambda params, dtype, device, operations: operations_model(operations)),
                   "test_native.hunyuan_image_3.model_base": SimpleNamespace(HunyuanImage3ModelConfig=Config, HunyuanImage3Model=Model),
                   "test_native.latent_formats": SimpleNamespace(HunyuanImage3=lambda: None),
                   "tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(from_file=Mock(return_value=tokenizer)))}
        with patch.dict(sys.modules, modules), patch.object(q, "_family"):
            with self.assertRaisesRegex(q.QuantizationError, "ambiguous"):
                q.load_gguf(self.path, loader, max_matrix_bytes=4096)
            report = q.validate_checkpoint(self.path, "cpu", allow_experimental_gguf=True,
                                           original_loader=loader, model_type="base")
            self.assertTrue(report["native_contract_validated"])
            self.assertTrue(report["supported"])
            self.assertTrue(report["adapters_supported"])
            self.assertFalse(report["bank_weight_merges_supported"])
            patcher, result = q.load_gguf(self.path, loader, model_type="base", max_matrix_bytes=4096)
            self.assertIs(result, tokenizer)
            self.assertEqual(patcher.memory_required([1]), 16385)
            self.assertTrue(patcher.model.gguf_experimental)
            torch.testing.assert_close(patcher.model.diffusion_model.bank.expert_linear(torch.ones(1, 32), 1),
                                       self.model.bank.expert_linear(torch.ones(1, 32), 1))
            self.assertGreater(patcher.loaded_ram_size(), 0)
            patcher.partially_unload_ram(1)
            self.assertEqual(patcher.loaded_ram_size(), 0)
            with self.assertRaisesRegex(q.QuantizationError, "target is not mapped"):
                patcher.add_patches({"bank.weight": object()})
            patcher.unpatch_model()
        del patcher  # release its retained mapped handle

    def test_dense_lora_injection_clone_eject_and_bank_rejection(self):
        import sys
        from contextlib import nullcontext
        torch = self.torch
        wrapper = torch.nn.Module()
        wrapper.diffusion_model = self.model
        wrapper.gguf_store = self.store
        class Base:
            def __init__(self, model):
                self.model, self.injections = model, {}
            def use_ejected(self):
                return nullcontext()
            def set_injections(self, key, entries):
                self.injections[key] = entries
            def clone(self):
                clone = type(self)(self.model)
                clone.injections = self.injections.copy()
                return clone
            def inject(self):
                for entries in self.injections.values():
                    for entry in entries:
                        entry.inject(self)
            def eject(self):
                for entries in self.injections.values():
                    for entry in entries:
                        entry.eject(self)
        patcher = q._mapped_patcher_class(Base)(wrapper)
        up, down = torch.ones(2, 1), torch.ones(1, 32)
        adapter = SimpleNamespace(name="lora", weights=(up, down, None, None, None, None),
                                  calculate_weight=Mock(), calculate_shape=lambda key: None)
        def calculate(patches, weight, key, **kwargs):
            for strength, value, _, _, _ in patches:
                weight += strength * (value.weights[0] @ value.weights[1]).reshape_as(weight)
            return weight
        modules = {"comfy.lora": SimpleNamespace(calculate_weight=calculate),
                   "comfy.patcher_extension": SimpleNamespace(PatcherInjection=lambda inject, eject: SimpleNamespace(inject=inject, eject=eject))}
        x = torch.ones(1, 32)
        baseline = self.model.dense(x)
        with patch.dict(sys.modules, modules):
            self.assertEqual(patcher.add_patches({"diffusion_model.dense.weight": adapter}, .5), ["diffusion_model.dense.weight"])
            clone = patcher.clone()
            patcher.inject()
            torch.testing.assert_close(self.model.dense(x), baseline + 16)
            with self.assertRaisesRegex(q.QuantizationError, "Concurrent"):
                clone.inject()
            patcher.eject()
            torch.testing.assert_close(self.model.dense(x), baseline)
            clone.add_patches({"diffusion_model.dense.weight": adapter}, .25)
            clone.inject()
            torch.testing.assert_close(self.model.dense(x), baseline + 24)
            clone.eject()
            torch.testing.assert_close(self.model.dense(x), baseline)
            with self.assertRaisesRegex(q.QuantizationError, "bank weight merges"):
                patcher.add_patches({"diffusion_model.bank.weight": adapter})
            bad = SimpleNamespace(name="glora", weights=adapter.weights)
            with self.assertRaisesRegex(q.QuantizationError, "registry adapter"):
                patcher.add_patches({"diffusion_model.dense.weight": bad})
            for name in ("lora", "lokr", "loha", "oft", "boft"):
                adapter.name = name
                adapter.weights = (up, down, None, None, torch.ones(2), None)
                fresh = q._mapped_patcher_class(Base)(wrapper)
                fresh.add_patches({"diffusion_model.dense.weight": adapter}, .5)
                fresh.inject()
                torch.testing.assert_close(self.model.dense(x), baseline + 16)
                fresh.eject()
                torch.testing.assert_close(self.model.dense(x), baseline)
            adapter.calculate_shape = lambda key: (99, 32)
            with self.assertRaisesRegex(q.QuantizationError, "shape/budget"):
                patcher.add_patches({"diffusion_model.dense.weight": adapter})
        self.assertEqual(self.store.cached_bytes, 0)

    def native_module(self):
        """Execute the actual bundled architecture; stub only Comfy/GPU services."""
        import sys
        from types import ModuleType
        modules = {}
        for name in ("comfy", "comfy.ldm", "comfy.ldm.modules", "quant_native_test"):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        for name in ("model_management", "model_patcher", "ops", "quant_ops"):
            item = ModuleType("comfy." + name)
            modules["comfy." + name] = item
            setattr(modules["comfy"], name, item)
        modules["comfy.ldm.modules.attention"] = SimpleNamespace(optimized_attention_masked=Mock())
        modules["quant_native_test.lookahead"] = SimpleNamespace(LayerLookahead=Mock())
        modules["quant_native_test.ops"] = SimpleNamespace(expert_linear_sliced=lambda module, x, i: module.expert_linear(x, i))
        spec = importlib.util.spec_from_file_location("quant_native_test.model", ROOT / "vendor/hunyuan/hunyuan_image_3/model.py")
        module = importlib.util.module_from_spec(spec)
        modules[spec.name] = module
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        return module

    def test_actual_80b_architecture_constructs_entirely_on_meta(self):
        native = self.native_module()
        config = json.loads((ROOT / "vendor/hunyuan/hunyuan_image_3/tencent/config.json").read_text())
        model = native.HunyuanImage3(native.params_from_config(config), dtype=self.torch.bfloat16,
                                    device=self.torch.device("cpu"), operations=self.ops)
        self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 70_000_000_000)
        self.assertEqual(self.store.cached_bytes, 0)

    def test_actual_native_moe_forward_matches_reference_without_bank_decode(self):
        torch, np, gguf = self.torch, self.np, self.gguf
        native = self.native_module()
        config = SimpleNamespace(num_experts=3, moe_topk=2, hidden_size=32,
                                 moe_intermediate_size=32, num_shared_expert=1, mlp_bias=False)
        shapes = {"shared_mlp.gate_and_up_proj.weight": (64, 32),
                  "shared_mlp.down_proj.weight": (32, 32), "gate.wg.weight": (3, 32),
                  "experts_gate_up_proj.weight": (3, 64, 32),
                  "experts_down_proj.weight": (3, 32, 32)}
        path = Path(self.temp.name) / "native_moe.gguf"
        writer = gguf.GGUFWriter(str(path), "hunyuan_image_3")
        rng = np.random.default_rng(7)
        for name, shape in shapes.items():
            kind = gguf.GGMLQuantizationType.Q4_0 if name.startswith("experts_") else gguf.GGMLQuantizationType.F32
            writer.add_tensor(name, gguf.quants.quantize(rng.standard_normal(shape).astype(np.float32) * .05, kind), raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with patch.object(q, "_family"):
            store = q.GGUFExpertReader(path, cache_bytes=16384, max_matrix_bytes=16384)
        ops = q.make_gguf_operations(store)
        model = native.HunyuanImage3MoE(config, dtype=torch.float32, device="cpu", operations=ops)
        q.bind_gguf_model(model, store, ops)
        reference = {name: torch.tensor(gguf.quants.dequantize(t.data, t.tensor_type)) for name, t in store.tensors.items()}
        x = torch.arange(128, dtype=torch.float32).reshape(1, 4, 32) / 128
        F = torch.nn.functional
        def swiglu(value):
            first, second = value.chunk(2, dim=-1)
            return first * F.silu(second)
        flat = x.reshape(-1, 32)
        probabilities = F.linear(flat, reference["gate.wg.weight"]).softmax(-1)
        weights, indices = torch.topk(probabilities, 2, dim=-1)
        weights /= weights.sum(-1, keepdim=True)
        expected = F.linear(swiglu(F.linear(flat, reference["shared_mlp.gate_and_up_proj.weight"])), reference["shared_mlp.down_proj.weight"])
        for token in range(4):
            for route in range(2):
                expert = int(indices[token, route])
                projected = F.linear(flat[token], reference["experts_gate_up_proj.weight"][expert])
                expected[token] += weights[token, route] * F.linear(swiglu(projected), reference["experts_down_proj.weight"][expert])
        with patch.object(gguf.quants, "dequantize", wraps=gguf.quants.dequantize) as spy:
            torch.testing.assert_close(model(x), expected.reshape_as(x))
            self.assertTrue(all(call.args[0].ndim <= 2 for call in spy.call_args_list))
        self.assertLessEqual(store.cached_bytes, 16384)
        with self.assertRaisesRegex(q.QuantizationError, "Whole-bank"):
            store.tensor("experts_gate_up_proj.weight")


if __name__ == "__main__":
    unittest.main()
