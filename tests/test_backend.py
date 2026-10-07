"""Backend contracts and line protocol without weights, Forge or a GPU."""
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hunyuan_backend_test", ROOT / "pi_hunyuan/backend.py")
backend_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend_module)
HunyuanBackend = backend_module.HunyuanBackend


def node_output(*values):
    return SimpleNamespace(result=values)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.values, axis))

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.object(HunyuanBackend, "_bootstrap"):
            self.backend = HunyuanBackend(root=self.root)
        self.quant_patch = patch('pi_hunyuan.quantization.validate_checkpoint', return_value={'format': 'bf16', 'container': 'safetensors'})
        self.quant_patch.start()
        self.addCleanup(self.quant_patch.stop)
        b = self.backend
        b.torch = SimpleNamespace(inference_mode=nullcontext, from_numpy=Tensor)
        b.np, b.Image, b.ImageOps = np, Image, ImageOps
        b.mm = SimpleNamespace(unload_all_models=Mock(), soft_empty_cache=Mock(),
                               interrupt_current_processing=Mock(), cuda_device_context=lambda _: nullcontext())
        b.folders = SimpleNamespace(folder_names_and_paths={
            category: ([], {".safetensors"}) for category in ("diffusion_models", "vae", "clip_vision")},
            filename_list_cache={"old": "cached"})
        self.models = {}

        def load_model(path):
            model_type = Path(path).stem
            model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(
                config=SimpleNamespace(model_type=model_type))), load_device="cpu")
            self.models[model_type] = model
            return model, object()

        self.vae = SimpleNamespace(decode=Mock(return_value=[Tensor(np.full((32, 48, 3), 0.5))]))
        self.vision = object()
        b.loader = SimpleNamespace(load_hunyuan_image_3=Mock(side_effect=load_model))
        b.clip_vision = SimpleNamespace(load=Mock(return_value=self.vision))
        b.rewrite = SimpleNamespace(_HEADS={}, AUTO="auto", RewriteSettings=lambda **kw: SimpleNamespace(**kw))
        self.positive, self.negative = [[None, {"ids": "tokens"}]], [[None, {"ids": "unconditional"}]]
        encode_result = node_output(self.positive, self.negative, "rewritten text")
        b.nodes = SimpleNamespace(
            _encode=Mock(return_value=encode_result),
            HunyuanImage3ImageEncode=SimpleNamespace(execute=Mock(return_value=encode_result)),
            HunyuanImage3Guidance=SimpleNamespace(execute=Mock(side_effect=lambda cond, scale: node_output(cond))),
            HunyuanImage3EmptyLatent=SimpleNamespace(execute=Mock(return_value=node_output({"samples": object()}))),
            HunyuanImage3VAELoader=SimpleNamespace(execute=Mock(return_value=node_output(self.vae))),
            HunyuanImage3Spectrum=SimpleNamespace(execute=Mock(side_effect=lambda model, enabled, **kw: node_output(model))),
        )

        def sample(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kwargs):
            kwargs["callback"](0, None, None, steps)
            kwargs["callback"](steps - 1, None, None, steps)
            return "vae-space-samples"

        b.sample = SimpleNamespace(prepare_noise=Mock(return_value="noise"), sample=Mock(side_effect=sample))
        self.paths = {name: self.file(f"{name}/{name}.safetensors") for name in ("vae", "vision", "head")}
        self.events = []

    def file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return str(path)

    def request(self, model_type="instruct_distil", **changes):
        request = dict(id="job-7", paths=dict(self.paths, model=self.file(f"models/{model_type}.safetensors")),
                       prompt="a bird", references=[], rewrite="off", spectrum=False, seed=42,
                       width=48, height=32, steps=50, cfg=7.0, guidance=3.0,
                       output=str(self.root / "results/result.png"))
        request.update(changes)
        return request

    def test_all_model_contracts_and_effective_metadata(self):
        for kind, steps, cfg in (("instruct_distil", 8, 1.0), ("instruct", 23, 7.0), ("base", 23, 7.0)):
            with self.subTest(kind=kind):
                result = self.backend.generate(self.request(kind, steps=23), self.events.append)
                call = self.backend.sample.sample.call_args
                self.assertIs(call.args[0], self.models[kind])
                self.assertEqual(call.args[2:6], (steps, cfg, "euler", "simple"))
                self.assertEqual((result["steps"], result["cfg"], result["guidance"]), (steps, cfg, 3.0))
                self.assertEqual(result["model_type"], kind)
                self.assertEqual(result["id"], "job-7")
                self.vae.decode.assert_called_with("vae-space-samples")
                with Image.open(result["output"]) as image:
                    self.assertEqual((image.size, image.mode), ((48, 32), "RGB"))
                self.assertEqual(self.events[-1], dict(event="progress", id="job-7", step=steps, total=steps))
        self.backend.nodes.HunyuanImage3Guidance.execute.assert_called_once_with(self.positive, 3.0)

    def test_same_checkpoint_reuses_model_and_vae_after_unload(self):
        request = self.request()
        self.backend.generate(request, self.events.append)
        retained = self.backend.model
        self.backend.mm.unload_all_models.assert_called()
        self.backend.generate(request, self.events.append)
        self.assertIs(self.backend.model, retained)
        self.backend.loader.load_hunyuan_image_3.assert_called_once()
        self.backend.nodes.HunyuanImage3VAELoader.execute.assert_called_once()

    def test_editing_rewriting_and_spectrum(self):
        refs = []
        for index in range(3):
            path = self.root / f"reference-{index}.png"
            Image.new("RGB", (13, 17), "red").save(path)
            refs.append(str(path))
        request = self.request("instruct", references=refs, rewrite="think + rewrite", spectrum=True)
        result = self.backend.generate(request, self.events.append)
        call = self.backend.nodes.HunyuanImage3ImageEncode.execute.call_args
        self.assertIs(call.args[0], self.backend.model)
        self.assertIs(call.args[1], self.vae)
        self.assertIs(call.args[2], self.vision)
        self.assertEqual(list(call.args[3]), ["image_1", "image_2", "image_3"])
        pixels = call.args[3]["image_1"].numpy()
        self.assertEqual(pixels.shape, (1, 17, 13, 3))
        self.assertEqual(float(pixels.max()), 1.0)
        settings = call.kwargs["prompt_rewriting"]
        self.assertEqual((settings.mode, settings.cot_head, settings.seed), ("think + rewrite", "head.safetensors", 42))
        self.assertEqual(result["rewritten_prompt"], "rewritten text")
        self.backend.nodes._encode.assert_not_called()
        self.backend.nodes.HunyuanImage3Spectrum.execute.assert_called_once()
        self.backend.clip_vision.load.assert_called_once_with(self.paths["vision"])

    def test_exact_head_path_has_priority_over_same_basename(self):
        request = self.request("instruct", rewrite="rewrite only")
        request["paths"]["head"] = self.file("selected/instruct.safetensors")
        self.backend.generate(request, self.events.append)
        folders = self.backend.folders.folder_names_and_paths["diffusion_models"][0]
        token = self.backend.nodes._encode.call_args.args[5].cot_head
        found = next(Path(folder) / token for folder in folders if (Path(folder) / token).is_file())
        self.assertEqual(found, Path(request["paths"]["head"]))
        self.assertFalse(self.backend.folders.filename_list_cache)

    def test_base_edit_rejected_and_base_rewriting_skipped(self):
        ref = self.root / "reference.png"
        Image.new("RGB", (8, 8)).save(ref)
        with self.assertRaisesRegex(ValueError, "text-to-image only"):
            self.backend.generate(self.request("base", references=[str(ref)]), self.events.append)
        self.backend.sample.sample.assert_not_called()
        self.backend.generate(self.request("base", rewrite="rewrite only"), self.events.append)
        self.assertIsNone(self.backend.nodes._encode.call_args.args[5])

    def test_invalid_requests_do_not_load_weights(self):
        request = self.request()
        for changes in ({"width": 47}, {"cfg": float("nan")}, {"guidance": float("inf")},
                        {"references": ["x"] * 4}, {"rewrite": "unknown"}, {"spectrum": "true"},
                        {"seed": -1}, {"steps": 0}, {"output": "bad.jpg"}, {"prompt": 123}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, FileNotFoundError)):
                self.backend.generate(dict(request, **changes), self.events.append)
        self.backend.loader.load_hunyuan_image_3.assert_not_called()

    def test_sampling_error_releases_cache_and_allows_retry(self):
        request = self.request()
        original = self.backend.sample.sample.side_effect
        self.backend.sample.sample.side_effect = RuntimeError("sampling failed")
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            self.backend.generate(request, self.events.append)
        self.assertIsNone(self.backend.model)
        self.backend.mm.soft_empty_cache.assert_called()
        self.backend.sample.sample.side_effect = original
        self.assertEqual(self.backend.generate(request, self.events.append)["event"], "result")

    def test_failed_vae_replacement_does_not_reuse_stale_key(self):
        request = self.request()
        self.backend.generate(request, self.events.append)
        self.backend.nodes.HunyuanImage3VAELoader.execute.side_effect = RuntimeError("bad VAE")
        changed = dict(request, paths=dict(request["paths"], vae=self.file("other/other.safetensors")))
        with self.assertRaisesRegex(RuntimeError, "bad VAE"):
            self.backend.generate(changed, self.events.append)
        self.assertIsNone(self.backend._vae_key)
        self.backend.nodes.HunyuanImage3VAELoader.execute.side_effect = None
        self.assertEqual(self.backend.generate(request, self.events.append)["event"], "result")

    def test_cached_repeat_and_tiled_decode(self):
        request = self.request()
        first = self.backend.generate(request, self.events.append)
        self.vae.decode_tiled = Mock(return_value=[Tensor(np.full((32, 48, 3), 0.5))])
        second = self.backend.generate(dict(request, vae_tiled=True, offload_after=True), self.events.append)
        self.assertFalse(first['model_reused'])
        self.assertTrue(second['model_reused'])
        self.assertTrue(second['vae_tiled'])
        self.backend.loader.load_hunyuan_image_3.assert_called_once()
        self.vae.decode_tiled.assert_called_once()
        self.assertIn('metrics', second)
        self.assertIn('sample', second['stage_seconds'])

    def test_small_profile_enforces_no_prefetch_after_core_rewrites_flag(self):
        forwarding = Mock(return_value='result')
        diffusion = SimpleNamespace(forward=forwarding)
        self.backend.model = SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion), model_options={})
        self.backend._disable_prefetch()
        options = {'prefetch_dynamic_vbars': True, 'other': 1}
        self.assertEqual(diffusion.forward('x', transformer_options=options), 'result')
        self.assertFalse(forwarding.call_args.kwargs['transformer_options']['prefetch_dynamic_vbars'])
        self.assertTrue(options['prefetch_dynamic_vbars'])

    def test_other_gpu_job_fails_before_model_allocation(self):
        self.backend.torch.cuda = SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (6*2**30, 32*2**30))
        self.backend.memory_plan = {'budget_gib': 13}
        with self.assertRaisesRegex(RuntimeError, 'GPU memory is busy'):
            self.backend._check_available_budget()
        self.backend.loader.load_hunyuan_image_3.assert_not_called()

    def test_single_frame_video_vae_saves_rgb_image(self):
        self.vae.decode.return_value = [Tensor(np.full((1, 32, 48, 3), 0.5))]
        result = self.backend.generate(self.request(), self.events.append)
        with Image.open(result["output"]) as image:
            self.assertEqual((image.size, image.mode), ((48, 32), "RGB"))
            self.assertEqual(image.getpixel((0, 0)), (127, 127, 127))

    def test_multiple_vae_frames_are_rejected(self):
        self.vae.decode.return_value = [Tensor(np.full((2, 32, 48, 3), 0.5))]
        request = self.request()
        with self.assertRaisesRegex(RuntimeError, "Unexpected VAE image shape"):
            self.backend.generate(request, self.events.append)
        self.assertFalse(Path(request["output"]).exists())

    def test_nonfinite_pixels_never_save_result(self):
        self.vae.decode.return_value = [Tensor(np.full((32, 48, 3), np.nan))]
        request = self.request()
        with self.assertRaisesRegex(RuntimeError, "non-finite pixels"):
            self.backend.generate(request, self.events.append)
        self.assertFalse(Path(request["output"]).exists())


class BootstrapTests(unittest.TestCase):
    def test_missing_dependency_identifies_worker_install_location(self):
        missing = ModuleNotFoundError("missing comfy_kitchen", name="comfy_kitchen")
        with patch.object(HunyuanBackend, "_bootstrap", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, r"comfy_kitchen.*vendor.*python"):
                HunyuanBackend()

    def test_auto_enables_dynamic_vram_before_imports_and_fails_closed(self):
        for initialized in (True, False):
            with self.subTest(initialized=initialized), tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
                root = Path(temp)
                for file in ("vendor/ComfyUI/comfy/options.py", "vendor/hunyuan/__init__.py"):
                    path = root / file
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")
                args = SimpleNamespace(reserve_vram=None, disable_nvml_pressure=False, vram_headroom=0)
                cli = SimpleNamespace(args=args, enables_dynamic_vram=lambda: True)
                control = SimpleNamespace(init=Mock(), init_devices=Mock(return_value=initialized))
                dynamic = object()
                patchers = SimpleNamespace(CoreModelPatcher=object(), ModelPatcherDynamic=dynamic)
                memory = SimpleNamespace(aimdo_enabled=False)
                mm = SimpleNamespace(is_nvidia=lambda: True, is_amd=lambda: False,
                                     get_total_memory=lambda _: 32*2**30, get_torch_device=lambda: SimpleNamespace(index=0),
                                     torch_version_numeric=(2, 8), get_all_torch_devices=lambda: [SimpleNamespace(index=0)])
                options = SimpleNamespace(enable_args_parsing=Mock())
                modules = {"comfy.options": options, "comfy.cli_args": cli,
                           "comfy_aimdo.control": control, "comfy.model_management": mm,
                           "comfy.memory_management": memory, "comfy.model_patcher": patchers}

                def importing(name):
                    self.assertIn(str(root / "vendor/python"), sys.path)
                    if name == "comfy.cli_args":
                        self.assertIn("--enable-dynamic-vram", sys.argv)
                        self.assertIn("--disable-cuda-malloc", sys.argv)
                        options.enable_args_parsing.assert_called_once()
                    self.assertNotIn(name, ("main", "server", "nodes"))
                    return modules.get(name, SimpleNamespace())

                spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
                original_argv = sys.argv
                with patch.object(sys, "path", list(sys.path)), patch.dict(sys.modules), \
                     patch.object(backend_module.importlib, "import_module", side_effect=importing), \
                     patch.object(backend_module.importlib.util, "spec_from_file_location", return_value=spec), \
                     patch.object(backend_module.importlib.util, "module_from_spec", return_value=SimpleNamespace()):
                    if initialized:
                        HunyuanBackend(root=root)
                        self.assertIs(patchers.CoreModelPatcher, dynamic)
                        self.assertTrue(memory.aimdo_enabled)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "could not initialize dynamic VRAM"):
                            HunyuanBackend(root=root)
                        self.assertFalse(memory.aimdo_enabled)
                    self.assertIs(sys.argv, original_argv)

    def test_backend_top_level_imports_are_standard_library_only(self):
        tree = ast.parse((ROOT / "pi_hunyuan/backend.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for imported in node.names:
                    self.assertIn(imported.name.split(".")[0], sys.stdlib_module_names)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split(".")[0], sys.stdlib_module_names)


class WorkerProtocolTests(unittest.TestCase):
    def run_worker(self, startup_error=False):
        code = '''
import os, runpy, sys, types
class Backend:
    def __init__(self, mode, vram_profile='auto'):
        assert mode == 'auto'
        print('upstream Python noise', flush=True)
        os.write(1, b'upstream native noise\\n')
        if STARTUP_ERROR:
            raise RuntimeError('missing dependency fixture')
    def generate(self, request, emit):
        if request.get('fail'):
            raise RuntimeError('generation fixture')
        emit(dict(event='progress', id=request['id'], step=1, total=8))
        return dict(event='result', id=request['id'], output='result.png',
                    model_type='instruct_distil', rewritten_prompt='', steps=8, cfg=1.0, guidance=2.5)
sys.modules['backend'] = types.SimpleNamespace(HunyuanBackend=Backend)
sys.argv = [WORKER, '--mode', 'auto']
runpy.run_path(WORKER, run_name='__main__')
'''
        code = "STARTUP_ERROR = " + repr(startup_error) + "\nWORKER = " + repr(str(ROOT / "pi_hunyuan/worker.py")) + "\n" + code
        return subprocess.run([sys.executable, "-B", "-u", "-c", code],
                              input='not JSON\n[]\n{"id":"bad","fail":true}\n{"id":"good"}\n',
                              capture_output=True, text=True, timeout=20)

    def test_clean_json_native_noise_redirection_and_error_recovery(self):
        process = self.run_worker()
        self.assertEqual(process.returncode, 0, process.stderr)
        events = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual([e["event"] for e in events], ["ready", "error", "error", "error", "progress", "result"])
        self.assertEqual([e["id"] for e in events], [None, None, None, "bad", "good", "good"])
        self.assertEqual((events[-1]["steps"], events[-1]["cfg"]), (8, 1.0))
        self.assertIn("upstream Python noise", process.stderr)
        self.assertIn("upstream native noise", process.stderr)
        self.assertIn("generation fixture", events[3]["message"])

    def test_startup_error_is_structured_without_ready(self):
        process = self.run_worker(startup_error=True)
        self.assertEqual(process.returncode, 1)
        events = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["event"], events[0]["id"]), ("error", None))
        self.assertIn("missing dependency fixture", events[0]["message"])


if __name__ == "__main__":
    unittest.main()
