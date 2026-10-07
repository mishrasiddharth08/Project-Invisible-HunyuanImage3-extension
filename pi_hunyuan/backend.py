"""Serverless Hunyuan backend. Import this module only in the isolated worker.

Importing the module itself uses only the standard library; vendor imports happen
after the worker's search paths and Comfy arguments have been configured.
"""
from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import math
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext
from functools import wraps

# Also supports the worker's direct-script import and standalone backend tests.
_EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))


class HunyuanBackend:
    def __init__(self, mode="auto", root=None, vram_profile='auto'):
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        from pi_hunyuan.profiles import normalize
        self.vram_profile = normalize(vram_profile)
        self.memory_plan = {}
        self.model = self.vae = self.vision = None
        self._model_key = self._vae_key = self._vision_key = None
        self._head_key = None
        self._adapter_key = None
        self._active_model = None
        self._adapter_audit = []
        self._quantization = None
        try:
            self._bootstrap(mode)
        except ModuleNotFoundError as exc:
            name = {"PIL": "Pillow", "yaml": "PyYAML"}.get(exc.name, exc.name)
            raise RuntimeError(
                f"Missing worker dependency: {name}. Install the bundled ComfyUI "
                f"requirements into {self.root / 'vendor' / 'python'} using this "
                f"Python ({sys.executable}); restart the worker."
            ) from exc

    def _bootstrap(self, mode):
        if mode not in ("auto", "dynamic", "legacy", "speed", "low-memory"):
            raise ValueError(f"Unknown worker mode: {mode}")
        vendor = self.root / "vendor"
        comfy_path, upstream_path = vendor / "ComfyUI", vendor / "hunyuan"
        for required in (comfy_path / "comfy" / "options.py", upstream_path / "__init__.py"):
            if not required.is_file():
                raise FileNotFoundError(f"Missing bundled backend file: {required}")
        # Add package paths before importing any torch/Comfy/upstream modules.
        import site
        packages = vendor / "python"
        if packages.is_dir():
            site.addsitedir(str(packages))
        for directory in (comfy_path, packages):
            value = str(directory)
            if value in sys.path:
                sys.path.remove(value)
            sys.path.insert(0, value)
        old_argv = sys.argv
        sys.argv = ["pi-hunyuan-worker", "--disable-cuda-malloc"]
        if mode in ("auto", "dynamic", "speed", "low-memory"):
            sys.argv.append("--enable-dynamic-vram")
        elif mode == "legacy":
            sys.argv.append("--disable-dynamic-vram")
        if mode == "low-memory":
            sys.argv.extend(["--vram-headroom", "3", "--disable-smart-memory"])
        elif mode in ('auto', 'dynamic'):
            sys.argv.extend(['--vram-headroom', '2'])
        elif mode == "speed":
            sys.argv.extend(["--async-offload", "2"])
        try:
            options = importlib.import_module("comfy.options")
            options.enable_args_parsing()
            cli = importlib.import_module("comfy.cli_args")
            control = importlib.import_module("comfy_aimdo.control")
            if os.name == "nt":
                os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
            if cli.enables_dynamic_vram():
                headroom = None if cli.args.reserve_vram is None else int(cli.args.reserve_vram * 2**30)
                try:
                    control.init(simple_vram_headroom=headroom,
                                 nvml_pressure=not cli.args.disable_nvml_pressure)
                except TypeError:
                    try:
                        control.init(simple_vram_headroom=headroom)
                    except TypeError:
                        control.init()
            self.torch = importlib.import_module("torch")
            if mode != 'speed' or self.vram_profile != 'auto':
                cli.args.async_offload = 1
            elif hasattr(self.torch, 'cuda') and self.torch.cuda.is_available():
                from pi_hunyuan.profiles import plan as preliminary_plan
                device_gib = self.torch.cuda.get_device_properties(self.torch.cuda.current_device()).total_memory / 2**30
                if preliminary_plan(device_gib, self.vram_profile, mode)['nominal_gb'] <= 12:
                    cli.args.async_offload = 1
            self.mm = importlib.import_module("comfy.model_management")
            from pi_hunyuan.profiles import plan
            total_gib = self.mm.get_total_memory(self.mm.get_torch_device()) / 2**30
            self.memory_plan = plan(total_gib, self.vram_profile, mode)
            cli.args.vram_headroom = self.memory_plan['headroom_gib']
            logging.info('Hunyuan memory profile: %s', self.memory_plan)
            memory = importlib.import_module("comfy.memory_management")
            patchers = importlib.import_module("comfy.model_patcher")
            supported = self.mm.is_nvidia() or (
                self.mm.is_amd() and self.mm.rocm_version >= (7, 14))
            if cli.enables_dynamic_vram() and supported:
                if self.mm.torch_version_numeric < (2, 8):
                    raise RuntimeError("Dynamic VRAM requires PyTorch 2.8 or newer.")
                devices = self.mm.get_all_torch_devices()
                try:
                    initialized = control.init_devices(
                        (device.index, int(cli.args.vram_headroom * 2**30)) for device in devices)
                except TypeError:
                    initialized = control.init_devices(device.index for device in devices)
                if not initialized:
                    raise RuntimeError("comfy-aimdo could not initialize dynamic VRAM. "
                                       "Check its CUDA/driver compatibility or explicitly use legacy mode.")
                patchers.CoreModelPatcher = patchers.ModelPatcherDynamic
                memory.aimdo_enabled = True
                logging.info("Hunyuan worker: dynamic VRAM enabled")
            elif mode in ("auto", "dynamic", "speed", "low-memory"):
                raise RuntimeError("Dynamic VRAM is unavailable on this device.")
            self.folders = importlib.import_module("folder_paths")
            self.sample = importlib.import_module("comfy.sample")
            self.clip_vision = importlib.import_module("comfy.clip_vision")
            self.np = importlib.import_module("numpy")
            self.Image = importlib.import_module("PIL.Image")
            self.ImageOps = importlib.import_module("PIL.ImageOps")
            alias = "pi_hunyuan_upstream"
            spec = importlib.util.spec_from_file_location(
                alias, upstream_path / "__init__.py",
                submodule_search_locations=[str(upstream_path)])
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import bundled Hunyuan package: {upstream_path}")
            package = importlib.util.module_from_spec(spec)
            sys.modules[alias] = package
            spec.loader.exec_module(package)
            self.nodes = importlib.import_module(f"{alias}.nodes")
            self.loader = importlib.import_module(f"{alias}.hunyuan_image_3.loader")
            self.rewrite = importlib.import_module(f"{alias}.hunyuan_image_3.rewrite")
        finally:
            sys.argv = old_argv

    @staticmethod
    def _file(value, label, required=True):
        if not value:
            if required:
                raise ValueError(f"Missing {label} path")
            return None
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
        return path

    @staticmethod
    def _key(path):
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _number(request, name, default, minimum=0):
        value = float(request.get(name, default))
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"{name} must be finite and at least {minimum}")
        return value

    def _folders(self, paths):
        # Use only the selected file directories; clear Comfy's filename cache
        # so a same-name file from a previous request cannot win resolution.
        for category, files in (
            # The model loader uses an absolute path. Give the explicitly
            # selected head priority even if another directory has its basename.
            ("diffusion_models", [paths.get("head"), paths["model"]]),
            ("vae", [paths["vae"]]), ("clip_vision", [paths.get("vision")]),
        ):
            directories = list(dict.fromkeys(str(p.parent) for p in files if p))
            extensions = self.folders.folder_names_and_paths[category][1]
            self.folders.folder_names_and_paths[category] = (directories, extensions)
        self.folders.filename_list_cache.clear()

    def _release(self):
        self.model = self.vae = self.vision = None
        self._active_model = None
        self._adapter_key = None
        self._adapter_audit = []
        self._model_key = self._vae_key = self._vision_key = None
        self._head_key = None
        self.rewrite._HEADS.clear()
        gc.collect()
        self.mm.unload_all_models()
        self.mm.soft_empty_cache()

    def _adapters(self, paths):
        records = []
        if not isinstance(paths, list):
            raise ValueError('Adapters must be a list of local files and strengths.')
        for item in paths:
            if not isinstance(item, dict):
                raise ValueError('Invalid adapter description.')
            path = self._file(item.get('path'), 'adapter')
            strength = float(item.get('strength', 1))
            if not math.isfinite(strength) or not -10 <= strength <= 10:
                raise ValueError('Adapter strength must be finite and between -10 and 10.')
            records.append((self._key(path), strength))
        key = (self._model_key, tuple(records))
        if key != self._adapter_key:
            if self._active_model is not None and self._active_model is not self.model:
                self.mm.unload_all_models()
            self._active_model = None
            if paths:
                from pi_hunyuan.adapters import apply_adapters
                self._active_model, self._adapter_audit = apply_adapters(self.model, paths)
            else:
                self._active_model, self._adapter_audit = self.model, []
            self._adapter_key = key
        return self._active_model

    def _budget_plain_banks(self):
        if not self.memory_plan.get('tiled', False):
            return
        for _, module in self.model.model.diffusion_model.named_modules():
            if callable(getattr(module, 'expert_linear', None)) and callable(getattr(module, 'bank_resident', None)) and not getattr(module, 'quant_format', None):
                # Quantless/GGUF paths can fetch one expert. Avoid staging an
                # entire BF16 bank on cards with a small working-memory budget.
                module.bank_resident = lambda _input: nullcontext()

    def _disable_prefetch(self):
        self.model.model_options.setdefault('transformer_options', {})['prefetch_dynamic_vbars'] = False
        diffusion = self.model.model.diffusion_model
        original_forward = diffusion.forward
        @wraps(original_forward)
        def budget_forward(*args, **kwargs):
            options = dict(kwargs.get('transformer_options') or {})
            options['prefetch_dynamic_vbars'] = False
            kwargs['transformer_options'] = options
            return original_forward(*args, **kwargs)
        diffusion.forward = budget_forward

    def _check_available_budget(self):
        cuda = getattr(self.torch, 'cuda', None)
        if not cuda or not cuda.is_available() or not self.memory_plan:
            return
        free, total = cuda.mem_get_info()
        used_gib = (total-free) / 2**30
        budget = self.memory_plan['budget_gib']
        if budget-used_gib < 3:
            raise RuntimeError(f'GPU memory is busy: {used_gib:.1f} GiB already used, '
                               f'{budget:.1f} GiB profile budget. Finish other GPU jobs or '
                               'release their model memory before starting Hunyuan.')

    def generate(self, request, emit):
        from pi_hunyuan.metrics import Measurement
        measurement = Measurement(self.torch).begin()
        try:
            result = self._generate(request, emit)
            result['metrics'] = measurement.finish()
            return result
        except BaseException as error:
            error.hunyuan_metrics = measurement.finish()
            raise

    def _generate(self, request, emit):
        """Generate one PNG. The parent trusts paths and terminates us to cancel."""
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        selected = request.get("paths")
        if not isinstance(selected, dict):
            raise ValueError("paths must contain model and vae files")
        references = request.get("references") or []
        if not isinstance(references, list) or len(references) > 3:
            raise ValueError("references must be a list of at most three image paths")
        references = [self._file(p, "reference") for p in references]
        rewriting = request.get("rewrite", "off")
        if rewriting not in ("off", "rewrite only", "think + rewrite"):
            raise ValueError("rewrite must be off, rewrite only, or think + rewrite")
        paths = {name: self._file(selected.get(name), name, name in ("model", "vae"))
                 for name in ("model", "vae", "vision", "head")}
        if references and paths["vision"] is None:
            raise ValueError("Image editing requires paths.vision")
        width, height = int(request.get("width", 1024)), int(request.get("height", 1024))
        if any(n < 16 or n > 16384 or n % 16 for n in (width, height)):
            raise ValueError("width and height must be multiples of 16, from 16 to 16384")
        seed = int(request.get("seed", 0))
        if not 0 <= seed <= 0xffffffffffffffff:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not request.get("output"):
            raise ValueError("Missing output path")
        output = Path(request["output"]).expanduser().resolve()
        if output.suffix.lower() != ".png":
            raise ValueError("output must name a PNG file")
        prompt = request.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be text")
        guidance = self._number(request, "guidance", 2.5)
        if "cfg" in request:
            self._number(request, "cfg", 1.0)
        requested_steps = int(request.get("steps", 50))
        if requested_steps < 1:
            raise ValueError("steps must be positive")
        if not isinstance(request.get("spectrum", False), bool):
            raise ValueError("spectrum must be a boolean")
        self._folders(paths)
        started = time.perf_counter()
        key = self._key(paths["model"])
        reused_model = self._model_key == key
        if self._model_key != key:
            self._check_available_budget()
            self._release()
            from pi_hunyuan.quantization import validate_checkpoint, load_gguf
            self._quantization = validate_checkpoint(paths['model'], model_management=self.mm,
                                                     allow_experimental_gguf=paths['model'].suffix.lower() == '.gguf',
                                                     original_loader=self.loader)
            if paths['model'].suffix.lower() == '.gguf':
                self.model, _ = load_gguf(paths['model'], original_loader=self.loader)
            else:
                self.model, _ = self.loader.load_hunyuan_image_3(str(paths["model"]))
            self._model_key = key
            if self.memory_plan.get('tiled', False):
                self._budget_plain_banks()
            if self.memory_plan.get('headroom_gib', 0) > 0:
                # Comfy rewrites the flag: enforce it at the model entry point.
                self._disable_prefetch()
        active_model = self._adapters(request.get('adapters', []))
        model_type = self.model.model.diffusion_model.config.model_type
        if model_type == "base" and references:
            raise ValueError("Base supports text-to-image only; choose Instruct or Instruct-Distil for editing")
        steps = 8 if model_type == "instruct_distil" else requested_steps
        cfg = self._number(request, "cfg", 5.0 if model_type == "base" else 2.5)
        if model_type == "instruct_distil":
            if requested_steps != 8 or cfg != 1.0:
                logging.info("Instruct-Distil contract: using 8 steps and CFG 1.0")
            cfg = 1.0
        vae_key = self._key(paths["vae"])
        if self._vae_key != vae_key:
            self.vae = None
            self._vae_key = None
            gc.collect()
            self.mm.unload_all_models()
            self.vae = self.nodes.HunyuanImage3VAELoader.execute(paths["vae"].name).result[0]
            self._vae_key = vae_key
        rewriting_settings = None
        if rewriting != "off" and model_type != "base":
            head_key = self._key(paths["head"]) if paths["head"] else None
            if head_key != self._head_key:
                self.rewrite._HEADS.clear()
                self._head_key = head_key
                gc.collect()
                self.mm.unload_all_models()
            rewriting_settings = self.rewrite.RewriteSettings(
                mode=rewriting, cot_head=paths["head"].name if paths["head"] else self.rewrite.AUTO,
                seed=seed)
        elif rewriting != "off":
            logging.info("Base does not support prompt rewriting; skipping it")
        loaded_at = time.perf_counter()
        self.mm.interrupt_current_processing(False)
        try:
            with self.torch.inference_mode():
                if references:
                    vision_key = self._key(paths["vision"])
                    if self._vision_key != vision_key:
                        self.vision = None
                        self._vision_key = None
                        gc.collect()
                        self.vision = self.clip_vision.load(str(paths["vision"]))
                        if self.vision is None:
                            raise ValueError("Vision checkpoint is not a supported SigLIP2 checkpoint")
                        self._vision_key = vision_key
                    images = {}
                    for index, path in enumerate(references, 1):
                        with self.Image.open(path) as image:
                            image = self.ImageOps.exif_transpose(image).convert("RGB")
                            pixels = self.np.array(image, dtype=self.np.float32) / 255.0
                        images[f"image_{index}"] = self.torch.from_numpy(pixels).unsqueeze(0)
                    positive, negative, rewritten = self.nodes.HunyuanImage3ImageEncode.execute(
                        active_model, self.vae, self.vision, images, prompt, width, height,
                        prompt_rewriting=rewriting_settings).result
                else:
                    positive, negative, rewritten = self.nodes._encode(
                        active_model, prompt, width, height, None, rewriting_settings).result
                if model_type == "instruct_distil":
                    positive = self.nodes.HunyuanImage3Guidance.execute(positive, guidance).result[0]
                encoded_at = time.perf_counter()
                model = active_model
                if request.get("spectrum", False):
                    model = self.nodes.HunyuanImage3Spectrum.execute(
                        model, True, warmup_steps=5, window_size=2.0, flex_window=0.75,
                        w=0.5, max_w=0.8, M=4, lam=0.1, history=12,
                        time_axis="step", validate=False, verbose=False).result[0]
                latent = self.nodes.HunyuanImage3EmptyLatent.execute(width, height, 1).result[0]["samples"]
                noise = self.sample.prepare_noise(latent, seed)
                emit({"event": "progress", "id": request.get("id"), "step": 0, "total": steps})

                def callback(step, x0, x, total):
                    emit({"event": "progress", "id": request.get("id"),
                          "step": step + 1, "total": total})

                with self.mm.cuda_device_context(model.load_device):
                    samples = self.sample.sample(
                        model, noise, steps, cfg, "euler", "simple", positive, negative,
                        latent, denoise=1.0, callback=callback, disable_pbar=True, seed=seed)
                sampled_at = time.perf_counter()
                # Comfy's sampler returns VAE-space latents: do not rescale again.
                use_tiled = bool(request.get('vae_tiled', False)) or self.memory_plan.get('tiled', False)
                if use_tiled:
                    decoded = self.vae.decode_tiled(samples, tile_x=32, tile_y=32, overlap=8, tile_t=2, overlap_t=1)
                else:
                    decoded = self.vae.decode(samples)
                pixels = decoded[0].detach().float().cpu().numpy()
                # The Hunyuan image VAE returns [batch, frame, height, width, RGB].
                # Keep support for conventional [batch, height, width, RGB] VAEs.
                if pixels.ndim == 4 and pixels.shape[0] == 1:
                    pixels = pixels[0]
                if pixels.ndim != 3 or pixels.shape[-1] != 3:
                    raise RuntimeError(f"Unexpected VAE image shape: {pixels.shape}")
                if not self.np.isfinite(pixels).all():
                    raise RuntimeError("VAE produced non-finite pixels")
                pixels = self.np.clip(pixels * 255.0, 0, 255).astype(self.np.uint8)
                output.parent.mkdir(parents=True, exist_ok=True)
                self.Image.fromarray(pixels).save(output, format="PNG")
                decoded_at = time.perf_counter()
            if request.get('offload_after', False):
                self.mm.unload_all_models()
                self.mm.soft_empty_cache()
            return {"event": "result", "id": request.get("id"), "output": str(output),
                    "model_type": model_type, "rewritten_prompt": rewritten or "",
                    "steps": steps, "cfg": cfg, "guidance": guidance,
                    "model_reused": reused_model, "vae_tiled": use_tiled,
                    "memory_profile": self.memory_plan,
                    "adapters": self._adapter_audit,
                    "quantization": {key: self._quantization.get(key) for key in ('format', 'quantization', 'dtypes', 'container')} if self._quantization else None,
                    "stage_seconds": {'load': round(loaded_at-started, 3), 'encode': round(encoded_at-loaded_at, 3),
                                      'sample': round(sampled_at-encoded_at, 3), 'decode_save': round(decoded_at-sampled_at, 3)}}
        except Exception:
            try:
                self._release()
            except Exception:
                logging.exception("Worker cleanup failed after generation error")
            raise
