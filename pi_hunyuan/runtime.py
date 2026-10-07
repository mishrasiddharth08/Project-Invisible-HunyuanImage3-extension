import tempfile
import threading
import time
from pathlib import Path
from PIL import Image
from .assets import scan, resolve, companion, variant
from .request import validate
from .transport import Worker
from .config import models_root, settings
import sys

LOCK = threading.RLock()
_worker = None
_cancel = threading.Event()
_idle_timer = None
_idle_epoch = 0


def _cancel_idle():
    global _idle_timer, _idle_epoch
    _idle_epoch += 1
    if _idle_timer:
        _idle_timer.cancel()
        _idle_timer = None


def _arm_idle():
    global _idle_timer
    _cancel_idle()
    worker, token = _worker, _idle_epoch
    seconds = max(10, min(3600, float(settings().get('cache_idle_seconds', 180))))
    deadline = time.monotonic() + seconds
    def expire():
        global _idle_timer
        with LOCK:
            if token != _idle_epoch or worker is not _worker:
                return
            pressure = False
            try:
                import psutil
                floor = max(1, float(settings().get('cache_ram_floor_gib', 8)))
                pressure = psutil.virtual_memory().available < floor * 2**30
            except (ImportError, OSError):
                pass
            remaining = deadline - time.monotonic()
            if pressure or remaining <= 0:
                release()
            else:
                _idle_timer = threading.Timer(min(10, remaining), expire)
                _idle_timer.daemon = True
                _idle_timer.start()
    _idle_timer = threading.Timer(min(10, seconds), expire)
    _idle_timer.daemon = True
    _idle_timer.start()


def release():
    global _worker
    _cancel.set()
    _cancel_idle()
    worker, _worker = _worker, None
    if worker:
        worker.close()


def generate(p, options):
    global _worker
    from modules import shared, processing, images
    from backend import memory_management
    with LOCK:
        _cancel_idle()
        _cancel.clear()
        performance = options.get('performance', 'balanced')
        if performance not in ('balanced', 'speed', 'low-memory'):
            raise ValueError('Choose Balanced, Speed or Low memory.')
        mode = {'balanced': 'auto', 'speed': 'speed', 'low-memory': 'low-memory'}[performance]
        from .profiles import normalize
        vram_profile = normalize(options.get('vram_profile', 'auto'))
        tiled = bool(options.get('vae_tiled', False)) or performance == 'low-memory'
        editing = isinstance(p, processing.StableDiffusionProcessingImg2Img)
        refs = list(getattr(p, 'init_images', []) or []) if editing else []
        if len(refs) > 1:
            raise ValueError('Use one img2img image and the two additional reference slots.')
        refs += [image for image in options.get('refs', []) if image is not None] if editing else []
        if editing and not refs:
            raise ValueError('Add an image to img2img before Generate.')
        if any(not isinstance(image, Image.Image) for image in refs):
            raise ValueError('Each reference must be an image.')
        validate(p, refs, options)
        inventory = scan()
        paths = {role: resolve(options.get(role), role, inventory) for role in ('model', 'vae')}
        rewriting = options.get('rewrite', 'off') != 'off'
        if variant(paths['model']) == 'base' and (refs or rewriting):
            raise ValueError('Base supports text-to-image only. Choose Instruct or Instruct-Distil for editing or rewriting.')
        paths['vision'] = companion(options.get('vision'), 'vision', paths['model'], inventory, bool(refs)) if refs else None
        paths['head'] = companion(options.get('head'), 'head', paths['model'], inventory, rewriting) if rewriting else None
        processing.fix_seed(p)
        count = int(p.batch_size * p.n_iter)
        output, seeds, prompts, infos = [], [], [], []
        shared.state.job_count = count
        cancelled = lambda: bool(shared.state.interrupted or shared.state.skipped or _cancel.is_set())
        try:
            memory_management.unload_all_models()
            for index in range(count):
                if shared.state.interrupted or _cancel.is_set():
                    break
                shared.state.skipped = False
                prompt = p.prompt[index % len(p.prompt)] if isinstance(p.prompt, list) else p.prompt
                if getattr(p, 'styles', []):
                    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, p.styles)
                    negative = shared.prompt_styles.apply_negative_styles_to_prompt('', p.styles)
                    if negative.strip():
                        raise ValueError('Choose a style without a negative prompt for HunyuanImage 3.')
                from .adapters import parse_prompt, resolve_adapters
                clean_prompt, tags = parse_prompt(prompt)
                roots = [models_root() / 'Lora', models_root() / 'LyCORIS']
                roots.extend(Path(value) for value in settings().get('adapter_roots', []))
                command_options = getattr(shared, 'cmd_opts', None)
                for key in ('lora_dir', 'lyco_dir', 'lycoris_dir'):
                    value = getattr(command_options, key, None)
                    if value:
                        roots.append(Path(value))
                native_networks = sys.modules.get('networks')
                adapters = []
                for name, strength in tags:
                    aliases = getattr(native_networks, 'available_network_aliases', {}) or {}
                    names = getattr(native_networks, 'available_networks', {}) or {}
                    known = names.get(name) or aliases.get(name)
                    filename = getattr(known, 'filename', None)
                    if filename and Path(filename).is_file():
                        adapters.append({'path': str(Path(filename).resolve()), 'strength': strength})
                    else:
                        adapters.extend(resolve_adapters([(name, strength)], roots))
                seed = (int(p.seed) + index) % (2**63)
                shared.state.job = f'HunyuanImage 3: {index + 1}/{count}'
                shared.state.sampling_steps = int(p.steps)
                shared.state.sampling_step = 0
                if _worker is not None and (getattr(_worker, 'mode', mode) != mode or getattr(_worker, 'vram_profile', vram_profile) != vram_profile):
                    release()
                    _cancel.clear()
                if _worker is None or _worker.process.poll() is not None:
                    _worker = Worker(mode=mode, vram_profile=vram_profile)
                    worker = _worker
                    try:
                        worker.wait_ready(cancelled)
                    except (InterruptedError, RuntimeError, OSError):
                        if cancelled():
                            break
                        raise
                worker = _worker
                if worker is None or cancelled():
                    break
                with tempfile.TemporaryDirectory(prefix='pi-hunyuan3-') as temp:
                    references = []
                    for n, image in enumerate(refs):
                        path = Path(temp) / f'reference-{n}.png'
                        image.convert('RGB').save(path)
                        references.append(str(path))
                    def progress(event):
                        shared.state.sampling_step = int(event.get('step', 0))
                        shared.state.sampling_steps = int(event.get('total', p.steps))
                    request = dict(paths=paths, prompt=clean_prompt, adapters=adapters, references=references, rewrite=options.get('rewrite', 'off'), spectrum=bool(options.get('spectrum', False)), seed=seed, width=int(p.width), height=int(p.height), steps=int(p.steps), cfg=float(p.cfg_scale), guidance=float(getattr(p, 'distilled_cfg_scale', 2.5)), vae_tiled=tiled, offload_after=performance != 'speed', output=str(Path(temp) / 'result.png'))
                    try:
                        event = worker.generate(request, cancelled, progress)
                    except (InterruptedError, RuntimeError, OSError):
                        if not cancelled():
                            raise
                        if shared.state.skipped and not shared.state.interrupted and not _cancel.is_set():
                            worker, _worker = _worker, None
                            if worker:
                                worker.close()
                            shared.state.skipped = False
                            shared.state.nextjob()
                            continue
                        break
                    with Image.open(request['output']) as loaded:
                        result = loaded.convert('RGB').copy()
                if cancelled():
                    if shared.state.skipped and not shared.state.interrupted and not _cancel.is_set():
                        shared.state.skipped = False
                        shared.state.nextjob()
                        continue
                    break
                if result.size != (p.width, p.height):
                    raise RuntimeError('Hunyuan returned an unexpected image size.')
                params = {'Hunyuan model type': event.get('model_type'), 'Hunyuan references': len(refs), 'Hunyuan rewrite': options.get('rewrite', 'off'), 'Hunyuan Spectrum': bool(options.get('spectrum', False))}
                params.update({'Hunyuan memory mode': performance, 'Hunyuan tiled VAE': event.get('vae_tiled', tiled),
                               'Hunyuan VRAM profile': event.get('memory_profile', {}).get('nominal_gb', vram_profile),
                               'Hunyuan warm cache': event.get('model_reused', False)})
                if event.get('adapters'):
                    params['Hunyuan adapters'] = event['adapters']
                if event.get('quantization'):
                    params['Hunyuan quantization'] = event['quantization'].get('format', event['quantization']) if isinstance(event['quantization'], dict) else event['quantization']
                metrics = event.get('metrics', {})
                for key, label in [('gpu_total_peak_mib', 'Hunyuan total GPU peak MiB'), ('process_rss_peak_gib', 'Hunyuan worker RAM peak GiB'), ('generation_seconds', 'Hunyuan worker seconds')]:
                    if key in metrics:
                        params[label] = metrics[key]
                if event.get('rewritten_prompt'):
                    params['Hunyuan rewritten prompt'] = event['rewritten_prompt']
                p.extra_generation_params.update(params)
                p.sd_model_name = Path(paths['model']).stem
                p.sd_vae_name = Path(paths['vae']).stem
                effective_steps = event.get('steps', p.steps)
                effective_cfg = event.get('cfg', p.cfg_scale)
                p.steps, p.cfg_scale = int(effective_steps), float(effective_cfg)
                guidance_info = f'Distilled CFG Scale: {event.get("guidance", request["guidance"])}, ' if event.get('model_type') == 'instruct_distil' else ''
                info = f'{prompt}\nSteps: {effective_steps}, Sampler: Euler, Schedule type: Simple, CFG scale: {effective_cfg}, {guidance_info}Seed: {seed}, Size: {p.width}x{p.height}, Model: {p.sd_model_name}, VAE: {p.sd_vae_name}, ' + ', '.join(f'{k}: {v}' for k, v in params.items())
                result.info['parameters'] = info
                if cancelled():
                    break
                if shared.opts.samples_save and not getattr(p, 'do_not_save_samples', False):
                    saved = images.save_image(result, p.outpath_samples, '', seed, prompt, extension=shared.opts.samples_format, info=info, p=p)
                    if not saved or not saved[0] or not Path(saved[0]).is_file():
                        raise RuntimeError('Image generated, but Forge could not save it.')
                output.append(result)
                seeds.append(seed)
                prompts.append(prompt)
                infos.append(info)
                shared.state.current_image = result
                shared.state.nextjob()
            return processing.Processed(p, output, seed=seeds[0] if seeds else int(p.seed), info=infos[0] if infos else 'HunyuanImage 3 stopped.', all_seeds=seeds, all_prompts=prompts, all_negative_prompts=[''] * len(output), infotexts=infos)
        except BaseException:
            release()
            raise
        finally:
            if not options.get('keep_loaded', False) or cancelled() or performance == 'low-memory':
                release()
            else:
                _arm_idle()
