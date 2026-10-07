"""Extension-owned checkpoint routing through Forge's existing Generate/API flow."""
from functools import wraps
from pathlib import Path

from . import assets, runtime
from .config import ROOT, LABEL, PRESET

KEYS = ('model', 'vae', 'vision', 'head', 'rewrite', 'spectrum', 'keep_loaded', 'reference2', 'reference3', 'performance', 'vae_tiled', 'vram_profile')
DEFAULTS = ('Auto', 'Auto', 'Auto', 'Auto', 'off', False, True, None, None, 'balanced', False, 'auto')
MARKER = '_pi_hunyuan3'
PANELS = []
COMPONENTS = {}
BOUND = set()
HOOKS = []
ENABLED = True


def _owned_hook(value):
    # functools.wraps copies attributes onto other extensions' wrappers.
    # A marker alone is not proof that the current callable is ours.
    return any(value is replacement for _, _, _, replacement in HOOKS)


def _set_hook(owner, name, replacement):
    previous = getattr(owner, name)
    if previous is replacement:
        return
    HOOKS.append((owner, name, previous, replacement))
    setattr(owner, name, replacement)


def uninstall():
    global ENABLED
    ENABLED = False
    try:
        runtime.release()
    finally:
        for owner, name, previous, replacement in reversed(HOOKS):
            if getattr(owner, name) is replacement:
                setattr(owner, name, previous)
        HOOKS.clear()
        reset_ui()
        install._registered = False


def setting(p, name, default=None):
    from modules import shared
    overrides = getattr(p, 'override_settings', {}) or {}
    return overrides[name] if name in overrides else getattr(shared.opts, name, default)


def active(p=None):
    return setting(p, 'forge_preset') == PRESET


def identity(value):
    return str(Path(str(value)).resolve()).casefold()


def checkpoint(p=None):
    from modules import sd_models
    value = setting(p, 'sd_model_checkpoint', '')
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    return value, item, getattr(item, 'filename', value)


def owned(p=None):
    value, item, _ = checkpoint(p)
    return value == LABEL or getattr(item, MARKER, False) is True


def selected(p=None):
    if not active(p):
        return False
    if owned(p):
        return True
    _, _, filename = checkpoint(p)
    return identity(filename) in {identity(path) for path in assets.scan()['model']}


def register():
    from modules import sd_models
    marker = str(Path(ROOT) / 'resources' / 'hunyuan3' / 'model_index.json')
    candidates = [(LABEL, marker)] + [(f'HunyuanImage 3.0 â€” {path}', str(path)) for path in assets.scan()['model']]
    for label, path in candidates:
        existing = next((item for item in sd_models.checkpoints_list.values()
                         if getattr(item, MARKER, False) is True and item.filename == path), None)
        if existing is not None:
            continue
        item = object.__new__(sd_models.CheckpointInfo)
        item.filename = path
        item.name = item.title = item.short_title = item.model_name = item.name_for_extra = label
        item.hash = PRESET
        item.sha256 = item.shorthash = None
        item.metadata = {}
        item.is_safetensors = False
        item.ids = [label, path]
        item.calculate_shorthash = lambda: None
        setattr(item, MARKER, True)
        item.register()


def options(runner, p):
    result = dict(zip(KEYS, DEFAULTS))
    for script in getattr(runner, 'alwayson_scripts', []) or []:
        if getattr(script, MARKER, False):
            values = list(getattr(p, 'script_args', []) or [])[script.args_from:script.args_to]
            result.update((key, value) for key, value in zip(KEYS, values) if value is not None)
            break
    inventory = assets.scan()
    value, item, filename = checkpoint(p)
    if identity(filename) in {identity(path) for path in inventory['model']}:
        result['model'] = filename
    elif value != LABEL and not getattr(item, MARKER, False):
        raise ValueError('Choose a HunyuanImage 3.0 checkpoint before Generate.')
    overrides = getattr(p, 'override_settings', {}) or {}
    modules = setting(p, 'forge_additional_modules', []) or []
    if 'forge_additional_modules' in overrides:
        for role in ('vae', 'vision', 'head'):
            result[role] = 'Auto'
    for module in modules:
        matches = [role for role in ('vae', 'vision', 'head')
                   if identity(module) in {identity(path) for path in inventory[role]}]
        if not matches:
            raise ValueError(f'Unrecognized Hunyuan component: {module}')
        for role in matches:
            result[role] = str(module)
    if result['rewrite'] not in ('off', 'rewrite only', 'think + rewrite'):
        raise ValueError('Choose a supported Hunyuan prompt rewrite mode.')
    for role in ('model', 'vae'):
        result[role] = assets.resolve(result[role], role, inventory, required=True)
    from modules import processing
    if not isinstance(p, processing.StableDiffusionProcessingImg2Img):
        result['reference2'] = result['reference3'] = None
    result['refs'] = [result[key] for key in ('reference2', 'reference3') if result[key] is not None]
    return result


def wrap(original, runner=False):
    if _owned_hook(original):
        return original
    @wraps(original)
    def routed(*args, **kwargs):
        if not ENABLED:
            return original(*args, **kwargs)
        index = 1 if runner else 0
        p = args[index] if len(args) > index else kwargs['p']
        scripts = (args[0] if args else kwargs.get('runner')) if runner else getattr(p, 'scripts', None)
        if active(p):
            if not selected(p):
                raise ValueError('Choose a HunyuanImage 3.0 checkpoint for the hunyuan3 preset.')
            return runtime.generate(p, options(scripts, p))
        if owned(p):
            raise ValueError('Select the hunyuan3 preset to use this checkpoint.')
        runtime.release()
        return original(*args, **kwargs)
    setattr(routed, MARKER, True)
    return routed


def selection_release():
    from modules import shared
    for key in ('forge_preset', 'sd_model_checkpoint', 'forge_additional_modules'):
        info = shared.opts.data_labels.get(key)
        if info is None or _owned_hook(info.onchange):
            continue
        previous = info.onchange
        initial = getattr(shared.opts, key, None)
        last = [initial.copy() if isinstance(initial, list) else initial]
        def changed(*args, previous=previous, key=key, last=last, **kwargs):
            value = getattr(shared.opts, key, None)
            if ENABLED and value != last[0]:
                runtime.release()
            last[0] = value.copy() if isinstance(value, list) else value
            if previous is not None:
                return previous(*args, **kwargs)
        setattr(changed, MARKER, True)
        shared.opts.onchange(key, changed, call=False)
        HOOKS.append((info, "onchange", previous, changed))


def bind_panels(*_args):
    import gradio as gr
    components = [COMPONENTS.get('forge_ui_preset') or COMPONENTS.get('setting_forge_preset')]
    for component in components:
        if component is None:
            continue
        if component in BOUND or not PANELS:
            continue
        panels = list(PANELS)
        denoise = COMPONENTS.get('img2img_denoising_strength')
        outputs = panels + ([denoise] if denoise is not None else [])
        def updates(value, panels=panels, denoise=denoise):
            result = [gr.update(visible=value == PRESET) for _ in panels]
            if denoise is not None:
                result.append(gr.update(value=1.0) if value == PRESET else gr.update())
            return result
        component.change(updates, inputs=[component], outputs=outputs, queue=False)
        try:
            from gradio.context import Context
        except ImportError:
            Context = None
        if Context is not None and Context.root_block is not None:
            Context.root_block.load(updates, inputs=[component], outputs=outputs, queue=False)
        BOUND.add(component)
    return []


def capture(component, **kwargs):
    elem_id = kwargs.get('elem_id', getattr(component, 'elem_id', None))
    if elem_id in ('setting_forge_preset', 'forge_ui_preset', 'img2img_denoising_strength'):
        COMPONENTS[elem_id] = component


def reset_ui(*_args):
    PANELS.clear()
    COMPONENTS.clear()
    BOUND.clear()


def protect_loader():
    from modules import sd_models
    original = sd_models.forge_model_reload
    if not _owned_hook(original):
        @wraps(original)
        def reload_model(*args, **kwargs):
            if not ENABLED:
                return original(*args, **kwargs)
            parameters = getattr(sd_models.model_data, 'forge_loading_parameters', {}) or {}
            item = parameters.get('checkpoint_info')
            if owned() or getattr(item, MARKER, False):
                raise RuntimeError('Hunyuan worker checkpoint cannot be loaded by the standard Forge loader.')
            return original(*args, **kwargs)
        setattr(reload_model, MARKER, True)
        _set_hook(sd_models, "forge_model_reload", reload_model)
    from modules_forge import main_entry
    original_distill = getattr(main_entry, 'use_distill', None)
    if callable(original_distill) and not _owned_hook(original_distill):
        def use_distill(preset):
            return True if ENABLED and preset == PRESET else original_distill(preset)
        setattr(use_distill, MARKER, True)
        _set_hook(main_entry, "use_distill", use_distill)
    original_change = main_entry.checkpoint_change
    if not _owned_hook(original_change):
        @wraps(original_change)
        def change(value, preset, save=True, refresh=True):
            if not ENABLED:
                return original_change(value, preset, save=save, refresh=refresh)
            from types import SimpleNamespace
            from modules import shared
            p = SimpleNamespace(override_settings={'sd_model_checkpoint': value, 'forge_preset': preset})
            if not selected(p):
                if owned(p):
                    raise ValueError('Select the hunyuan3 preset to use this checkpoint.')
                return original_change(value, preset, save=save, refresh=refresh)
            changed = identity(checkpoint()[2]) != identity(checkpoint(p)[2])
            shared.opts.set('sd_model_checkpoint', value)
            shared.opts.set('forge_checkpoint_' + PRESET, value)
            if save:
                shared.opts.save(shared.config_filename)
            return changed
        setattr(change, MARKER, True)
        _set_hook(main_entry, "checkpoint_change", change)


def ready(*_args, **_kwargs):
    import sys
    from modules import processing
    _set_hook(processing, "process_images", wrap(processing.process_images))
    for name in ('modules.api.api', 'modules.txt2img', 'modules.img2img'):
        module = sys.modules.get(name)
        if module is not None and callable(getattr(module, 'process_images', None)):
            _set_hook(module, "process_images", wrap(module.process_images))
    selection_release()
    protect_loader()
    register()


def install_component_refresh():
    from modules_forge import main_entry
    original = getattr(main_entry, 'refresh_models', None)
    if not callable(original) or _owned_hook(original):
        return
    @wraps(original)
    def refresh(*args, **kwargs):
        if not ENABLED:
            return original(*args, **kwargs)
        checkpoints, modules = original(*args, **kwargs)
        inventory = assets.scan()
        for role in ('vae', 'vision', 'head'):
            for filename in inventory[role]:
                name = Path(filename).name
                # Forge maps display names to paths. Preserve existing names.
                if name in main_entry.module_list and identity(main_entry.module_list[name]) != identity(filename):
                    name = filename
                main_entry.module_list[name] = filename
        return checkpoints, sorted(set(modules) | set(main_entry.module_list))
    setattr(refresh, MARKER, True)
    _set_hook(main_entry, "refresh_models", refresh)


def install():
    global ENABLED
    from modules import scripts, processing, sd_models, script_callbacks
    from modules_forge import main_entry
    required = (getattr(processing, 'process_images', None), getattr(scripts.ScriptRunner, 'run', None),
                getattr(sd_models, 'forge_model_reload', None), getattr(main_entry, 'checkpoint_change', None))
    if not all(callable(value) for value in required):
        raise RuntimeError('Forge Generate hooks are unavailable; Hunyuan integration disabled.')
    ENABLED = True
    install_component_refresh()
    _set_hook(scripts.ScriptRunner, "run", wrap(scripts.ScriptRunner.run, runner=True))
    _set_hook(processing, "process_images", wrap(processing.process_images))
    original = sd_models.list_models
    if not _owned_hook(original):
        @wraps(original)
        def listing(*args, **kwargs):
            result = original(*args, **kwargs)
            if ENABLED:
                register()
            return result
        setattr(listing, MARKER, True)
        _set_hook(sd_models, "list_models", listing)
    if not getattr(install, '_registered', False):
        script_callbacks.on_app_started(ready)
        script_callbacks.on_before_ui(reset_ui)
        script_callbacks.on_after_component(capture)
        script_callbacks.on_ui_tabs(bind_panels)
        script_callbacks.on_script_unloaded(uninstall)
        install._registered = True
    ready()
