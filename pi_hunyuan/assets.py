"""Bounded safetensors inspection; never load a tensor in Forge's process."""
import json
import struct
from functools import lru_cache
from pathlib import Path
from .config import models_root, settings


@lru_cache(maxsize=256)
def _header(path, size, mtime):
    with open(path, 'rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Incomplete model file: ' + path)
        length = struct.unpack('<Q', raw)[0]
        if not 2 <= length <= min(32 * 1024 * 1024, size - 8):
            raise ValueError('Invalid safetensors header: ' + path)
        value = json.loads(stream.read(length))
        if not isinstance(value, dict):
            raise ValueError('Invalid model header: ' + path)
        for key, item in value.items():
            if key == '__metadata__':
                continue
            if not isinstance(item, dict):
                raise ValueError('Invalid tensor header: ' + path)
            offsets = item.get('data_offsets', [])
            if len(offsets) != 2 or not all(isinstance(v, int) for v in offsets) or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length:
                raise ValueError('Incomplete tensor data: ' + path)
        return value


def header(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _header(str(path), stat.st_size, stat.st_mtime_ns)


def classify(path):
    path = Path(path)
    if path.suffix.lower() == '.gguf':
        name = path.name.lower().replace('-', '_')
        if 'hunyuan' not in name or not ('image_3' in name or 'image3' in name):
            return None
        try:
            with path.open('rb') as stream:
                raw = stream.read(8)
            # Discovery only. The isolated loader validates the real tensor
            # family and dimensions before any model is built.
            return 'model' if len(raw) == 8 and raw[:4] == b'GGUF' and struct.unpack('<I', raw[4:])[0] in (2, 3) else None
        except OSError:
            return None
    if path.suffix.lower() != '.safetensors':
        return None
    name = path.name.lower().replace('-', '_')
    # Inspect only likely files during recursive scanning; exact selectors are
    # validated again before a worker is allowed to allocate memory.
    if 'hunyuan' not in name:
        return None
    try:
        keys = header(path)
    except (OSError, ValueError, struct.error):
        return None
    if 'model.layers.0.input_layernorm.weight' in keys and 'model.wte.weight' in keys:
        return 'model'
    if 'lm_head.weight' in keys:
        return 'head'
    if any('vision_model' in key for key in keys) and 'siglip' in name:
        return 'vision'
    if 'vae' in name and any('decoder.' in key for key in keys) and any('encoder.' in key for key in keys):
        return 'vae'
    return None


def scan():
    inventory = {key: [] for key in ('model', 'vae', 'vision', 'head')}
    base = models_root()
    roots = [base / name for name in ('HunyuanImage3', 'HunyuanImage-3.0', 'Stable-diffusion', 'diffusion_models', 'VAE', 'vae', 'clip_vision')]
    roots += [Path(path).expanduser() for path in settings().get('model_roots', [])]
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        paths = list(root.rglob('*.safetensors')) + list(root.rglob('*.gguf'))
        for path in paths:
            resolved = str(path.resolve())
            if resolved.casefold() in seen:
                continue
            seen.add(resolved.casefold())
            role = classify(path)
            if role:
                inventory[role].append(resolved)
    for role in inventory:
        inventory[role].sort(key=lambda p: ('distil' not in p.lower(), 'w4a8' not in p.lower(), p.casefold()))
    return inventory


def resolve(value, role, inventory, required=True):
    candidates = inventory[role]
    if value in (None, '', 'Auto', '(none)'):
        if candidates:
            return candidates[0]
        if not required:
            return None
        raise ValueError(f'Missing HunyuanImage 3 {role}. Add the converted file, then Refresh local files.')
    path = Path(value).expanduser().resolve()
    if not path.is_file() or classify(path) != role:
        raise ValueError(f'Choose a valid HunyuanImage 3 {role} file: {value}')
    return str(path)


def variant(path):
    name = Path(path).name.lower()
    return 'instruct_distil' if 'distil' in name else 'instruct' if 'instruct' in name else 'base' if 'base' in name else None


def companion(value, role, model, inventory, required):
    kind = variant(model)
    candidates = inventory[role]
    if value in (None, '', 'Auto', '(none)'):
        candidates = [path for path in candidates if variant(path) == kind or role == 'head' and kind in ('instruct', 'instruct_distil') and variant(path) in ('instruct', 'instruct_distil')]
    selected = resolve(value, role, {role: candidates}, required)
    if selected and role == 'vision' and kind and variant(selected) != kind:
        raise ValueError('Select the vision encoder matching the chosen Hunyuan model variant.')
    if selected and role == 'head' and kind in ('instruct', 'instruct_distil') and variant(selected) not in ('instruct', 'instruct_distil'):
        raise ValueError('Instruct and Instruct-Distil require an Instruct rewriting head.')
    return selected
