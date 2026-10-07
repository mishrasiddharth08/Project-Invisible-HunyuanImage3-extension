"""Strict local adapters. Returns (cloned Comfy patcher, audit); never loads CLIP.

Dense weights use Comfy's adapter registry and canonical load_lora keymap.
Expert banks use reversible patcher injections and output factors, never bank
weight patches. Individual bank LoHa uses cross-rank factors. Other compatible
forms use one selected expert matrix, with a 256 MiB workspace ceiling.
"""
import importlib
import math
import re
import sys
import uuid
from pathlib import Path


class AdapterError(ValueError):
    pass


_TAG = re.compile(r'<(?:lora|lyco):([^<>]+)>', re.I)
_EXT = {'.safetensors', '.pt', '.pth', '.ckpt', '.bin'}
MAX_EXPERT_MATRIX_BYTES = 256 * 1024 * 1024
_SUFFIX = re.compile(
    r'(?P<suffix>\.(?:lora_(?:up|down|mid)\.weight|lora_[AB](?:\.default)?(?:\.weight)?|'
    r'lora\.(?:up|down)\.weight|lora_linear_layer\.(?:up|down)\.weight|'
    r'lokr_(?:w[12](?:_[ab])?|t2)|hada_(?:w[12]_[ab]|t[12])|'
    r'oft_blocks|rescale|alpha|dora_scale|lora_magnitude_vector(?:\.default)?(?:\.weight)?|'
    r'reshape_weight)|_lora\.(?:up|down)\.weight)$')


def _strength(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError('Adapter strength must be a finite number.') from exc
    if not math.isfinite(value):
        raise AdapterError('Adapter strength must be a finite number.')
    return value


def parse_prompt(prompt):
    """Remove Forge lora/lyco tags; preserve all other prompt characters."""
    tags = []
    def extract(match):
        body = match[1]
        if ':' not in body:
            raise AdapterError('Adapter tag needs name:strength.')
        name, value = body.rsplit(':', 1)
        if not name.strip():
            raise AdapterError('Adapter name is empty.')
        tags.append((name.strip(), _strength(value)))
        return ''
    clean = _TAG.sub(extract, prompt)
    if re.search(r'<(?:lora|lyco):', clean, re.I):
        raise AdapterError('Malformed adapter tag.')
    return clean, tags


def resolve_adapters(tags, roots):
    """Resolve exact relative names or unique stems inside existing local roots."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    inventory = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in _EXT:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                continue
            rel = path.relative_to(root)
            for alias in (rel.as_posix(), rel.with_suffix('').as_posix(), path.name, path.stem):
                inventory.setdefault(alias.casefold(), set()).add(resolved)
    result = []
    for name, strength in tags:
        name = str(name).replace('\\', '/')
        if Path(name).is_absolute() or '..' in name.split('/') or ':' in name:
            raise AdapterError(f'Adapter name must be local: {name}')
        matches = inventory.get(name.casefold(), set())
        if len(matches) != 1:
            reason = 'ambiguous' if matches else 'not found in local models/Lora roots'
            raise AdapterError(f'Adapter {name}: {reason}')
        result.append(dict(path=str(next(iter(matches))), strength=_strength(strength)))
    return result


def _aliases(key):
    stem = key.removesuffix('.weight')
    short = stem.removeprefix('diffusion_model.')
    names = {stem, short}
    if short.startswith('model.'):
        names.add(short[6:])
    for name in tuple(names):
        names.update(('base_model.model.' + name, 'lora_unet_' + name.replace('.', '_'),
                      'lycoris_' + name.replace('.', '_'), name.replace('.', '_')))
    return names


def _targets(model):
    targets, aliases = {}, {}
    modules = dict(model.named_modules())
    def register(target, module, shape, bank, logical):
        targets[target] = (module, shape, bank)
        for alias in _aliases(logical):
            aliases.setdefault(alias, set()).add(target)
    def halves(key, expert, module, shape, bank, logical):
        if shape[0] % 2:
            raise AdapterError(f'{key}: invalid fused gate/up output dimension')
        width = shape[0] // 2
        # Tencent uses x1 * silu(x2): standard up is first, gate second.
        for projection, start in (('up_proj', 0), ('gate_proj', width)):
            rows = tuple(range(start, start + width))
            register((key, expert, rows), module, (width, shape[1]), bank,
                     logical.replace('gate_and_up_proj', projection))
    for name, module in modules.items():
        weight = getattr(module, 'weight', None)
        if weight is None:
            continue
        key = name + '.weight' if name else 'weight'
        bank = callable(getattr(module, 'expert_linear', None))
        mapped_shape = getattr(module, '_gguf_logical_shape', None)
        shape = tuple(mapped_shape) if mapped_shape is not None else tuple(weight.shape)
        if bank:
            dimensions = tuple(getattr(module, attr, None) for attr in ('num_experts', 'out_features', 'in_features'))
            if all(isinstance(d, int) and d > 0 for d in dimensions):
                shape = dimensions
            elif len(shape) != 3:
                raise AdapterError(f'{key}: flat expert bank requires expert/out/input metadata')
        elif mapped_shape is None and all(isinstance(getattr(module, attr, None), int) and getattr(module, attr) > 0
                 for attr in ('out_features', 'in_features')):
            shape = (module.out_features, module.in_features)
        # Quantized wrappers or plain packed payloads can report storage shapes.
        # The loaded operations' dimensions are the authoritative geometry.
        register((key, None), module, shape, bank, key)
        if bank:
            for i in range(shape[0]):
                target = (key, i)
                logical = key.replace('experts_gate_up_proj', f'experts.{i}.gate_and_up_proj')
                logical = logical.replace('experts_down_proj', f'experts.{i}.down_proj')
                if logical == key:
                    continue
                register(target, module, shape[1:], True, logical)
                if 'experts_gate_up_proj' in key:
                    halves(key, i, module, shape[1:], True, logical)
        elif key.endswith('gate_and_up_proj.weight') and len(shape) == 2:
            halves(key, None, module, shape, False, key)
        elif key.endswith('qkv_proj.weight') and len(shape) == 2:
            parent = modules[name.rsplit('.', 1)[0]]
            dims = [getattr(parent, attr, None) for attr in
                    ('num_key_value_heads', 'num_key_value_groups', 'head_dim')]
            if not all(isinstance(d, int) and d > 0 for d in dims):
                continue  # never guess fused QKV ordering from total shape
            kv, groups, head = dims
            stride = (groups + 2) * head
            if shape[0] != kv * stride:
                raise AdapterError(f'{key}: fused QKV shape disagrees with attention metadata')
            for projection, offset, width in (('q_proj', 0, groups * head),
                                               ('k_proj', groups * head, head),
                                               ('v_proj', (groups + 1) * head, head)):
                rows = tuple(row for i in range(kv) for row in
                             range(i * stride + offset, i * stride + offset + width))
                register((key, None, rows), module, (len(rows), shape[1]), False,
                         key.replace('qkv_proj', projection))
    return targets, aliases


def _normalise(state, aliases):
    groups = {}
    for key, tensor in state.items():
        match = _SUFFIX.search(key)
        if match is None:
            raise AdapterError(f'Unsupported or unmatched adapter key: {key}')
        prefix, suffix = key[:match.start()], match['suffix']
        candidates = aliases.get(prefix, set())
        if len(candidates) != 1:
            raise AdapterError(f'Adapter target missing or ambiguous: {key} (external CLIP is unsupported)')
        target = next(iter(candidates))
        group = groups.setdefault(target, {})
        # Convert PEFT DoRA to the bundled registry spelling.
        if suffix.startswith('.lora_magnitude_vector'):
            suffix = '.dora_scale'
        if suffix.startswith('_lora.'):
            suffix = '.lora_' + ('up' if '.up.' in suffix else 'down') + '.weight'
        if suffix in group:
            raise AdapterError(f'Duplicate adapter key for {target}: {key}')
        group[suffix] = tensor
    if not groups:
        raise AdapterError('Adapter has no weights.')
    return groups


def _load_group(group, registry):
    import torch
    for suffix, value in group.items():
        if not isinstance(value, torch.Tensor):
            raise AdapterError(f'Non-tensor adapter value: {suffix}')
        if not torch.isfinite(value).all().item():
            raise AdapterError(f'Non-finite adapter value: {suffix}')
        if value.numel() == 0:
            raise AdapterError(f'Empty adapter factor: {suffix}')
    alpha = group.get('.alpha')
    if alpha is not None and alpha.numel() != 1:
        raise AdapterError('Adapter alpha must be scalar.')
    alpha = alpha.item() if alpha is not None else None
    # Bundled BOFT compares alpha > 0 without a None guard.
    if '.oft_blocks' in group and alpha is None:
        alpha = 0.0
    state = {'target' + k: v for k, v in group.items()}
    used = {'target.alpha'} if '.alpha' in group else set()
    if '.dora_scale' in group:
        used.add('target.dora_scale')
    found = []
    try:
        for cls in registry.adapters:
            if cls.name not in {'lora', 'lokr', 'loha', 'oft', 'boft'}:
                continue
            adapter = cls.load('target', state, alpha, group.get('.dora_scale'), set())
            if adapter is not None:
                found.append(adapter)
                used.update(adapter.loaded_keys)
    except (KeyError, IndexError, TypeError, RuntimeError) as exc:
        raise AdapterError(f'Missing or invalid adapter factors: {exc}') from exc
    if len(found) != 1 or used != set(state):
        raise AdapterError(f'Incomplete, mixed or unsupported adapter factors: {sorted(set(state) - used)}')
    return found[0]


def _validate(adapter, shape):
    """Validate logical geometry using allocation-free meta tensor operations."""
    import torch
    def meta(t):
        return None if t is None else torch.empty(tuple(t.shape), device='meta')
    v = [meta(t) if isinstance(t, torch.Tensor) else t for t in adapter.weights]
    name = adapter.name
    try:
        if name == 'lora':
            up, down, _, mid, dora, reshape = v
            if reshape is not None and tuple(adapter.weights[5]) != shape:
                raise AdapterError('Adapter resizing is unsupported.')
            if mid is not None:
                down = (down.transpose(0, 1).flatten(1) @ mid.transpose(0, 1).flatten(1)).reshape(
                    down.shape[1], down.shape[0], *mid.shape[2:]).transpose(0, 1)
            diff = up.flatten(1) @ down.flatten(1)
            if diff.shape[0] != shape[0] or diff.shape[1] != math.prod(shape[1:]):
                raise AdapterError('LoRA output/input shape mismatch')
        elif name == 'lokr':
            w1, w2, _, a, b, c, d, t, dora = v
            w1 = w1 if w1 is not None else a @ b
            w2 = w2 if w2 is not None else (c @ d if t is None else
                torch.einsum('i j k l, j r, i p -> p r k l', t, d, c))
            if w1.ndim != 2 or w2.ndim not in (2, 4):
                raise AdapterError('Unsupported LoKr factor geometry')
            expected = (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1], *w2.shape[2:])
            if expected != shape and not (len(shape) == 2 and expected[0] == shape[0]
                                          and math.prod(expected[1:]) == shape[1]):
                raise AdapterError('LoKr shape mismatch')
        elif name == 'loha':
            a, b, _, c, d, t1, t2, dora = v
            if t1 is None:
                m1, m2 = a @ b, c @ d
            else:
                m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, b, a)
                m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, d, c)
            if m1.shape != m2.shape or m1.shape[0] != shape[0] or m1.numel() != math.prod(shape):
                raise AdapterError('LoHa shape mismatch')
        else:
            blocks, rescale, _, dora = v
            if blocks.shape[-1] != blocks.shape[-2] or blocks.shape[-3] * blocks.shape[-1] != shape[0]:
                raise AdapterError('OFT/BOFT shape mismatch')
            if name == 'boft' and (blocks.shape[-1] % 2 or any(
                    shape[0] % (2**i * blocks.shape[-1]) for i in range(blocks.shape[0]))):
                raise AdapterError('BOFT butterfly shape mismatch')
            if rescale is not None and torch.broadcast_shapes(tuple(rescale.shape), shape) != shape:
                raise AdapterError('OFT rescale shape mismatch')
        if dora is not None:
            norm_shape = ((shape[0], *([1] * (len(shape) - 1))) if dora.shape[0] == shape[0]
                          else (1, shape[1], *([1] * (len(shape) - 2))))
            if torch.broadcast_shapes(tuple(dora.shape), norm_shape) != norm_shape:
                raise AdapterError('DoRA magnitude shape mismatch')
    except (RuntimeError, ValueError, IndexError, AttributeError) as exc:
        raise AdapterError(f'{name} shape validation failed for {shape}: {exc}') from exc


def _factor_output(adapter, x, strength):
    """2D additive factors; never materialize an expert's full weight delta."""
    import torch
    import torch.nn.functional as F
    def cast(t):
        return None if t is None else t.to(device=x.device, dtype=x.dtype)
    v = [cast(t) if hasattr(t, 'to') else t for t in adapter.weights]
    if adapter.name == 'lora':
        up, down, alpha, _, _, _ = v
        scale = 1.0 if alpha is None else alpha / down.shape[0]
        return F.linear(F.linear(x, down), up) * (strength * scale)
    if adapter.name == 'loha':
        a, b, alpha, c, d, _, _, _ = v
        scale = 1.0 if alpha is None else alpha / b.shape[0]
        hidden = torch.einsum('...i,ri,si->...rs', x, b, d)
        return torch.einsum('...rs,or,os->...o', hidden, a, c) * (strength * scale)
    w1, w2, alpha, a, b, c, d, _, _ = v
    rank = None
    if w1 is None:
        w1, rank = a @ b, b.shape[0]
    if w2 is None:
        w2, rank = c @ d, d.shape[0]
    scale = alpha / rank if alpha is not None and rank is not None else 1.0
    grouped = x.reshape(*x.shape[:-1], w1.shape[1], w2.shape[1])
    out = F.linear(F.linear(grouped, w2).transpose(-1, -2), w1).transpose(-1, -2)
    return out.flatten(-2) * (strength * scale)


def _needs_matrix(adapter):
    v = adapter.weights
    if adapter.name == 'lora':
        return v[3] is not None or v[4] is not None or v[5] is not None or any(t.ndim != 2 for t in v[:2])
    if adapter.name == 'lokr':
        return v[7] is not None or v[8] is not None or (v[1] is not None and v[1].ndim != 2)
    if adapter.name == 'loha':
        return v[5] is not None or v[6] is not None or v[7] is not None
    return True


def _matrix_budget(shape, adapters, dtype, extra_bytes=0):
    import torch
    size = torch.empty((), dtype=dtype).element_size()
    # Reserve full matrix buffers: base, working copy and Comfy intermediates.
    # Also account for adapter casts and FP32 orthogonal inverse workspaces.
    factors = sum(t.numel() * max(size, t.element_size()) for adapter in adapters
                  for t in adapter.weights if isinstance(t, torch.Tensor))
    orthogonal = sum(adapter.weights[0].numel() * 4 * 6 for adapter in adapters
                     if adapter.name in {'oft', 'boft'})
    buffers = 6 if any(adapter.name == 'loha' for adapter in adapters) else 5
    required = math.prod(shape) * size * buffers + factors + orthogonal + extra_bytes
    if required > MAX_EXPERT_MATRIX_BYTES:
        raise AdapterError(f'Selected-expert matrix workspace {required / 2**20:.1f} MiB exceeds '
                           f'{MAX_EXPERT_MATRIX_BYTES / 2**20:g} MiB budget')
    return required


def _selected_matrix(module, expert, x, shape, dtype):
    """Slice before moving/dequantizing. Never call a whole-bank conversion."""
    import torch
    weight = module.weight
    if getattr(module, '_gguf_logical_shape', None) is not None:
        fetch = getattr(module, 'expert_weight', None)
        if expert is not None and callable(fetch):
            weight = fetch(expert)
        elif callable(getattr(module, '_value', None)):
            # Native mapped ops use store.matrix(key, expert) inside _value.
            # Passing a scalar dtype/device marker avoids casting an activation.
            marker = torch.empty((), device=x.device, dtype=dtype)
            weight = module._value('weight', marker, expert=expert)
        else:
            raise AdapterError('GGUF selected matrix API unavailable; refusing marker/bank conversion')
    elif expert is not None:
        fetch = getattr(module, '_expert_qt_from', None)
        if callable(fetch) and getattr(module, 'quant_format', None) is not None:
            weight = fetch(weight, expert)
        elif len(weight.shape) == 3:
            weight = weight[expert]
        elif len(weight.shape) == 2 and weight.is_floating_point() and not hasattr(weight, '_qdata'):
            weight = weight.view(module.num_experts, module.out_features, module.in_features)[expert]
        else:
            raise AdapterError('Selected-expert slice unavailable for this quantized bank/store')
    if tuple(weight.shape) != shape:
        raise AdapterError(f'Selected-expert logical shape mismatch: {tuple(weight.shape)} versus {shape}')
    weight = weight.to(device=x.device)
    if hasattr(weight, '_qdata') or (type(weight) is not torch.Tensor and callable(getattr(weight, 'dequantize', None))):
        weight = weight.dequantize()
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise AdapterError('Selected-expert float conversion unavailable; raw packed weights cannot be patched')
    return weight.to(dtype=dtype).detach()


def _calculate_matrix(adapter, weight, strength):
    """Promote Comfy's swallowed calculation errors to explicit failures."""
    import logging
    import threading
    import torch
    if adapter.name == 'lokr' and adapter.weights[7] is not None:
        # Bundled Tucker einsum can produce noncontiguous factors that torch.kron
        # rejects. Rebuild only its small factors, retaining Comfy's final math.
        v = adapter.weights
        def cast(t):
            return t.to(device=weight.device, dtype=weight.dtype)
        w1 = cast(v[0]) if v[0] is not None else cast(v[3]) @ cast(v[4])
        w2 = torch.einsum('i j k l, j r, i p -> p r k l', cast(v[7]), cast(v[6]), cast(v[5]))
        scale = 1.0 if v[2] is None else v[2] / v[6].shape[0]
        adapter = type(adapter)(set(), (w1.contiguous(), (w2 * scale).contiguous(), None,
                                        None, None, None, None, None, v[8]))
    key = 'pi_hunyuan.selected_expert'
    thread = threading.get_ident()
    class FailOnComfyError(logging.Handler):
        def emit(self, record):
            if record.thread == thread and record.getMessage().startswith(f'ERROR {adapter.name} {key}'):
                raise AdapterError(record.getMessage())
    handler = FailOnComfyError(level=logging.ERROR)
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        result = adapter.calculate_weight(weight, key, strength, 1.0, None, lambda delta: delta,
                                          intermediate_dtype=weight.dtype)
        if tuple(result.shape) != tuple(weight.shape) or not torch.isfinite(result).all().item():
            raise AdapterError('Selected-expert patch returned invalid weights')
        return result
    except (RuntimeError, TypeError, IndexError) as exc:
        raise AdapterError(f'{adapter.name} selected-expert calculation failed: {exc}') from exc
    finally:
        logger.removeHandler(handler)


def _matrix_output(module, expert, updates, x):
    """Compose selected updates in file order, including nonlinear DoRA/OFT."""
    import torch
    import torch.nn.functional as F
    shape = (module.out_features, module.in_features) if hasattr(module, 'out_features') else (
        tuple(module.weight.shape[1:]) if expert is not None else tuple(module.weight.shape))
    dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
    size = torch.empty((), dtype=dtype).element_size()
    output_workspace = math.prod(x.shape[:-1]) * shape[0] * size * 2
    input_workspace = x.numel() * size if x.dtype != dtype else 0
    _matrix_budget(shape, [adapter for _, adapter, _, _ in updates], dtype,
                   output_workspace + input_workspace)
    base = _selected_matrix(module, expert, x, shape, dtype)
    patched = base.clone()
    for _, adapter, strength, rows in updates:
        if strength == 0:
            continue
        if rows is None:
            patched = _calculate_matrix(adapter, patched, strength)
        else:
            indices = torch.tensor(rows, device=patched.device)
            selected = patched.index_select(0, indices)
            selected = _calculate_matrix(adapter, selected, strength)
            patched.index_copy_(0, indices, selected)
    patched.sub_(base)
    return F.linear(x.to(dtype), patched).to(x.dtype)


class _BankInjection:
    """Resolve modules on injection, including dynamic/deep clones; restore on eject."""
    def __init__(self, updates):
        self.updates = updates
        self.active = {}

    def inject(self, patcher):
        if id(patcher) in self.active:
            return
        modules = dict(patcher.model.named_modules())
        grouped = {}
        for key, expert, adapter, strength, rows in self.updates:
            grouped.setdefault(key.removesuffix('.weight'), []).append((expert, adapter, strength, rows))
        restores = []
        live = {}
        originals = {}
        try:
            for name, updates in grouped.items():
                module = modules[name]
                if getattr(module, '_pi_adapter_bank_owner', None) is not None:
                    raise AdapterError('Concurrent expert-bank adapter injections are unsupported.')
                def add(x, i, out, updates=updates, module=module):
                    i = None if i is None else int(i)
                    selected = [update for update in updates if update[0] == i and update[2] != 0]
                    if any(_needs_matrix(update[1]) for update in selected):
                        return out + _matrix_output(module, i, selected, x).to(out.dtype)
                    for expert, adapter, strength, rows in updates:
                        if expert == i and strength != 0:
                            delta = _factor_output(adapter, x, strength)
                            if rows is None:
                                out = out + delta
                            else:
                                import torch
                                out = out.clone()
                                out.index_add_(-1, torch.tensor(rows, device=out.device), delta.to(out.dtype))
                    return out
                if not callable(getattr(module, 'expert_linear', None)):
                    def hook(module, args, out, add=add):
                        return add(args[0], None, out)
                    handle = module.register_forward_hook(hook)
                    module._pi_adapter_bank_owner = self
                    def restore_dense(module=module, handle=handle):
                        handle.remove()
                        del module._pi_adapter_bank_owner
                    restores.append(restore_dense)
                    continue
                original = module.expert_linear
                originals[id(module)] = original
                had_method = 'expert_linear' in module.__dict__
                old_method = module.__dict__.get('expert_linear')
                def wrapped(x, i, original=original, add=add):
                    return add(x, i, original(x, i))
                module.expert_linear = wrapped
                module._pi_adapter_bank_owner = self
                live[id(module)] = add
                def restore(module=module, had=had_method, old=old_method):
                    if had:
                        module.expert_linear = old
                    else:
                        del module.expert_linear
                    del module._pi_adapter_bank_owner
                restores.append(restore)
            # Decode uses a free function and bypasses expert_linear for quantized
            # banks. Wrap the actual imported call site too, without triggering
            # weight_function / full-bank casting in the dynamic patcher.
            callsites = [m for m in tuple(sys.modules.values()) if m is not None
                         and getattr(m, '__name__', '').endswith('hunyuan_image_3.model')]
            if not callsites and any(hasattr(modules[name], 'quant_format') and
                                    callable(getattr(modules[name], 'expert_linear', None)) for name in grouped):
                raise AdapterError('Quantized expert bank: Hunyuan sliced decode call site unavailable; safe runtime patching unsupported.')
            for site in callsites:
                original = getattr(site, 'expert_linear_sliced', None)
                if original is None:
                    continue
                def sliced(module, x, i, original=original):
                    # Resident/nonquantized paths already call the wrapped method.
                    add = live.get(id(module))
                    if add is None:
                        return original(module, x, i)
                    # A read-only facade prevents double updates in fallback paths
                    # without changing the shared module during a forward call.
                    class View:
                        expert_linear = staticmethod(originals[id(module)])
                        def __getattr__(self, name):
                            return getattr(module, name)
                    out = original(View(), x, i)
                    return add(x, i, out)
                site.expert_linear_sliced = sliced
                restores.append(lambda site=site, original=original: setattr(site, 'expert_linear_sliced', original))
            self.active[id(patcher)] = restores
        except BaseException:
            for restore in reversed(restores):
                restore()
            raise

    def eject(self, patcher):
        for restore in reversed(self.active.pop(id(patcher), [])):
            restore()


def apply_adapters(model, adapters):
    """Return (clone, audit). Validate every tensor before registering any patch."""
    registry = importlib.import_module('comfy.weight_adapter')
    utils = importlib.import_module('comfy.utils')
    lora = importlib.import_module('comfy.lora')
    targets, aliases = _targets(model.model)
    pending, bank_updates, audit = [], [], []
    for spec in adapters:
        path = Path(spec['path'])
        if not path.is_file() or path.suffix.lower() not in _EXT:
            raise AdapterError(f'Adapter file does not exist or has unsupported type: {path}')
        strength = _strength(spec['strength'])
        state = utils.load_torch_file(str(path), safe_load=True)
        groups = _normalise(state, aliases)
        dense_state, keymap, formats, matched = {}, {}, set(), []
        matrix_count = 0
        for target, group in groups.items():
            key, expert = target[:2]
            rows = target[2] if len(target) == 3 else None
            module, shape, bank = targets[target]
            magnitude = group.get('.dora_scale')
            if magnitude is not None and magnitude.ndim == 1 and (not bank or expert is not None):
                if magnitude.shape[0] != shape[0]:
                    raise AdapterError(f'{key}: PEFT DoRA magnitude must match output dimension')
                group['.dora_scale'] = magnitude.reshape(shape[0], *([1] * (len(shape) - 1)))
            adapter = _load_group(group, registry)
            label = 'DoRA' if '.dora_scale' in group else ('LoCon' if '.lora_mid.weight' in group else adapter.name)
            formats.add(label)
            matched.append((key if expert is None else f'{key}[{expert}]') +
                           (f':rows={list(rows)}' if rows is not None else ''))
            if bank or rows is not None:
                if not callable(getattr(model, 'set_injections', None)):
                    raise AdapterError(f'{key}: patcher has no safe expert-bank injection hooks')
                if bank and key in getattr(model, 'patches', {}):
                    raise AdapterError(f'{key}: existing bank weight patches prevent safe factor updates')
                if bank and expert is None:
                    if adapter.name != 'lora' or any(adapter.weights[i] is not None for i in (3, 4, 5)):
                        raise AdapterError(f'{key}: stacked {label} unsupported; target individual experts')
                    up, down = adapter.weights[:2]
                    if up.ndim != 3 or down.ndim != 3 or up.shape[0] != shape[0] or down.shape[0] != shape[0]:
                        raise AdapterError(f'{key}: stacked bank LoRA needs [experts,out,rank] and [experts,rank,in]')
                    for i in range(shape[0]):
                        sliced = type(adapter)(set(), (up[i], down[i], *adapter.weights[2:]))
                        _validate(sliced, shape[1:])
                        bank_updates.append((key, i, sliced, strength, None))
                else:
                    _validate(adapter, shape)
                    if _needs_matrix(adapter):
                        matrix_count += 1
                        import torch
                        full_shape = ((module.out_features, module.in_features) if hasattr(module, 'out_features')
                                      else (targets[(key, None)][1][1:] if bank else targets[(key, None)][1]))
                        _matrix_budget(full_shape, [adapter], torch.bfloat16)
                    bank_updates.append((key, expert, adapter, strength, rows))
            else:
                _validate(adapter, shape)
                prefix = key.removesuffix('.weight')
                keymap[prefix] = key
                for suffix, tensor in group.items():
                    dense_state[prefix + suffix] = tensor
                if adapter.name == 'boft' and '.alpha' not in group:
                    import torch
                    dense_state[prefix + '.alpha'] = torch.tensor(0.0)
        patches = lora.load_lora(dense_state, keymap, log_missing=False) if dense_state else {}
        if set(patches) != set(keymap.values()):
            raise AdapterError(f'{path.name}: Comfy failed to load every canonical dense target')
        pending.append((patches, strength))
        audit.append(dict(name=path.stem, path=str(path), format='+'.join(sorted(formats)),
                          strength=strength, matched=len(groups), keys=matched,
                          missing=[], runtime_experts=sum(targets[t][2] for t in groups)))
        audit[-1]['matrix_fallbacks'] = matrix_count
        audit[-1]['matrix_budget_bytes'] = MAX_EXPERT_MATRIX_BYTES
    clone = model.clone()
    for patches, strength in pending:
        if patches and set(clone.add_patches(patches, strength)) != set(patches):
            raise AdapterError('Comfy patcher rejected adapter targets.')
    if bank_updates:
        extension = importlib.import_module('comfy.patcher_extension')
        for key, entries in list(getattr(clone, 'injections', {}).items()):
            if key.startswith('pi_hunyuan_adapters_'):
                for previous in entries:
                    bank_updates = previous.inject.__self__.updates + bank_updates
                clone.injections.pop(key)
        for key, expert, adapter, _, _ in bank_updates:
            if expert is None and _needs_matrix(adapter) and key in getattr(clone, 'patches', {}):
                raise AdapterError(f'{key}: matrix fallback cannot compose with dense patcher weight hooks')
        injection = _BankInjection(bank_updates)
        # Comfy's clone_has_same_weights compares injection keys, not values.
        # A fresh key forces bank-only adapter changes to switch active clones.
        clone.set_injections('pi_hunyuan_adapters_' + uuid.uuid4().hex,
                             [extension.PatcherInjection(injection.inject, injection.eject)])
    return clone, audit
