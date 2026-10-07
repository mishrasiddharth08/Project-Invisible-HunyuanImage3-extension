import math
import re


def validate(p, references, options):
    for label, value, low, high in [('Width', p.width, 256, 2048), ('Height', p.height, 256, 2048), ('Steps', p.steps, 1, 150), ('Batch size', p.batch_size, 1, 64), ('Batch count', p.n_iter, 1, 64)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or not low <= value <= high:
            raise ValueError(f'{label} must be a whole number from {low} to {high}.')
    if p.width % 16 or p.height % 16 or p.width * p.height > 1536 * 1536:
        raise ValueError('Use multiples of 16, up to about 2.3 megapixels. Start at 1024 × 1024.')
    if p.batch_size * p.n_iter > 64:
        raise ValueError('Choose at most 64 total images.')
    if str(p.sampler_name).lower() != 'euler' or str(getattr(p, 'scheduler', 'Simple')).lower() != 'simple':
        raise ValueError('HunyuanImage 3 requires Euler and Simple. Select those Forge controls.')
    if not math.isfinite(float(p.cfg_scale)) or not 0 <= float(p.cfg_scale) <= 100:
        raise ValueError('CFG must be between 0 and 100.')
    if getattr(p, 'enable_hr', False) or getattr(p, 'image_mask', None) is not None or getattr(p, 'restore_faces', False) or getattr(p, 'tiling', False):
        raise ValueError('Turn off Hires fix, inpainting, Restore faces and Tiling for HunyuanImage 3.')
    if getattr(p, 'subseed_strength', 0) or getattr(p, 'seed_resize_from_w', 0) > 0 or getattr(p, 'seed_resize_from_h', 0) > 0:
        raise ValueError('Set variation strength and seed resizing to zero.')
    if len(references) > 3:
        raise ValueError('Use at most three reference images.')
    if references and float(getattr(p, 'denoising_strength', 1)) != 1:
        raise ValueError('Set img2img Denoising strength to 1. Hunyuan uses image conditioning.')
    if options.get('rewrite', 'off') not in ('off', 'rewrite only', 'think + rewrite'):
        raise ValueError('Unknown prompt rewriting mode.')
    for prompt in ([p.prompt] if isinstance(p.prompt, str) else p.prompt):
        if re.search(r'<hypernet:', prompt, re.I):
            raise ValueError('Hypernetworks require a compatible native attention backend and cannot be applied here.')
    negatives = [p.negative_prompt] if isinstance(p.negative_prompt, str) else p.negative_prompt
    if any(str(value or '').strip() for value in negatives):
        raise ValueError('Clear Negative prompt. Hunyuan builds its own unconditional conditioning.')
