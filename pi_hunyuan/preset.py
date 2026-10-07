"""Add a separate preset using Forge's own settings template."""
from copy import copy, deepcopy

from .config import LABEL, PRESET


def install():
    from modules import shared
    from modules_forge import presets
    template = {}
    presets.register(template)
    ours = {}
    defaults = {'forge_checkpoint_' + PRESET: LABEL,
                'forge_additional_modules_' + PRESET: []}
    for mode in ('t2i', 'i2i'):
        for suffix, value in {'sampler': 'Euler', 'scheduler': 'Simple', 'step': 8,
                              'cfg': 1.0, 'dcfg': 2.5, 'width': 1024, 'height': 1024}.items():
            defaults[f'{PRESET}_{mode}_{suffix}'] = value
    for suffix, value in {'step': 8, 'cfg': 1.0, 'dcfg': 2.5}.items():
        defaults[f'{PRESET}_t2i_hr_{suffix}'] = value
    for key, info in template.items():
        if key.startswith('sd_'):
            target = PRESET + key[2:]
        elif key in ('forge_checkpoint_sd', 'forge_additional_modules_sd', 'forge_unet_storage_dtype_sd'):
            target = key[:-2] + PRESET
        else:
            continue
        item = copy(info)
        item.default = deepcopy(defaults.get(target, info.default))
        if getattr(info, 'section', (None,))[0] == 'ui_sd':
            item.section = ('ui_hunyuan3', 'HUNYUANIMAGE 3.0')
        ours[target] = item
    required = {'forge_checkpoint_' + PRESET, PRESET + '_t2i_step', PRESET + '_t2i_cfg'}
    if not required.issubset(ours):
        raise RuntimeError('Forge preset settings changed; Hunyuan preset unavailable.')
    # SD's template has no distilled guidance settings; clone its numeric controls.
    for mode in ('t2i', 't2i_hr', 'i2i'):
        source = ours.get(f'{PRESET}_{mode}_cfg', ours[f'{PRESET}_t2i_cfg'])
        item = copy(source)
        item.default = 2.5
        item.label = 'Hunyuan distilled guidance'
        ours[f'{PRESET}_{mode}_dcfg'] = item
    item = copy(ours[f'{PRESET}_i2i_cfg'] if f'{PRESET}_i2i_cfg' in ours else ours[f'{PRESET}_t2i_cfg'])
    item.default = 1.0
    item.label = 'Hunyuan img2img denoising strength'
    item.component_args = {'minimum': 0, 'maximum': 1, 'step': 0.01}
    ours[f'{PRESET}_i2i_denoising_strength'] = item
    for key, info in ours.items():
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
    original = presets.PresetArch.choices
    if not getattr(original, '_pi_hunyuan3', False):
        def choices(*args, **kwargs):
            values = [item for item in original(*args, **kwargs)
                      if item != PRESET and not (isinstance(item, (tuple, list)) and len(item) == 2 and item[1] == PRESET)]
            values.append(('HunyuanImage 3.0', PRESET))
            return values
        choices._pi_hunyuan3 = True
        presets.PresetArch.choices = staticmethod(choices)
