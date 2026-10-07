"""Conditional controls inside Forge's existing image pages."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from modules import scripts
from pi_hunyuan import assets, forge, preset


class Script(scripts.Script):
    _pi_hunyuan3 = True

    def title(self):
        return 'Project Invisible HunyuanImage 3.0'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        inventory = assets.scan()
        mode = 'img2img' if is_img2img else 'txt2img'
        # Retain saved/API argument positions; selection belongs to Forge.
        files = [gr.Dropdown(choices=['Auto'] + list(inventory[role]), value='Auto',
                             label=label, visible=False, elem_id=f'pi_hunyuan3_{mode}_{role}')
                 for role, label in zip(('model', 'vae', 'vision', 'head'),
                                        ('Model', 'VAE', 'Vision encoder', 'Head'))]
        with gr.Accordion('HunyuanImage 3.0', open=False, visible=forge.active(),
                          elem_id=f'pi_hunyuan3_{mode}_panel') as panel:
            performance = gr.Dropdown(['balanced', 'speed', 'low-memory'], value='balanced', label='Speed / memory')
            profile = gr.Dropdown([('Automatic', 'auto')] + [(f'{size} GB', size) for size in ('8','10','12','16','24','32')], value='auto', label='VRAM profile')
            keep = gr.Checkbox(True, label='Keep model loaded for faster repeats')
            with gr.Accordion('Advanced', open=False):
                rewrite = gr.Radio(['off', 'rewrite only', 'think + rewrite'], value='off', label='Prompt rewrite')
                spectrum = gr.Checkbox(False, label='Spectrum')
                tiled = gr.Checkbox(False, label='Tiled VAE: lower decoding memory')
                unload = gr.Button('Unload model now', size='sm')
                unload.click(lambda: forge.runtime.release(), outputs=[], queue=False)
            with gr.Row(visible=is_img2img):
                reference2 = gr.Image(type='pil', label='Reference 2')
                reference3 = gr.Image(type='pil', label='Reference 3')
        forge.PANELS.append(panel)
        return files + [rewrite, spectrum, keep, reference2, reference3, performance, tiled, profile]


preset.install()
forge.install()
