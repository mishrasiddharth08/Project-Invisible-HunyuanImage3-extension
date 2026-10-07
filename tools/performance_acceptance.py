"""Measured cold/warm, dense-adapter and tiled-VAE smoke using existing weights."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi_hunyuan.transport import Worker

OUT = ROOT / 'work' / 'performance-2026-10-07'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = Path('G:/FORGE UI NEO/sd-webui-forge-classic/models/HunyuanImage3')
paths = {'model': str(MODELS / 'hunyuan_image_3_instruct_distil_w4a8.safetensors'),
         'vae': str(MODELS / 'vae/hunyuan_image_3_vae_fp16.safetensors'), 'vision': None, 'head': None}


def fixtures():
    import torch
    from safetensors.torch import save_file
    generator = torch.Generator().manual_seed(710707)
    key = 'diffusion_model.final_layer.model.0.emb_layers.1'
    lora = OUT / 'synthetic-hunyuan-lora.safetensors'
    lokr = OUT / 'synthetic-hunyuan-lokr.safetensors'
    save_file({key+'.lora_down.weight': torch.randn(2, 4096, generator=generator) * 0.01,
               key+'.lora_up.weight': torch.randn(2048, 2, generator=generator) * 0.01}, str(lora))
    save_file({key+'.lokr_w1': torch.randn(32, 64, generator=generator) * 0.01,
               key+'.lokr_w2': torch.randn(64, 64, generator=generator) * 0.01}, str(lokr))
    return lora, lokr


def main():
    lora, lokr = fixtures()
    continuing = '--adapters-only' in sys.argv
    low_memory = '--low-memory' in sys.argv
    report_file = OUT / ('low-memory-report.json' if low_memory else 'report.json')
    worker = Worker(mode='low-memory' if low_memory else 'speed')
    report = json.loads(report_file.read_text()) if continuing and report_file.is_file() else []
    try:
        worker.wait_ready(lambda: False)
        base = dict(paths=paths, prompt='A realistic photographic portrait of an adult woman, natural skin texture, soft window light, simple white shirt, sharp eyes, neutral background.',
                    seed=710707, width=1024, height=1024, steps=8, cfg=1, guidance=2.5,
                    references=[], rewrite='off', spectrum=False, offload_after=False)
        cases = [('cold', {}), ('warm', {}),
                 ('lora', {'width': 256, 'height': 256, 'adapters': [{'path': str(lora), 'strength': 1.0}]}),
                 ('lokr', {'width': 256, 'height': 256, 'adapters': [{'path': str(lokr), 'strength': 1.0}]}),
                 ('tiled', {'vae_tiled': True, 'offload_after': True})]
        if continuing:
            cases = cases[2:]
        if low_memory:
            cases = [('low-memory', {'vae_tiled': True, 'offload_after': True})]
        for name, changes in cases:
            request = dict(base, **changes, output=str(OUT / (name+'.png')))
            print('START', name, flush=True)
            started = time.perf_counter()
            result = worker.generate(request, lambda: False, lambda event: None)
            result['case'] = name
            result['request_seconds'] = round(time.perf_counter()-started, 3)
            report.append(result)
            report_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print('DONE', name, result['request_seconds'], result.get('metrics'), flush=True)
        with Image.open(OUT/'cold.png') as a, Image.open(OUT/'warm.png') as b:
            x, y = np.array(a), np.array(b)
            consistency = {'same_pixels': bool(np.array_equal(x,y)), 'mean_absolute_pixel_error': float(np.abs(x.astype(float)-y).mean())}
        (OUT/'consistency.json').write_text(json.dumps(consistency, indent=2), encoding='utf-8')
        print('CONSISTENCY', consistency, flush=True)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
