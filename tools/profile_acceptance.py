"""Simulate each card's software budget on the available 32 GiB GPU."""
import json
from pathlib import Path
import sys
import time
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi_hunyuan.transport import Worker

OUT = ROOT / 'work' / 'vram-profiles-2026-10-07'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = Path('G:/FORGE UI NEO/sd-webui-forge-classic/models/HunyuanImage3')
wait_for_idle = '--wait-for-idle' in sys.argv
cases = [value for value in sys.argv[1:] if not value.startswith('--')] or ['8','10','12','16','24','32']
report = json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').is_file() else []
for profile in cases:
    if wait_for_idle:
        import pynvml
        pynvml.nvmlInit()
        try:
            gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            deadline = time.monotonic() + 1800
            announced = False
            while pynvml.nvmlDeviceGetMemoryInfo(gpu).used > 3 * 2**30:
                if not announced:
                    print('WAITING for the other GPU job to release memory', flush=True)
                    announced = True
                if time.monotonic() > deadline:
                    raise RuntimeError('Shared GPU remained occupied for 30 minutes; no other job was stopped.')
                time.sleep(2)
        finally:
            pynvml.nvmlShutdown()
    worker = Worker(mode='speed', vram_profile=profile)
    try:
        worker.wait_ready(lambda: False)
        request = dict(paths={'model': str(MODELS/'hunyuan_image_3_instruct_distil_w4a8.safetensors'),
                              'vae': str(MODELS/'vae/hunyuan_image_3_vae_fp16.safetensors'), 'vision': None, 'head': None},
                       prompt='A realistic photographic portrait of an adult woman, natural skin texture, soft window light, simple white shirt, sharp eyes, neutral background.',
                       seed=710707, width=1024, height=1024, steps=8, cfg=1, guidance=2.5,
                       references=[], rewrite='off', spectrum=False, offload_after=True,
                       output=str(OUT/(profile+'gb.png')))
        print('START profile', profile, flush=True)
        started=time.perf_counter()
        event=worker.generate(request, lambda: False, lambda event: None)
        event['profile_test']=profile
        event['request_seconds']=round(time.perf_counter()-started,3)
        event['hardware_note']='Software-budget simulation on RTX 5090 32 GiB; not a physical smaller-card test.'
        with Image.open(request['output']) as image:
            assert image.size==(1024,1024)
        report=[item for item in report if item['profile_test']!=profile]+[event]
        (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('DONE profile',profile,event['request_seconds'],event.get('metrics'),flush=True)
    except RuntimeError as error:
        details = getattr(error, 'worker_event', {'message': str(error)})
        (OUT / ('error-' + profile + '.json')).write_text(json.dumps(details, indent=2), encoding='utf-8')
        print('PROFILE ERROR', profile, details, flush=True)
        raise
    finally:
        worker.close()
