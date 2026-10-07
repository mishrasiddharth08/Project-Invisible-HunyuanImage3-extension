"""Three real Forge API portrait runs; no model downloads or settings writes."""
import base64
import io
import json
from pathlib import Path
import time
import requests
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work' / 'human-quality-2026-10-07'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'http://127.0.0.1:7860'
deadline = time.monotonic() + 240
while True:
    try:
        check = requests.get(URL + '/sdapi/v1/progress', params={'skip_current_image': 'true'}, timeout=3)
        if check.status_code == 200:
            if check.json().get('state', {}).get('job_count', 0) > 0:
                raise RuntimeError('Forge is busy; refusing to interrupt another job.')
            break
    except requests.RequestException:
        if time.monotonic() > deadline:
            raise RuntimeError('Forge API did not become ready. Check forge-startup.log.')
        time.sleep(2)

cases = [
    ('01-face', 1024, 1024, 710701, 'A candid photographic portrait of an adult woman in her thirties, head and shoulders, looking toward the camera, realistic human face, subtle natural asymmetry, brown eyes in crisp focus, visible natural skin texture and fine facial hair, softly lit by a large window, plain cream shirt, neutral grey background, 85mm portrait photograph, natural color, no retouching.'),
    ('02-hands', 1024, 1536, 710702, 'A realistic editorial photograph of an adult man in his thirties sitting at a wooden cafe table outdoors, shown from his head to his waist, both hands resting separately on the table with all fingers clearly visible and relaxed, natural human anatomy and proportions, simple navy cotton shirt, believable skin texture, soft overcast daylight, sharp face and hands, calm expression, documentary photography, natural color.'),
    ('03-full-body', 1024, 1536, 710703, 'A realistic full length photograph of an adult woman in her thirties standing naturally in a quiet park, her entire body visible from the top of her head to both shoes with clear space above and below, arms relaxed at her sides with both hands visible, anatomically correct natural human proportions, simple light blue shirt and beige trousers, neutral flat shoes, realistic facial features, soft morning daylight, documentary fashion photography, natural color, sharp detail.'),
]
report = []
for name, width, height, seed, prompt in cases:
    print('START', name, f'{width}x{height}', flush=True)
    payload = dict(prompt=prompt, negative_prompt='', width=width, height=height, steps=8,
                   cfg_scale=1, distilled_cfg_scale=2.5, seed=seed, sampler_name='Euler',
                   scheduler='Simple', batch_size=1, n_iter=1, enable_hr=False,
                   save_images=True, send_images=True,
                   override_settings={'forge_preset': 'hunyuan3',
                                      'sd_model_checkpoint': 'PROJECT INVISIBLE — HunyuanImage 3.0',
                                      'forge_additional_modules': []},
                   override_settings_restore_afterwards=True,
                   alwayson_scripts={'Project Invisible HunyuanImage 3.0':
                                     {'args': ['Auto', 'Auto', 'Auto', 'Auto', 'off', False, False, None, None]}})
    started = time.monotonic()
    response = requests.post(URL + '/sdapi/v1/txt2img', json=payload, timeout=1800)
    if response.status_code != 200:
        (OUT / (name + '-error.txt')).write_text(response.text, encoding='utf-8')
        raise RuntimeError(f'{name}: HTTP {response.status_code}: {response.text[:800]}')
    data = response.json()
    if len(data.get('images', [])) != 1:
        raise RuntimeError(f'{name}: expected one image, received {len(data.get("images", []))}')
    raw = base64.b64decode(data['images'][0].split(',', 1)[-1])
    path = OUT / (name + '.png')
    path.write_bytes(raw)
    info = json.loads(data.get('info') or '{}')
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (width, height):
            raise RuntimeError(f'{name}: unexpected dimensions {image.size}')
        stats = ImageStat.Stat(image.convert('RGB'))
        if max(stats.stddev) < 2:
            raise RuntimeError(f'{name}: image is nearly flat')
        params = image.info.get('parameters', '') or (info.get('infotexts') or [''])[0]
    item = {'name': name, 'path': str(path), 'seconds': round(time.monotonic() - started, 2),
            'size': [width, height], 'seed': seed, 'mean': stats.mean, 'stddev': stats.stddev,
            'parameters': params, 'api_info': info, 'visual_quality': 'pending visual inspection'}
    report.append(item)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('DONE', name, item['seconds'], str(path), flush=True)
print('PORTRAITS_GENERATED', len(report), flush=True)
