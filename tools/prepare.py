"""Release assembly and source-reference review. Run from the workspace."""
import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]


def bundle():
    revisions = {}
    for source, target, key in [('.hunyuan-upstream', 'hunyuan', 'hunyuan'), ('.hunyuan-comfy', 'ComfyUI', 'comfy')]:
        folder = BASE / source
        revisions[key] = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        destination = ROOT / 'vendor' / target
        destination.mkdir(parents=True, exist_ok=True)
        files = subprocess.check_output(['git', '-C', str(folder), 'ls-files'], text=True).splitlines()
        for item in files:
            if item.startswith(('assets/', 'tests/', 'tools/', 'workflows/', '.github/', 'docs/', 'web/')) and key == 'hunyuan':
                continue
            path = destination / item
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / item, path)
    (ROOT / 'vendor' / 'revisions.json').write_text(json.dumps(revisions, indent=2), encoding='utf-8')
    shutil.copy2(BASE / '.hunyuan-upstream' / 'LICENSE', ROOT / 'LICENSE')


def references():
    urls = {
        'reddit': 'https://www.reddit.com/r/StableDiffusion/comments/1wzclz5/hunyuanimage_30_80b_running_natively_in_comfyui/.json',
        'comparison': 'https://pedromarinhodev.github.io/ComfyUI-HunyuanImage3/',
        'user-repositories': 'https://api.github.com/users/mishrasiddharth08/repos?per_page=100',
        'forge-neo': 'https://api.github.com/repos/Haoming02/sd-webui-forge-classic/branches/neo',
    }
    for model in ('Instruct-Distil', 'Instruct', 'Base'):
        urls[model] = 'https://huggingface.co/api/models/PedroMarinhoDev/HunyuanImage-3.0-' + model + '-ComfyUI'
        urls[model + '-card'] = 'https://huggingface.co/PedroMarinhoDev/HunyuanImage-3.0-' + model + '-ComfyUI/raw/main/README.md'
    destination = ROOT / 'research'
    destination.mkdir(exist_ok=True)
    results = {}
    for key, url in urls.items():
        try:
            with urlopen(Request(url, headers={'User-Agent': 'ProjectInvisibleSourceReview/1.0'}), timeout=25) as response:
                data = response.read().decode('utf-8')
                results[key] = {'url': url, 'status': response.status, 'bytes': len(data)}
                (destination / (key + ('.json' if url.startswith(('https://api.', 'https://huggingface.co/api/')) or key == 'reddit' else '.txt'))).write_text(data, encoding='utf-8')
        except Exception as error:
            results[key] = {'url': url, 'error': str(error)}
        print(key, results[key].get('status', results[key].get('error')))
    (destination / 'reference-status.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import sys
    references() if '--references' in sys.argv else bundle()
