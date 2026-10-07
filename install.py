"""Install worker-only packages, preserving Forge's torch and environment."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'vendor' / 'python'
# Pinned to the bundled Comfy revision. No torch/torchvision replacement.
PACKAGES = ('comfy-kitchen==0.2.37', 'comfy-aimdo==0.5.5', 'blake3==1.0.8', 'simpleeval==1.0.3', 'pydantic-settings==2.10.1', 'python-dotenv==1.1.1', 'gguf==0.19.0')


def install():
    if not (ROOT / 'vendor' / 'ComfyUI' / 'comfy' / 'ops.py').is_file() or not (ROOT / 'vendor' / 'hunyuan' / 'hunyuan_image_3' / 'tencent' / 'tokenizer.json').is_file():
        raise RuntimeError('Incomplete extension: download the complete release, including vendor/.')
    TARGET.mkdir(parents=True, exist_ok=True)
    for requirement in PACKAGES:
        name, version = requirement.split('==')
        found = list(importlib.metadata.distributions(path=[str(TARGET)]))
        if any(d.metadata['Name'].lower().replace('_', '-') == name and d.version == version for d in found):
            continue
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps', '--upgrade', '--target', str(TARGET), requirement])
    print('[PI-Hunyuan3] Worker packages ready; Forge packages unchanged. Models are downloaded separately.')


if __name__ == '__main__':
    install()
