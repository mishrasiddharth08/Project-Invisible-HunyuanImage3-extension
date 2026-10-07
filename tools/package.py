"""Build a source ZIP or copy a fresh extension into the specified local Forge."""
import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.pytest_cache', 'logs', 'research', 'work'}


def package():
    destination = ROOT.parent / 'Project-Invisible-HunyuanImage3-Forge-Neo.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(item in SKIP or item.startswith('.hunyuan-') or '-backups-' in item for item in relative.parts) or relative.parts[:2] == ('vendor', 'python') or path.suffix.lower() in {'.safetensors', '.gguf', '.ckpt', '.pt', '.pth'}:
                continue
            archive.write(path, str(Path(ROOT.name) / relative))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.sha256').write_text(digest + '  ' + destination.name + '\n', encoding='utf-8')
    print(destination)
    print('SHA256', digest)


def deploy():
    forge = Path('G:/FORGE UI NEO/sd-webui-forge-classic')
    destination = forge / 'extensions' / ROOT.name
    if not (forge / 'modules_forge' / 'main_entry.py').is_file():
        raise RuntimeError('The specified Forge Neo installation is missing.')
    if destination.exists():
        raise RuntimeError('Destination already exists; refusing to replace an existing installation.')
    shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns(*SKIP))
    print('INSTALLED', destination)


if __name__ == '__main__':
    deploy() if '--deploy' in sys.argv else package()
