from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import package


class PackageTests(unittest.TestCase):
    def test_zip_and_install_have_identical_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp) / 'extension'
            included = ['pi_hunyuan/runtime.py', 'tools/package.py', 'vendor/ComfyUI/main.py',
                        'resources/marker.json', 'README.md']
            excluded = ['.git/config', '__pycache__/a.pyc', '.pytest_cache/state', 'logs/a.log',
                        'research/a.txt', 'work/a.txt', '.hunyuan-test/a.txt',
                        'fix-backups-123/a.py', 'vendor/python/lib/a.py',
                        'weights/a.safetensors', 'weights/b.GGUF', 'weights/c.ckpt',
                        'weights/d.pt', 'weights/e.pth', 'nested/logs/a.txt']
            for name in included + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            installed = Path(temp) / 'installed'
            with patch.object(package, 'ROOT', root), patch('builtins.print'):
                package.package()
                shutil.copytree(root, installed, ignore=package.install_ignore)
            archive_path = Path(temp) / 'Project-Invisible-HunyuanImage3-Forge-Neo.zip'
            with zipfile.ZipFile(archive_path) as archive:
                zipped = {Path(name).relative_to(root.name).as_posix() for name in archive.namelist()}
            copied = {path.relative_to(installed).as_posix() for path in installed.rglob('*') if path.is_file()}
            self.assertEqual(zipped, set(included))
            self.assertEqual(copied, zipped)
            self.assertTrue(archive_path.with_suffix('.sha256').is_file())

    def test_deploy_uses_shared_filter(self):
        with patch.object(Path, 'is_file', return_value=True), patch.object(Path, 'exists', return_value=False), patch.object(package.shutil, 'copytree') as copy, patch('builtins.print'):
            package.deploy()
        self.assertIs(copy.call_args.kwargs['ignore'], package.install_ignore)
