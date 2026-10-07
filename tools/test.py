"""Use a writable test temp location on sandboxed Windows."""
from pathlib import Path
import sys
import site
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
packages = root / 'vendor' / 'python'
if packages.is_dir():
    site.addsitedir(str(packages))
    sys.path.insert(0, str(packages))
folder = root / 'logs' / 'test-temp'
folder.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(folder)
result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(root / 'tests')))
raise SystemExit(0 if result.wasSuccessful() else 1)
