from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'PROJECT INVISIBLE — HunyuanImage 3.0'
PRESET = 'hunyuan3'


def settings():
    path = ROOT / 'config.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def models_root():
    try:
        from modules.paths import models_path
        return Path(models_path)
    except ImportError:
        return ROOT.parent.parent / 'models'
