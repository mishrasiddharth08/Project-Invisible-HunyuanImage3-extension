"""Check the real worker handshake without loading model weights."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_hunyuan.transport import Worker

worker = Worker()
try:
    worker.wait_ready(lambda: False)
    print('WORKER_READY: isolated backend imports and dynamic VRAM initialization passed; no model loaded.')
finally:
    worker.close()
