"""Line-JSON worker; never import Forge or start a Comfy server."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description="Isolated HunyuanImage-3 worker")
    parser.add_argument("--mode", choices=("auto", "dynamic", "legacy", "speed", "low-memory"), default="auto")
    parser.add_argument('--vram-profile', choices=('auto','8','10','12','16','24','32'), default='auto')
    args = parser.parse_args()
    # Reserve a private protocol descriptor, then redirect fd 1 as well as
    # sys.stdout: Python prints and native-library output both go to parent logs.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)

    def emit(event):
        protocol.write(json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n")
        protocol.flush()

    def error(request_id, exc):
        traceback.print_exc(file=sys.stderr)
        emit({"event": "error", "id": request_id,
              "error": str(exc), "message": str(exc), "type": type(exc).__name__,
              "metrics": getattr(exc, 'hunyuan_metrics', {})})

    try:
        # Direct script launch and package launch are both supported. backend.py
        # itself imports only stdlib; it sets vendor paths before heavy imports.
        if __package__:
            from .backend import HunyuanBackend
        else:
            from backend import HunyuanBackend
        backend = HunyuanBackend(mode=args.mode, vram_profile=args.vram_profile)
    except Exception as exc:
        error(None, exc)
        protocol.close()
        return 1
    emit({"event": "ready", "id": None})
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                request_id = request.get("id")
                emit(backend.generate(request, emit))
            except Exception as exc:
                error(request_id, exc)
    finally:
        protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
