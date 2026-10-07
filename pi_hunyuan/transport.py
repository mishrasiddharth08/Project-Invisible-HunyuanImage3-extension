"""JSON worker protocol. Stop kills only this extension's owned process."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from .config import ROOT


class Worker:
    def __init__(self, mode='auto', vram_profile='auto'):
        from .profiles import normalize
        self.vram_profile = normalize(vram_profile)
        self.mode = mode
        self.events = queue.Queue()
        self.guard = threading.Lock()
        logs = ROOT / 'logs'
        logs.mkdir(exist_ok=True)
        self.log = open(logs / 'worker.log', 'w', encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'
        try:
            self.process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'pi_hunyuan' / 'worker.py'), '--mode', mode, '--vram-profile', self.vram_profile], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, cwd=str(ROOT), env=env, text=True, encoding='utf-8', bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            self.log.close()
            raise
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        self.events.put(event)
                except ValueError:
                    pass
        except (OSError, ValueError):
            pass
        finally:
            self.events.put({'event': 'eof'})

    def _next(self, cancelled, deadline=None):
        while True:
            if cancelled():
                self.close()
                raise InterruptedError('HunyuanImage 3 stopped.')
            if deadline and time.monotonic() > deadline:
                self.close()
                raise RuntimeError('Hunyuan worker startup timed out. See logs/worker.log.')
            try:
                event = self.events.get(timeout=0.15)
            except queue.Empty:
                continue
            if event.get('event') == 'eof':
                raise RuntimeError('Hunyuan worker exited. See logs/worker.log.')
            if event.get('event') == 'error':
                error = RuntimeError(str(event.get('message', 'Worker error')))
                error.worker_event = event
                raise error
            return event

    def wait_ready(self, cancelled):
        deadline = time.monotonic() + 180
        while self._next(cancelled, deadline).get('event') != 'ready':
            pass

    def generate(self, request, cancelled, progress):
        request = dict(request, id=uuid.uuid4().hex)
        with self.guard:
            if cancelled():
                raise InterruptedError('HunyuanImage 3 stopped.')
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=True) + '\n')
                self.process.stdin.flush()
            except (OSError, ValueError):
                if cancelled():
                    raise InterruptedError('HunyuanImage 3 stopped.') from None
                raise RuntimeError('Hunyuan worker closed before the request was sent.') from None
        while True:
            event = self._next(cancelled)
            if event.get('id') != request['id']:
                continue
            if event.get('event') == 'progress':
                progress(event)
            elif event.get('event') == 'result':
                if Path(event.get('output', '')).resolve() != Path(request['output']).resolve() or not Path(request['output']).is_file():
                    raise RuntimeError('Hunyuan worker did not return the requested image.')
                if cancelled():
                    raise InterruptedError('HunyuanImage 3 stopped.')
                return event

    def close(self):
        with self.guard:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.log.close()
