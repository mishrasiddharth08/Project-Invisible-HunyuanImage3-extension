"""Sample total GPU usage and worker RAM; track PyTorch allocator separately."""
import threading
import time


class Measurement:
    def __init__(self, torch, interval=0.1):
        self.torch = torch
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None
        self.started = time.perf_counter()
        self.data = {'sample_interval_ms': int(interval * 1000)}
        self.process = self.nvml = self.handle = None
        self.cuda = getattr(torch, 'cuda', None)
        try:
            import psutil
            self.process = psutil.Process()
        except (ImportError, OSError):
            pass
        try:
            if not self.cuda or not self.cuda.is_available():
                return
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            device = self.cuda.current_device() if self.cuda and self.cuda.is_available() else 0
            if self.cuda and hasattr(self.cuda, 'get_device_properties'):
                props = self.cuda.get_device_properties(device)
                identifier = getattr(props, 'uuid', None)
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(identifier) if identifier else pynvml.nvmlDeviceGetHandleByIndex(device)
            else:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception:
            self.nvml = self.handle = None

    def sample(self):
        if self.process:
            try:
                value = self.process.memory_info().rss / 2**30
                self.data['process_rss_peak_gib'] = max(value, self.data.get('process_rss_peak_gib', 0))
            except Exception:
                pass
        try:
            if self.nvml and self.handle:
                used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20
            elif self.cuda and self.cuda.is_available() and hasattr(self.cuda, 'mem_get_info'):
                free, total = self.cuda.mem_get_info()
                used = (total - free) / 2**20
            else:
                return
            self.data.setdefault('gpu_baseline_mib', used)
            self.data['gpu_total_peak_mib'] = max(used, self.data.get('gpu_total_peak_mib', 0))
        except Exception:
            pass

    def begin(self):
        self.started = time.perf_counter()
        try:
            if self.cuda and self.cuda.is_available():
                self.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.sample()
        def poll():
            while not self.stop.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=poll, name='pi-hunyuan-memory', daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.sample()
        self.data['generation_seconds'] = time.perf_counter() - self.started
        if 'gpu_total_peak_mib' in self.data:
            self.data['gpu_usage_delta_peak_mib'] = max(0, self.data['gpu_total_peak_mib'] - self.data['gpu_baseline_mib'])
        try:
            if self.cuda and self.cuda.is_available():
                self.data['torch_allocated_peak_mib'] = self.cuda.max_memory_allocated() / 2**20
                self.data['torch_reserved_peak_mib'] = self.cuda.max_memory_reserved() / 2**20
        except Exception:
            pass
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return {key: round(value, 3) if isinstance(value, float) else value for key, value in self.data.items()}
