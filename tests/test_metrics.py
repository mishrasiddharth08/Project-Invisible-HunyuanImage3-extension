from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from pi_hunyuan.metrics import Measurement


class MetricsTests(unittest.TestCase):
    def test_sampled_total_gpu_and_torch_peaks_are_distinct(self):
        cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=Mock(),
                               mem_get_info=Mock(side_effect=[(80*2**20, 100*2**20), (50*2**20, 100*2**20)]),
                               max_memory_allocated=lambda: 12*2**20, max_memory_reserved=lambda: 16*2**20)
        with patch.dict('sys.modules', {'pynvml': None}):
            measurement = Measurement(SimpleNamespace(cuda=cuda), interval=10)
        measurement.process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=3*2**30))
        measurement.begin()
        result = measurement.finish()
        self.assertEqual(result['gpu_total_peak_mib'], 50)
        self.assertEqual(result['gpu_usage_delta_peak_mib'], 30)
        self.assertEqual(result['torch_allocated_peak_mib'], 12)
        self.assertEqual(result['torch_reserved_peak_mib'], 16)
        self.assertEqual(result['process_rss_peak_gib'], 3)
        self.assertFalse(measurement.thread.is_alive())

    def test_no_cuda_does_not_invent_gpu_usage(self):
        measurement = Measurement(SimpleNamespace()).begin()
        result = measurement.finish()
        self.assertNotIn('gpu_total_peak_mib', result)
        self.assertIn('generation_seconds', result)
