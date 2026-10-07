from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from pi_hunyuan import runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = SimpleNamespace(interrupted=False, skipped=False, nextjob=Mock())
        self.saved = []
        self.workers = []
        self.p = SimpleNamespace(width=256, height=256, steps=8, cfg_scale=1, sampler_name='Euler', scheduler='Simple', batch_size=1, n_iter=2, seed=42, prompt='a fox', negative_prompt='', styles=[], extra_generation_params={}, outpath_samples=str(self.root))
        processing = SimpleNamespace(StableDiffusionProcessingImg2Img=type('Img2Img', (), {}), fix_seed=Mock(), Processed=lambda p, output, **kw: SimpleNamespace(images=output, **kw))
        shared = SimpleNamespace(state=self.state, opts=SimpleNamespace(samples_save=True, samples_format='png'))
        def save(image, folder, _, seed, prompt, **kw):
            path = self.root / f'{seed}.png'
            image.save(path)
            self.saved.append((seed, kw['info']))
            return str(path), None
        parent = ModuleType('modules')
        parent.shared, parent.processing, parent.images = shared, processing, SimpleNamespace(save_image=save)
        backend = ModuleType('backend')
        backend.memory_management = SimpleNamespace(unload_all_models=Mock())
        self.addCleanup(patch.stopall)
        patch.dict('sys.modules', {'modules': parent, 'backend': backend}).start()
        patch.object(runtime, 'scan', return_value={role: [] for role in ('model', 'vae', 'vision', 'head')}).start()
        patch.object(runtime, 'resolve', side_effect=lambda value, role, inv: str(self.root / f'hunyuan_{role}.safetensors')).start()
        owner = self
        class Worker:
            def __init__(self, mode='auto', vram_profile='auto'):
                self.mode = mode
                self.vram_profile = vram_profile
                self.closed = False
                self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
                owner.workers.append(self)
            def wait_ready(self, cancelled):
                if getattr(owner, 'stop_on_start', False):
                    owner.state.interrupted = True
                    raise InterruptedError()
            def generate(self, request, cancelled, progress):
                progress({'step': 8, 'total': 8})
                if getattr(owner, 'stop_on_generate', False):
                    owner.state.interrupted = True
                    raise InterruptedError()
                Image.new('RGB', (request['width'], request['height'])).save(request['output'])
                if getattr(owner, 'stop_on_result', False):
                    owner.state.interrupted = True
                return {'output': request['output'], 'model_type': 'instruct_distil', 'steps': 8, 'cfg': 1, 'rewritten_prompt': ''}
            def close(self):
                self.closed = True
        patch.object(runtime, 'Worker', Worker).start()
        runtime.release()
        self.addCleanup(runtime.release)

    def test_saved_batch_seeds_and_gallery(self):
        result = runtime.generate(self.p, {})
        self.assertEqual(result.all_seeds, [42, 43])
        self.assertEqual(len(result.images), 2)
        self.assertEqual([seed for seed, _ in self.saved], [42, 43])
        self.assertIn('Model: hunyuan_model', result.infotexts[0])
        self.assertIn('Steps: 8', result.images[0].info['parameters'])
        self.assertTrue(all(worker.closed for worker in self.workers))

    def test_stop_during_start_or_sampling_returns_empty_result(self):
        for where in ('stop_on_start', 'stop_on_generate'):
            with self.subTest(where=where):
                self.state.interrupted = False
                setattr(self, where, True)
                result = runtime.generate(self.p, {'keep_loaded': True})
                self.assertEqual(result.images, [])
                self.assertEqual(result.all_seeds, [])
                self.assertIsNone(runtime._worker)
                setattr(self, where, False)

    def test_keep_loaded_then_release(self):
        runtime.generate(self.p, {'keep_loaded': True})
        worker = runtime._worker
        self.assertIsNotNone(worker)
        self.assertFalse(worker.closed)
        runtime.release()
        self.assertTrue(worker.closed)

    def test_late_stop_does_not_save_result(self):
        self.stop_on_result = True
        result = runtime.generate(self.p, {})
        self.assertEqual(result.images, [])
        self.assertEqual(self.saved, [])

    def test_low_memory_releases_cache_even_when_keep_is_requested(self):
        runtime.generate(self.p, {'keep_loaded': True, 'performance': 'low-memory'})
        self.assertIsNone(runtime._worker)
        self.assertEqual(self.workers[0].mode, 'low-memory')

    def test_idle_cache_releases_under_ram_pressure(self):
        timers = []
        class Timer:
            def __init__(self, seconds, callback):
                self.callback, self.cancelled = callback, False
                timers.append(self)
            def start(self):
                pass
            def cancel(self):
                self.cancelled = True
        with patch.object(runtime.threading, 'Timer', Timer), patch('psutil.virtual_memory', return_value=SimpleNamespace(available=0)):
            runtime.generate(self.p, {'keep_loaded': True})
            self.assertIsNotNone(runtime._worker)
            timers[-1].callback()
            self.assertIsNone(runtime._worker)
            self.assertTrue(all(worker.closed for worker in self.workers))
