import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from pi_hunyuan.assets import classify, header, resolve, companion
from pi_hunyuan.request import validate


def fixture(path, keys):
    values = {key: {'dtype': 'F16', 'shape': [1], 'data_offsets': [n * 2, n * 2 + 2]} for n, key in enumerate(keys)}
    data = json.dumps(values).encode()
    path.write_bytes(struct.pack('<Q', len(data)) + data + b'\0\0' * len(keys))
    return str(path)


class AssetsTests(unittest.TestCase):
    def test_roles_and_truncation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            cases = {'model': ('hunyuan_image_3_instruct_distil_w4a8.safetensors', ['model.layers.0.input_layernorm.weight', 'model.wte.weight']), 'head': ('hunyuan_image_3_instruct_cot_head.safetensors', ['lm_head.weight']), 'vae': ('hunyuan_image_3_vae_fp16.safetensors', ['encoder.conv.weight', 'decoder.conv.weight']), 'vision': ('hunyuan_image_3_instruct_siglip2.safetensors', ['vision_model.embeddings.weight'])}
            for role, (name, keys) in cases.items():
                path = Path(fixture(root / name, keys))
                self.assertEqual(classify(path), role)
                self.assertEqual(resolve(str(path), role, {role: []}), str(path.resolve()))
                path.write_bytes(path.read_bytes()[:-1])
                self.assertIsNone(classify(path))

    def test_invalid_header(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / 'bad.safetensors'
            path.write_bytes(struct.pack('<Q', 2**40))
            with self.assertRaises(ValueError):
                header(path)
            with self.assertRaises(ValueError):
                resolve('Auto', 'model', {'model': []})

    def test_matching_companions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            instruct = fixture(root / 'hunyuan_image_3_instruct_siglip2.safetensors', ['vision_model.embeddings.weight'])
            distil = fixture(root / 'hunyuan_image_3_instruct_distil_siglip2.safetensors', ['vision_model.embeddings.weight'])
            inventory = {'vision': [distil, instruct]}
            self.assertEqual(companion('Auto', 'vision', 'hunyuan_image_3_instruct.safetensors', inventory, True), instruct)
            with self.assertRaises(ValueError):
                companion(distil, 'vision', 'hunyuan_image_3_instruct.safetensors', inventory, True)
            base_head = fixture(root / 'hunyuan_image_3_base_cot_head.safetensors', ['lm_head.weight'])
            with self.assertRaises(ValueError):
                companion(base_head, 'head', 'hunyuan_image_3_instruct.safetensors', {'head': [base_head]}, True)


class RequestTests(unittest.TestCase):
    def request(self, **changes):
        fields = dict(width=1024, height=1024, steps=8, batch_size=1, n_iter=1, sampler_name='Euler', scheduler='Simple', cfg_scale=1, prompt='a fox', negative_prompt='')
        fields.update(changes)
        return SimpleNamespace(**fields)

    def test_supported_and_rejections(self):
        validate(self.request(), [], {})
        for changes in ({'width': 1000}, {'width': 2048, 'height': 2048}, {'steps': float('nan')}, {'negative_prompt': 'bad'}, {'prompt': '<hypernet:test:1>'}, {'sampler_name': 'Euler a'}, {'enable_hr': True}, {'batch_size': 64, 'n_iter': 2}, {'cfg_scale': float('inf')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(self.request(**changes), [], {})
        with self.assertRaises(ValueError):
            validate(self.request(denoising_strength=0.5), [object()], {})


if __name__ == '__main__':
    unittest.main()
