"""Numerical tests use the bundled Comfy adapter implementations on CPU.

Only Comfy's hardware services and patcher lifecycle are simulated; adapter
loading and weight math come from vendor sources, not test reimplementations.
"""
import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from pi_hunyuan import adapters as api

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / 'vendor/ComfyUI/comfy'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Bank(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 30,
                                         requires_grad=False)
        self.bias = None
        self.quant_format = 'test-packed'

    def expert_linear(self, x, i):
        return F.linear(x, self.weight[int(i)])


class SliceOnlyQuantized(torch.Tensor):
    """Sentinel forbids dequantizing a whole bank, allows expert slices only."""
    def dequantize(self):
        if self.ndim != 2:
            raise AssertionError('Whole-bank dequantization forbidden')
        return self.as_subclass(torch.Tensor)


def fixture_model():
    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Module()
    d = root.diffusion_model
    d.model = torch.nn.Module()
    d.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = d.model.layers[0]
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
    layer.mlp = torch.nn.Module()
    layer.mlp.experts_down_proj = Bank()
    d.model.embed_tokens = torch.nn.Embedding(4, 4)
    d.conv = torch.nn.Conv2d(2, 4, 3, bias=False)
    return root


class Patcher:
    def __init__(self, model):
        self.model, self.patches, self.injections = model, {}, {}

    def clone(self):
        result = Patcher(self.model)
        result.patches = {k: list(v) for k, v in self.patches.items()}
        result.injections = self.injections.copy()
        return result

    def add_patches(self, patches, strength):
        keys = set(self.model.state_dict())
        for key, adapter in patches.items():
            if key in keys:
                self.patches.setdefault(key, []).append((strength, adapter))
        return set(patches) & keys

    def set_injections(self, key, injections):
        self.injections[key] = injections

    def get_injections(self, key):
        return self.injections.get(key, [])

    def inject_model(self):
        for entries in self.injections.values():
            for entry in entries:
                entry.inject(self)

    def eject_model(self):
        for entries in self.injections.values():
            for entry in entries:
                entry.eject(self)

    def effective(self, key):
        w = self.model.state_dict()[key].clone()
        for strength, adapter in self.patches.get(key, []):
            w = adapter.calculate_weight(w, key, strength, 1.0, None, lambda x: x)
        return w


DENSE = 'diffusion_model.model.layers.0.self_attn.q_proj.weight'
BANK = 'diffusion_model.model.layers.0.mlp.experts_down_proj.weight'
EXPERT = 'model.layers.0.mlp.experts.1.down_proj'


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = sys.modules.copy()
        self.addCleanup(self.restore_modules)
        comfy = types.ModuleType('comfy')
        comfy.__path__ = [str(COMFY)]
        sys.modules['comfy'] = comfy
        for name in ('memory_management', 'model_management', 'model_base', 'utils'):
            module = types.ModuleType('comfy.' + name)
            setattr(comfy, name, module)
            sys.modules[module.__name__] = module
        comfy.model_management.cast_to_device = lambda value, device, dtype, **kw: value.to(device=device, dtype=dtype)
        self.safe_load_calls = []
        def safe_load(path, safe_load=False):
            self.safe_load_calls.append(safe_load)
            if not safe_load:
                raise AssertionError('Unsafe load')
            return torch.load(path, weights_only=True)
        comfy.utils.load_torch_file = safe_load
        registry = types.ModuleType('comfy.weight_adapter')
        registry.__path__ = [str(COMFY / 'weight_adapter')]
        sys.modules[registry.__name__] = registry
        comfy.weight_adapter = registry
        load_source('comfy.weight_adapter.base', COMFY / 'weight_adapter/base.py')
        registry.adapters = []
        for filename, classname in [('lora', 'LoRAAdapter'), ('lokr', 'LoKrAdapter'),
                                    ('loha', 'LoHaAdapter'), ('oft', 'OFTAdapter'), ('boft', 'BOFTAdapter')]:
            module = load_source('comfy.weight_adapter.' + filename, COMFY / f'weight_adapter/{filename}.py')
            registry.adapters.append(getattr(module, classname))
        comfy.lora = load_source('comfy.lora', COMFY / 'lora.py')
        load_source('comfy.patcher_extension', COMFY / 'patcher_extension.py')
        self.site = types.ModuleType('test_vendor.hunyuan_image_3.model')
        # Both real decode cases: direct sliced projection and resident fallback.
        self.site.expert_linear_sliced = lambda module, x, i: F.linear(x, module.weight[int(i)])
        sys.modules[self.site.__name__] = self.site
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(11)
        self.base = Patcher(fixture_model())

    def restore_modules(self):
        for key in tuple(sys.modules):
            if key.startswith(('comfy', 'test_vendor.hunyuan_image_3')):
                sys.modules.pop(key, None)
        for key, value in self.saved_modules.items():
            if key.startswith(('comfy', 'test_vendor.hunyuan_image_3')):
                sys.modules[key] = value

    def apply(self, state, strength=0.5, model=None):
        path = self.root / 'my_adapter.pt'
        torch.save(state, path)
        return api.apply_adapters(model or self.base, [dict(path=str(path), strength=strength)])

    def lora(self, prefix, out=4, inp=4):
        return {prefix + '.lora_A.weight': torch.arange(2 * inp, dtype=torch.float32).reshape(2, inp) / 10,
                prefix + '.lora_B.weight': torch.arange(out * 2, dtype=torch.float32).reshape(out, 2) / 20,
                prefix + '.alpha': torch.tensor(4.)}

    def test_prompt_and_local_resolution(self):
        prompt = 'a <LoRA:sub/my_adapter:-.75> <lyco:other_name:1e-1>cat'
        clean, tags = api.parse_prompt(prompt)
        self.assertEqual(clean, 'a  cat')
        self.assertEqual(tags, [('sub/my_adapter', -.75), ('other_name', .1)])
        (self.root / 'sub').mkdir()
        (self.root / 'sub/my_adapter.safetensors').touch()
        (self.root / 'other_name.ckpt').touch()
        resolved = api.resolve_adapters(tags, [self.root])
        self.assertEqual(len(resolved), 2)
        self.assertTrue(resolved[0]['path'].endswith('my_adapter.safetensors'))
        for name in ('../my_adapter', 'missing', 'C:/outside'):
            with self.assertRaises(api.AdapterError):
                api.resolve_adapters([(name, 1)], [self.root])
        (self.root / 'my_adapter.pt').touch()
        with self.assertRaisesRegex(api.AdapterError, 'ambiguous'):
            api.resolve_adapters([('my_adapter', 1)], [self.root])

    def test_invalid_tags(self):
        for text in ('<lora:a:nan>', '<lyco:a:inf>', '<lora:a>', '<lora::1>', '<lora:a:1'):
            with self.subTest(text=text), self.assertRaises(api.AdapterError):
                api.parse_prompt(text)

    def test_dense_aliases_numerical_and_immutable(self):
        before = self.base.model.state_dict()[DENSE].clone()
        for prefix in ('model.layers.0.self_attn.q_proj', DENSE[:-7],
                       'base_model.model.model.layers.0.self_attn.q_proj',
                       'lora_unet_model_layers_0_self_attn_q_proj',
                       'lycoris_model_layers_0_self_attn_q_proj'):
            with self.subTest(prefix=prefix):
                state = self.lora(prefix)
                clone, audit = self.apply(state)
                expected = before + state[prefix + '.lora_B.weight'] @ state[prefix + '.lora_A.weight']
                torch.testing.assert_close(clone.effective(DENSE), expected)
                self.assertEqual(audit[0]['matched'], 1)
                self.assertEqual(audit[0]['missing'], [])
        self.assertEqual(self.base.patches, {})
        torch.testing.assert_close(self.base.model.state_dict()[DENSE], before)
        self.assertTrue(all(self.safe_load_calls))

    def test_dense_lokr_numerical(self):
        prefix = 'model.layers.0.self_attn.q_proj'
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        clone, audit = self.apply({prefix + '.lokr_w1': a, prefix + '.lokr_w2': b}, -.3)
        torch.testing.assert_close(clone.effective(DENSE), self.base.model.state_dict()[DENSE] - .3 * torch.kron(a, b))
        self.assertEqual(audit[0]['format'], 'lokr')

    def test_dense_remaining_registry_formats(self):
        prefix = 'model.layers.0.self_attn.q_proj'
        ones = torch.ones(4, 1)
        state = {prefix + '.hada_w1_a': ones, prefix + '.hada_w1_b': ones.T,
                 prefix + '.hada_w2_a': ones * 2, prefix + '.hada_w2_b': ones.T}
        clone, _ = self.apply(state)
        torch.testing.assert_close(clone.effective(DENSE), self.base.model.state_dict()[DENSE] + 1)
        for ndim in (3, 4):
            blocks = torch.zeros((2, 2, 2) if ndim == 3 else (1, 2, 2, 2))
            blocks[..., 0, 1] = .15
            clone, _ = self.apply({prefix + '.oft_blocks': blocks}, 1)
            changed = clone.effective(DENSE)
            self.assertFalse(torch.equal(changed, self.base.model.state_dict()[DENSE]))
            torch.testing.assert_close(changed.norm(), self.base.model.state_dict()[DENSE].norm())
            t = .15
            cosine, sine = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
            rotation = torch.tensor([[cosine, sine], [-sine, cosine]])
            # The bundled OFT uses R.T; one-stage BOFT uses R.
            matrix = rotation.T if ndim == 3 else rotation
            expected = torch.block_diag(matrix, matrix) @ self.base.model.state_dict()[DENSE]
            torch.testing.assert_close(changed, expected)

    def test_peft_dora_and_own_embedding(self):
        prefix = 'model.embed_tokens'
        state = self.lora('base_model.model.' + prefix)
        state['base_model.model.' + prefix + '.lora_magnitude_vector.default.weight'] = torch.ones(4)
        clone, audit = self.apply(state, 1)
        key = 'diffusion_model.model.embed_tokens.weight'
        up = state['base_model.model.' + prefix + '.lora_B.weight']
        down = state['base_model.model.' + prefix + '.lora_A.weight']
        base = self.base.model.state_dict()[key]
        expected = (base + 2 * (up @ down)) / (base.norm(dim=1, keepdim=True) + torch.finfo(base.dtype).eps)
        torch.testing.assert_close(clone.effective(key), expected)
        self.assertEqual(audit[0]['format'], 'DoRA')

    def test_locon_numerical(self):
        up, down, mid = torch.randn(4, 2, 1, 1), torch.randn(2, 2, 1, 1), torch.randn(2, 2, 3, 3)
        state = {'conv.lora_up.weight': up, 'conv.lora_down.weight': down, 'conv.lora_mid.weight': mid}
        clone, audit = self.apply(state)
        diff = torch.einsum('or,rsxy,si->oixy', up[:, :, 0, 0], mid, down[:, :, 0, 0])
        key = 'diffusion_model.conv.weight'
        torch.testing.assert_close(clone.effective(key), self.base.model.state_dict()[key] + .5 * diff)
        self.assertEqual(audit[0]['format'], 'LoCon')

    def test_bank_lora_both_call_paths_cleanup(self):
        state = self.lora(EXPERT)
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        before = module.weight.detach().clone()
        original_site = self.site.expert_linear_sliced
        clone, audit = self.apply(state)
        self.assertEqual(clone.patches, {})
        self.assertEqual(audit[0]['runtime_experts'], 1)
        x = torch.randn(3, 4)
        delta = F.linear(F.linear(x, state[EXPERT + '.lora_A.weight']), state[EXPERT + '.lora_B.weight'])
        with patch.object(torch, 'kron', side_effect=AssertionError('Full delta forbidden')):
            for _ in range(2):
                clone.inject_model()
                torch.testing.assert_close(module.expert_linear(x, 1), F.linear(x, before[1]) + delta)
                torch.testing.assert_close(self.site.expert_linear_sliced(module, x, 1), F.linear(x, before[1]) + delta)
                torch.testing.assert_close(module.expert_linear(x, 0), F.linear(x, before[0]))
                clone.eject_model()
                torch.testing.assert_close(module.expert_linear(x, 1), F.linear(x, before[1]))
        self.assertIs(self.site.expert_linear_sliced, original_site)
        self.assertNotIn('expert_linear', module.__dict__)
        torch.testing.assert_close(module.weight, before)

    def test_bank_resident_fallback_no_double_delta(self):
        self.site.expert_linear_sliced = lambda module, x, i: module.expert_linear(x, i)
        clone, _ = self.apply(self.lora(EXPERT))
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        clone.inject_model()
        try:
            x = torch.randn(1, 4)
            torch.testing.assert_close(self.site.expert_linear_sliced(module, x, 1), module.expert_linear(x, 1))
        finally:
            clone.eject_model()

    def test_bundled_quantized_decode_helper_never_dequantizes_bank(self):
        comfy = sys.modules['comfy']
        comfy.ops = types.ModuleType('comfy.ops')
        comfy.ops.disable_weight_init = type('Ops', (), {'Linear': torch.nn.Linear})
        sys.modules['comfy.ops'] = comfy.ops
        quant = types.ModuleType('comfy.quant_ops')
        quant.QUANT_ALGOS = {}
        quant.QuantizedTensor = SliceOnlyQuantized
        sys.modules[quant.__name__] = quant
        ops = load_source('test_vendor.hunyuan_image_3.ops', ROOT / 'vendor/hunyuan/hunyuan_image_3/ops.py')
        self.site.expert_linear_sliced = ops.expert_linear_sliced
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        before = module.weight.detach().clone()
        module.weight = torch.nn.Parameter(before.as_subclass(SliceOnlyQuantized), requires_grad=False)
        module._expert_qt_from = lambda bank, i: bank[int(i)]
        module._resident_bank = None
        module._full_precision_mm = True
        clone, _ = self.apply(self.lora(EXPERT))
        x = torch.randn(2, 4)
        baseline = ops.expert_linear_sliced(module, x, 1)
        clone.inject_model()
        try:
            state = self.lora(EXPERT)
            delta = F.linear(F.linear(x, state[EXPERT + '.lora_A.weight']), state[EXPERT + '.lora_B.weight'])
            torch.testing.assert_close(self.site.expert_linear_sliced(module, x, 1), baseline + delta)
            self.assertEqual(clone.patches, {})
            self.assertFalse(getattr(module, 'weight_function', None))
        finally:
            clone.eject_model()
        self.assertIsInstance(module.weight, SliceOnlyQuantized)
        torch.testing.assert_close(module.weight.as_subclass(torch.Tensor), before)

    def test_bank_lokr_factored_numerical(self):
        a, b, c = torch.randn(2, 1), torch.randn(1, 2), torch.randn(2, 2)
        state = {EXPERT + '.lokr_w1_a': a, EXPERT + '.lokr_w1_b': b,
                 EXPERT + '.lokr_w2': c, EXPERT + '.alpha': torch.tensor(2.)}
        clone, _ = self.apply(state, -.25)
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        x = torch.randn(2, 4)
        expected = module.expert_linear(x, 1) - .5 * F.linear(x, torch.kron(a @ b, c))
        with patch.object(torch, 'kron', side_effect=AssertionError('Full delta forbidden')):
            clone.inject_model()
            try:
                torch.testing.assert_close(module.expert_linear(x, 1), expected)
            finally:
                clone.eject_model()

    def test_bank_loha_cross_rank_numerical(self):
        a, b = torch.randn(4, 2), torch.randn(2, 4)
        c, d = torch.randn(4, 3), torch.randn(3, 4)
        state = {EXPERT + '.hada_w1_a': a, EXPERT + '.hada_w1_b': b,
                 EXPERT + '.hada_w2_a': c, EXPERT + '.hada_w2_b': d,
                 EXPERT + '.alpha': torch.tensor(4.)}
        clone, audit = self.apply(state, -.25)
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        x = torch.randn(2, 3, 4)
        baseline = module.expert_linear(x, 1)
        expected = baseline - .5 * F.linear(x, (a @ b) * (c @ d))
        clone.inject_model()
        try:
            torch.testing.assert_close(module.expert_linear(x, 1), expected)
            torch.testing.assert_close(self.site.expert_linear_sliced(module, x, 1), expected)
            self.assertEqual(audit[0]['format'], 'loha')
            self.assertEqual(clone.patches, {})
        finally:
            clone.eject_model()
        torch.testing.assert_close(module.expert_linear(x, 1), baseline)

    def test_stacked_bank_lora(self):
        prefix = BANK[:-7]
        up, down = torch.randn(2, 4, 1), torch.randn(2, 1, 4)
        clone, _ = self.apply({prefix + '.lora_up.weight': up, prefix + '.lora_down.weight': down})
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        before = module.weight.clone()
        clone.inject_model()
        try:
            x = torch.randn(2, 4)
            for i in range(2):
                torch.testing.assert_close(module.expert_linear(x, i), F.linear(x, before[i]) + .5 * F.linear(F.linear(x, down[i]), up[i]))
        finally:
            clone.eject_model()

    def test_bank_clone_and_sequential_apply(self):
        state = self.lora(EXPERT)
        first, _ = self.apply(state)
        second, _ = self.apply(state, model=first)
        self.assertNotEqual(set(first.injections), set(second.injections))
        self.assertEqual(len(second.injections), 1)
        second.model = copy.deepcopy(second.model)
        bank = second.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        x = torch.randn(1, 4)
        before = bank.expert_linear(x, 1)
        second.inject_model()
        try:
            delta = F.linear(F.linear(x, state[EXPERT + '.lora_A.weight']), state[EXPERT + '.lora_B.weight'])
            torch.testing.assert_close(bank.expert_linear(x, 1), before + 2 * delta)
            self.assertNotIn('_pi_adapter_bank_owner', self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj.__dict__)
        finally:
            second.eject_model()

    def test_strict_failure_no_partial_mutation(self):
        good = self.lora('model.layers.0.self_attn.q_proj')
        cases = [dict(good, **{'missing.lora_B.weight': torch.ones(4, 2)}),
                 {'model.layers.0.self_attn.q_proj.lora_B.weight': torch.ones(4, 2)},
                 self.lora('model.layers.0.self_attn.q_proj', out=3),
                 self.lora('lora_te_text_model_encoder_layers_0'),
                 {EXPERT + '.oft_blocks': torch.zeros(2, 3, 3)},
                 dict(self.lora(EXPERT), **{EXPERT + '.dora_scale': torch.ones(3, 1)}),
                 {'unknown.diff': torch.ones(4, 4)},
                 dict(good, **{'model.layers.0.self_attn.q_proj.alpha': torch.tensor(float('nan'))})]
        for state in cases:
            with self.subTest(keys=list(state)), self.assertRaises(api.AdapterError):
                self.apply(state)
        self.assertEqual(self.base.patches, {})
        self.assertEqual(self.base.injections, {})

    def test_missing_hooks_and_existing_bank_patches(self):
        self.base.set_injections = None
        with self.assertRaisesRegex(api.AdapterError, 'injection hooks'):
            self.apply(self.lora(EXPERT))
        self.base.set_injections = types.MethodType(Patcher.set_injections, self.base)
        self.base.patches[BANK] = [('bad', None)]
        with self.assertRaisesRegex(api.AdapterError, 'existing bank weight patches'):
            self.apply(self.lora(EXPERT))

    def test_packed_banks_do_not_abort_dense_adapters(self):
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        module.num_experts, module.out_features, module.in_features = 2, 4, 4
        module.weight = torch.nn.Parameter(torch.zeros(8, 2, dtype=torch.int8), requires_grad=False)
        clone, audit = self.apply(self.lora('model.layers.0.self_attn.q_proj'))
        self.assertIn(DENSE, clone.patches)
        targets, _ = api._targets(self.base.model)
        self.assertEqual(targets[(BANK, None)][1], (2, 4, 4))
        self.assertEqual(targets[(BANK, 1)][1], (4, 4))
        self.assertEqual(audit[0]['runtime_experts'], 0)

    def test_dense_packed_shape_uses_linear_dimensions(self):
        module = self.base.model.diffusion_model.model.layers[0].self_attn.q_proj
        module.weight = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.int8), requires_grad=False)
        targets, _ = api._targets(self.base.model)
        self.assertEqual(targets[(DENSE, None)][1], (4, 4))
        clone, audit = self.apply(self.lora('model.layers.0.self_attn.q_proj'))
        self.assertIn(DENSE, clone.patches)
        self.assertEqual(audit[0]['matched'], 1)

    def test_actual_quantized_tensor_logical_and_packed_shapes(self):
        qt_source = load_source('test_vendor.hunyuan_image_3.quant_tensor',
                                ROOT / 'vendor/python/comfy_kitchen/tensor/base.py')
        params = qt_source.BaseLayoutParams(scale=torch.ones(1), orig_dtype=torch.float32, orig_shape=(4, 4))
        tensor = qt_source.QuantizedTensor(torch.zeros(4, 2, dtype=torch.int8), 'shape_test', params)
        self.assertEqual(tuple(tensor.shape), (4, 4))
        self.assertEqual(tensor.storage_shape, (4, 2))
        module = torch.nn.Module()
        module.out_features, module.in_features, module.weight = 4, 4, tensor
        self.base.model.diffusion_model.model.layers[0].self_attn.q_proj = module
        targets, _ = api._targets(self.base.model)
        self.assertEqual(targets[(DENSE, None)][1], (4, 4))
        params = qt_source.BaseLayoutParams(scale=torch.ones(1), orig_dtype=torch.float32, orig_shape=(8, 4))
        bank = Bank()
        del bank.weight
        bank.num_experts, bank.out_features, bank.in_features = 2, 4, 4
        bank.weight = qt_source.QuantizedTensor(torch.zeros(8, 2, dtype=torch.int8), 'shape_test', params)
        self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj = bank
        targets, _ = api._targets(self.base.model)
        self.assertEqual(targets[(BANK, None)][1], (2, 4, 4))
        self.assertEqual(targets[(BANK, 1)][1], (4, 4))

    def test_qkv_separate_training_keys_interleaved_rows(self):
        attn = self.base.model.diffusion_model.model.layers[0].self_attn
        del attn.q_proj
        attn.num_key_value_heads, attn.num_key_value_groups, attn.head_dim = 2, 2, 1
        attn.qkv_proj = torch.nn.Linear(4, 8, bias=False)
        x = torch.randn(2, 3, 4)
        before = attn.qkv_proj(x).detach()
        weight_before = attn.qkv_proj.weight.detach().clone()
        for algorithm in ('lora', 'lokr'):
            for projection, rows in (('q_proj', [0, 1, 4, 5]), ('k_proj', [2, 6]), ('v_proj', [3, 7])):
                prefix = 'base_model.model.model.layers.0.self_attn.' + projection
                if algorithm == 'lora':
                    state = self.lora(prefix, out=len(rows))
                    diff = 2 * state[prefix + '.lora_B.weight'] @ state[prefix + '.lora_A.weight']
                else:
                    a, b = torch.randn(len(rows) // 2, 2), torch.randn(2, 2)
                    state = {prefix + '.lokr_w1': a, prefix + '.lokr_w2': b}
                    diff = torch.kron(a, b)
                clone, audit = self.apply(state, -.3)
                self.assertEqual(clone.patches, {})
                expected = before.clone()
                expected[..., rows] += -.3 * F.linear(x, diff)
                clone.inject_model()
                try:
                    actual = attn.qkv_proj(x)
                    torch.testing.assert_close(actual, expected)
                    untouched = [i for i in range(8) if i not in rows]
                    self.assertTrue(torch.equal(actual[..., untouched], before[..., untouched]))
                finally:
                    clone.eject_model()
                self.assertEqual(len(attn.qkv_proj._forward_hooks), 0)
                torch.testing.assert_close(attn.qkv_proj.weight, weight_before)
                self.assertEqual(audit[0]['matched'], 1)

    def test_expert_gate_up_training_keys_row_offsets(self):
        mlp = self.base.model.diffusion_model.model.layers[0].mlp
        mlp.experts_gate_up_proj = Bank()
        module = mlp.experts_gate_up_proj
        before_weight = module.weight.detach().clone()
        x = torch.randn(3, 4)
        before = module.expert_linear(x, 1)
        for algorithm in ('lora', 'lokr'):
            for projection, rows in (('up_proj', [0, 1]), ('gate_proj', [2, 3])):
                prefix = 'model.layers.0.mlp.experts.1.' + projection
                if algorithm == 'lora':
                    state = self.lora(prefix, out=2)
                    diff = 2 * state[prefix + '.lora_B.weight'] @ state[prefix + '.lora_A.weight']
                else:
                    a, b = torch.randn(1, 2), torch.randn(2, 2)
                    state = {prefix + '.lokr_w1': a, prefix + '.lokr_w2': b}
                    diff = torch.kron(a, b)
                clone, _ = self.apply(state)
                expected = before.clone()
                expected[..., rows] += .5 * F.linear(x, diff)
                clone.inject_model()
                try:
                    torch.testing.assert_close(module.expert_linear(x, 1), expected)
                    torch.testing.assert_close(self.site.expert_linear_sliced(module, x, 1), expected)
                    other_rows = [i for i in range(4) if i not in rows]
                    self.assertTrue(torch.equal(module.expert_linear(x, 1)[..., other_rows], before[..., other_rows]))
                    torch.testing.assert_close(module.expert_linear(x, 0), F.linear(x, before_weight[0]))
                finally:
                    clone.eject_model()
                torch.testing.assert_close(module.weight, before_weight)

    def test_shared_gate_up_training_keys_row_offsets(self):
        mlp = self.base.model.diffusion_model.model.layers[0].mlp
        mlp.shared_mlp = torch.nn.Module()
        mlp.shared_mlp.gate_and_up_proj = torch.nn.Linear(4, 4, bias=False)
        module = mlp.shared_mlp.gate_and_up_proj
        x = torch.randn(2, 4)
        before = module(x).detach()
        prefix = 'lycoris_model_layers_0_mlp_shared_mlp_gate_proj'
        state = self.lora(prefix, out=2)
        clone, _ = self.apply(state)
        clone.inject_model()
        try:
            actual = module(x)
            self.assertTrue(torch.equal(actual[..., :2], before[..., :2]))
            delta = F.linear(F.linear(x, state[prefix + '.lora_A.weight']), state[prefix + '.lora_B.weight'])
            torch.testing.assert_close(actual[..., 2:], before[..., 2:] + delta)
        finally:
            clone.eject_model()

    def test_qkv_aliases_require_verified_attention_metadata(self):
        attn = self.base.model.diffusion_model.model.layers[0].self_attn
        del attn.q_proj
        attn.qkv_proj = torch.nn.Linear(4, 8, bias=False)
        with self.assertRaisesRegex(api.AdapterError, 'target missing'):
            self.apply(self.lora('model.layers.0.self_attn.q_proj'))

    def advanced_cases(self, prefix, base):
        """Independent numerical references for the actual bundled algorithms."""
        out, inp = base.shape
        state = self.lora(prefix, out=out, inp=inp)
        magnitude = torch.linspace(.8, 1.2, out)
        state[prefix + '.lora_magnitude_vector.default.weight'] = magnitude
        up, down = state[prefix + '.lora_B.weight'], state[prefix + '.lora_A.weight']
        expected = (base + 2 * up @ down) * (magnitude[:, None] / (base.norm(dim=1, keepdim=True) + torch.finfo(base.dtype).eps))
        yield 'DoRA', state, expected
        t = .15
        rotation = torch.tensor([[(1 - t*t)/(1 + t*t), 2*t/(1 + t*t)],
                                 [-2*t/(1 + t*t), (1 - t*t)/(1 + t*t)]])
        for ndim, algorithm in ((3, 'OFT'), (4, 'BOFT')):
            blocks = torch.zeros((out // 2, 2, 2) if ndim == 3 else (1, out // 2, 2, 2))
            blocks[..., 0, 1] = t
            matrix = rotation.T if ndim == 3 else rotation
            yield algorithm, {prefix + '.oft_blocks': blocks}, torch.block_diag(*([matrix] * (out // 2))) @ base
        up, down, mid = torch.randn(out, 2, 1, 1), torch.randn(2, inp, 1, 1), torch.randn(2, 2, 1, 1)
        state = {prefix + '.lora_up.weight': up, prefix + '.lora_down.weight': down,
                 prefix + '.lora_mid.weight': mid}
        yield 'LoCon Tucker', state, base + up[:, :, 0, 0] @ mid[:, :, 0, 0] @ down[:, :, 0, 0]
        w1, a, b, t2 = torch.randn(out // 2, inp // 2), torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 1, 1, 1)
        state = {prefix + '.lokr_w1': w1, prefix + '.lokr_w2_a': a,
                 prefix + '.lokr_w2_b': b, prefix + '.lokr_t2': t2}
        yield 'LoKr Tucker', state, base + torch.kron(w1, a.T @ b * t2.item())
        a, b, c, d = (torch.randn(1, out), torch.randn(1, inp), torch.randn(1, out), torch.randn(1, inp))
        t1, t2 = torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)
        state = {prefix + '.hada_w1_a': a, prefix + '.hada_w1_b': b,
                 prefix + '.hada_w2_a': c, prefix + '.hada_w2_b': d,
                 prefix + '.hada_t1': t1, prefix + '.hada_t2': t2}
        yield 'LoHa Tucker', state, base + (a.T @ b * t1.item()) * (c.T @ d * t2.item())

    def test_selected_expert_advanced_variants_numerical_and_immutable(self):
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        original = module.weight.detach().clone()
        x = torch.randn(2, 4)
        for algorithm, state, expected in self.advanced_cases(EXPERT, original[1]):
            with self.subTest(algorithm=algorithm):
                clone, audit = self.apply(state, 1)
                self.assertEqual(audit[0]['matrix_fallbacks'], 1)
                self.assertEqual(audit[0]['matrix_budget_bytes'], 256 * 1024 * 1024)
                clone.inject_model()
                try:
                    torch.testing.assert_close(module.expert_linear(x, 1), F.linear(x, expected), atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(self.site.expert_linear_sliced(module, x, 1), F.linear(x, expected), atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(module.expert_linear(x, 0), F.linear(x, original[0]))
                finally:
                    clone.eject_model()
                torch.testing.assert_close(module.weight, original)
                self.assertEqual(clone.patches, {})

    def test_selected_expert_advanced_row_slices(self):
        mlp = self.base.model.diffusion_model.model.layers[0].mlp
        mlp.experts_gate_up_proj = Bank()
        module = mlp.experts_gate_up_proj
        original = module.weight.detach().clone()
        x = torch.randn(3, 4)
        before = module.expert_linear(x, 1)
        for projection, rows in (('up_proj', [0, 1]), ('gate_proj', [2, 3])):
            prefix = 'model.layers.0.mlp.experts.1.' + projection
            for algorithm, state, expected in self.advanced_cases(prefix, original[1, rows]):
                with self.subTest(projection=projection, algorithm=algorithm):
                    clone, _ = self.apply(state, 1)
                    clone.inject_model()
                    try:
                        output = module.expert_linear(x, 1)
                        torch.testing.assert_close(output[..., rows], F.linear(x, expected), atol=1e-5, rtol=1e-5)
                        untouched = [i for i in range(4) if i not in rows]
                        self.assertTrue(torch.equal(output[..., untouched], before[..., untouched]))
                    finally:
                        clone.eject_model()
                    torch.testing.assert_close(module.weight, original)

    def test_quantized_advanced_fallback_fetches_only_selected_expert(self):
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        original = module.weight.detach().clone()
        module.weight = torch.nn.Parameter(original.as_subclass(SliceOnlyQuantized), requires_grad=False)
        fetched = []
        def fetch(bank, i):
            self.assertIs(bank, module.weight)
            fetched.append(i)
            return bank[int(i)]
        module._expert_qt_from = fetch
        x = torch.randn(2, 4)
        for algorithm, state, expected in self.advanced_cases(EXPERT, original[1]):
            with self.subTest(algorithm=algorithm):
                fetched.clear()
                clone, _ = self.apply(state, 1)
                clone.inject_model()
                try:
                    torch.testing.assert_close(module.expert_linear(x, 1).as_subclass(torch.Tensor), F.linear(x, expected), atol=1e-5, rtol=1e-5)
                    self.assertEqual(fetched, [1])
                finally:
                    clone.eject_model()
                self.assertIsInstance(module.weight, SliceOnlyQuantized)
                torch.testing.assert_close(module.weight.as_subclass(torch.Tensor), original)

    def test_matrix_budget_fails_before_expert_fetch(self):
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        module.weight = torch.nn.Parameter(module.weight.detach().as_subclass(SliceOnlyQuantized), requires_grad=False)
        module._expert_qt_from = lambda *args: self.fail('Budget must reject before reading an expert')
        state = self.lora(EXPERT)
        state[EXPERT + '.dora_scale'] = torch.ones(4, 1)
        # BF16 preflight fits; runtime F32's larger workspace does not.
        with patch.object(api, 'MAX_EXPERT_MATRIX_BYTES', 300):
            clone, _ = self.apply(state)
            clone.inject_model()
            try:
                with self.assertRaisesRegex(api.AdapterError, 'budget'):
                    module.expert_linear(torch.randn(2, 4), 1)
            finally:
                clone.eject_model()

    def test_gguf_empty_markers_preserve_all_logical_shapes(self):
        root = torch.nn.Module()
        for name, shape in (('conv', (4, 2, 3, 3)), ('norm', (4,)), ('dense', (4, 4))):
            module = torch.nn.Module()
            module.weight = torch.empty(0)
            module._gguf_logical_shape = shape
            setattr(root, name, module)
        bank = Bank()
        del bank.weight
        bank.weight = torch.empty(0)
        bank._gguf_logical_shape = (2, 4, 4)
        bank.num_experts, bank.out_features, bank.in_features = 2, 4, 4
        root.bank = bank
        targets, _ = api._targets(root)
        for name in ('conv', 'norm', 'dense', 'bank'):
            self.assertEqual(targets[(name + '.weight', None)][1], getattr(root, name)._gguf_logical_shape)

    def test_gguf_selected_expert_api_and_native_value_fallback(self):
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        original = module.weight.detach().clone()
        del module.weight
        module.weight = torch.empty(0)
        module._gguf_logical_shape = (2, 4, 4)
        module.num_experts, module.out_features, module.in_features = 2, 4, 4
        module.expert_linear = lambda x, i: F.linear(x, original[int(i)])
        reads = []
        module.expert_weight = lambda i: (reads.append(i), original[int(i)])[1]
        state = self.lora(EXPERT)
        state[EXPERT + '.dora_scale'] = torch.ones(4, 1)
        for use_expert_api in (True, False):
            if not use_expert_api:
                del module.expert_weight
                def value(name, marker, expert=None):
                    self.assertEqual(name, 'weight')
                    self.assertEqual(expert, 1)
                    reads.append(expert)
                    return original[expert].to(marker)
                module._value = value
            clone, _ = self.apply(state, 1)
            clone.inject_model()
            try:
                reads.clear()
                output = module.expert_linear(torch.ones(1, 4), 1)
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(reads, [1])
            finally:
                clone.eject_model()
            self.assertEqual(module.weight.numel(), 0)

    def test_selected_matrix_composes_prior_factors_and_preserves_bias(self):
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        original = module.weight.detach().clone()
        bias = torch.randn(4)
        original_method = lambda x, i: F.linear(x, original[int(i)], bias)
        module.expert_linear = original_method
        first_state = self.lora(EXPERT)
        first, _ = self.apply(first_state)
        intermediate = original[1] + first_state[EXPERT + '.lora_B.weight'] @ first_state[EXPERT + '.lora_A.weight']
        _, state, expected = next(self.advanced_cases(EXPERT, intermediate))
        second, _ = self.apply(state, 1, model=first)
        second.inject_model()
        try:
            x = torch.randn(2, 4)
            torch.testing.assert_close(module.expert_linear(x, 1), F.linear(x, expected, bias), atol=1e-5, rtol=1e-5)
        finally:
            second.eject_model()
        self.assertIs(module.expert_linear, original_method)
        torch.testing.assert_close(module.weight, original)

    def test_comfy_swallowed_matrix_error_becomes_explicit_failure(self):
        import logging
        module = self.base.model.diffusion_model.model.layers[0].mlp.experts_down_proj
        clone, _ = self.apply({EXPERT + '.oft_blocks': torch.zeros(2, 2, 2)})
        cls = next(cls for cls in sys.modules['comfy.weight_adapter'].adapters if cls.name == 'oft')
        def broken(adapter, weight, key, *args, **kwargs):
            logging.error(f'ERROR oft {key} simulated calculation failure')
            return weight
        handlers = list(logging.getLogger().handlers)
        clone.inject_model()
        try:
            with patch.object(cls, 'calculate_weight', broken):
                with self.assertRaisesRegex(api.AdapterError, 'simulated calculation failure'):
                    module.expert_linear(torch.ones(1, 4), 1)
        finally:
            clone.eject_model()
        self.assertEqual(logging.getLogger().handlers, handlers)

    def test_large_expert_budget_is_dtype_sensitive_without_allocating_bank(self):
        registry = sys.modules['comfy.weight_adapter']
        group = {suffix.removeprefix(EXPERT): value for suffix, value in self.lora(EXPERT).items()}
        group['.dora_scale'] = torch.ones(4, 1)
        adapter = api._load_group(group, registry)
        self.assertLess(api._matrix_budget((6144, 4096), [adapter], torch.bfloat16), 256 * 1024 * 1024)
        with self.assertRaisesRegex(api.AdapterError, 'budget'):
            api._matrix_budget((6144, 4096), [adapter], torch.float32)


if __name__ == '__main__':
    unittest.main()
