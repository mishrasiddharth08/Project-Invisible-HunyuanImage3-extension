"""Standalone integration checks; no Forge, model weights or GPU required."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'PROJECT INVISIBLE â€” HunyuanImage 3.0'


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.inventory = dict(model=['/weights/model.safetensors'], vae=['/weights/vae.safetensors'],
                              vision=['/weights/vision.safetensors'], head=['/weights/head.safetensors'])
        self.runtime = module('pi_hunyuan.runtime', generate=Mock(return_value='worker result'), release=Mock())
        def resolve(value, role, inventory, required=True):
            if value == 'Auto':
                return inventory[role][0] if inventory[role] else None
            if value not in inventory[role]:
                raise ValueError('Invalid component')
            return value
        self.assets = module('pi_hunyuan.assets', scan=Mock(side_effect=lambda: self.inventory), resolve=Mock(side_effect=resolve))
        self.opts = SimpleNamespace(forge_preset='sd', sd_model_checkpoint='ordinary', forge_additional_modules=[],
                                    data_labels={}, save=Mock())
        def set_value(key, value):
            setattr(self.opts, key, value)
            info = self.opts.data_labels.get(key)
            if info and info.onchange:
                info.onchange()
        self.opts.set = set_value
        self.opts.onchange = lambda key, callback, call=False: setattr(self.opts.data_labels[key], 'onchange', callback)
        self.opts.add_option = lambda key, info: self.opts.data_labels.__setitem__(key, info)
        for key in ('forge_preset', 'sd_model_checkpoint', 'forge_additional_modules'):
            self.opts.data_labels[key] = SimpleNamespace(onchange=Mock())
        self.models = module('modules.sd_models', checkpoints_list={}, checkpoint_aliases={}, list_models=Mock(),
                             forge_model_reload=Mock(return_value='native model'),
                             model_data=SimpleNamespace(forge_loading_parameters={}))
        models = self.models
        class CheckpointInfo:
            def register(self):
                models.checkpoints_list[self.title] = self
                for alias in self.ids:
                    models.checkpoint_aliases[alias] = self
        self.models.CheckpointInfo = CheckpointInfo
        self.models.checkpoints_list['ordinary'] = SimpleNamespace(filename='/weights/ordinary.safetensors')
        class ScriptRunner:
            def run(self, p, *args, **kwargs):
                return 'ordinary runner'
        class Img2Img:
            pass
        self.processing = module('modules.processing', process_images=Mock(return_value='ordinary result'),
                                 StableDiffusionProcessingImg2Img=Img2Img)
        self.callbacks = module('modules.script_callbacks')
        for name in ('on_app_started', 'on_before_ui', 'on_after_component', 'on_script_unloaded', 'on_ui_tabs'):
            setattr(self.callbacks, name, Mock())
        self.scripts = module('modules.scripts', ScriptRunner=ScriptRunner, Script=type('Script', (), {}), AlwaysVisible=True)
        self.shared = module('modules.shared', opts=self.opts, config_filename='/unused/config.json')
        self.entry = module('modules_forge.main_entry', checkpoint_change=Mock(return_value='ordinary selection'),
                            use_distill=lambda preset: preset == 'flux')
        self.template = {
            key: SimpleNamespace(default=value, section=('ui_sd', 'SD'), onchange=None)
            for key, value in {'sd_t2i_step': 30, 'sd_t2i_cfg': 5, 'forge_checkpoint_sd': 'ordinary',
                               'forge_additional_modules_sd': [], 'unrelated': 1}.items()}
        for mode in ('t2i', 'i2i'):
            for suffix, value in {'sampler': 'Euler a', 'scheduler': 'Automatic', 'step': 30,
                                  'cfg': 7, 'width': 512, 'height': 512}.items():
                self.template[f'sd_{mode}_{suffix}'] = SimpleNamespace(default=value, section=('ui_sd', 'SD'), onchange=None)
        for suffix in ('step', 'cfg'):
            self.template[f'sd_t2i_hr_{suffix}'] = SimpleNamespace(default=30, section=('ui_sd', 'SD'), onchange=None)
        class Arch:
            choices = staticmethod(lambda: ['sd'])
        self.presets = module('modules_forge.presets', register=lambda result: result.update(self.template), PresetArch=Arch)
        package = module('pi_hunyuan', assets=self.assets, runtime=self.runtime)
        package.__path__ = [str(ROOT / 'pi_hunyuan')]
        self.api = module('modules.api.api', process_images=Mock(return_value='ordinary API'))
        enabled = {'pi_hunyuan': package, 'pi_hunyuan.assets': self.assets, 'pi_hunyuan.runtime': self.runtime,
                   'pi_hunyuan.config': module('pi_hunyuan.config', ROOT=ROOT, LABEL=LABEL, PRESET='hunyuan3'),
                   'modules': module('modules', shared=self.shared, sd_models=self.models, scripts=self.scripts,
                                     processing=self.processing, script_callbacks=self.callbacks),
                   'modules.shared': self.shared, 'modules.sd_models': self.models, 'modules.scripts': self.scripts,
                   'modules.processing': self.processing, 'modules.script_callbacks': self.callbacks,
                   'modules.api.api': self.api,
                   'modules_forge': module('modules_forge', presets=self.presets, main_entry=self.entry),
                   'modules_forge.presets': self.presets, 'modules_forge.main_entry': self.entry}
        self.module_patch = patch.dict(sys.modules, enabled)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.forge = load('pi_hunyuan.forge', 'pi_hunyuan/forge.py')
        self.preset = load('pi_hunyuan.preset', 'pi_hunyuan/preset.py')
        package.forge = self.forge
        package.preset = self.preset
        self.forge.install()

    def request(self, **overrides):
        script = SimpleNamespace(_pi_hunyuan3=True, args_from=1, args_to=10)
        return SimpleNamespace(override_settings=overrides, scripts=SimpleNamespace(alwayson_scripts=[script]),
                               script_args=[None, 'Auto', 'Auto', 'Auto', 'Auto', 'off', False, False, None, None])

    def test_unload_restores_only_current_owned_hooks(self):
        saved = list(self.forge.HOOKS)
        self.forge.uninstall()
        self.assertFalse(self.forge.ENABLED)
        self.assertFalse(self.forge.HOOKS)
        for owner, name, previous, replacement in saved:
            self.assertIs(getattr(owner, name), previous)
        self.runtime.release.assert_called_once()

    def test_later_wrapper_survives_unload_and_inner_hook_delegates(self):
        from functools import wraps
        inner = self.processing.process_images
        @wraps(inner)
        def later(*args, **kwargs):
            return ('later', inner(*args, **kwargs))
        self.processing.process_images = later
        self.forge.uninstall()
        self.assertIs(self.processing.process_images, later)
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL)
        self.assertEqual(later(p), ('later', 'ordinary result'))
        self.runtime.generate.assert_not_called()

    def test_preexisting_wrapper_restored_after_unload(self):
        self.forge.uninstall()
        previous = self.processing.process_images
        def before(*args, **kwargs):
            return ('before', previous(*args, **kwargs))
        self.processing.process_images = before
        self.forge.install()
        self.assertEqual(self.processing.process_images(self.request()), ('before', 'ordinary result'))
        self.forge.uninstall()
        self.assertIs(self.processing.process_images, before)

    def test_reinstall_handles_later_wrapper_with_inherited_marker(self):
        from functools import wraps
        inner = self.processing.process_images
        @wraps(inner)
        def later(*args, **kwargs):
            return inner(*args, **kwargs)
        self.processing.process_images = later
        self.forge.uninstall()
        self.forge.install()
        self.assertIsNot(self.processing.process_images, later)
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL)
        self.assertEqual(self.processing.process_images(p), 'worker result')
        self.runtime.generate.assert_called_once()

    def test_registration_and_refresh_are_idempotent(self):
        self.forge.register()
        owned = [item for item in self.models.checkpoints_list.values() if getattr(item, '_pi_hunyuan3', False)]
        self.assertEqual(len(owned), 2)
        self.assertIn('/weights/model.safetensors', self.models.checkpoint_aliases)
        self.assertEqual(self.models.checkpoint_aliases[LABEL].calculate_shorthash(), None)
        self.models.list_models()
        self.assertEqual(len(self.models.checkpoints_list), 3)

    def test_ordinary_generation_delegates_and_releases(self):
        self.assertEqual(self.processing.process_images(self.request()), 'ordinary result')
        self.runtime.generate.assert_not_called()
        self.runtime.release.assert_called_once()

    def test_generate_api_and_runner_use_worker(self):
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL)
        for route in (self.processing.process_images, self.api.process_images,
                      lambda p: self.scripts.ScriptRunner.run(p.scripts, p=p)):
            self.assertEqual(route(p), 'worker result')
        self.assertEqual(self.runtime.generate.call_count, 3)
        self.assertEqual(self.opts.forge_preset, 'sd')

    def test_physical_checkpoint_overrides_file_control(self):
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint='/weights/model.safetensors')
        p.script_args[1] = 'stale model'
        self.processing.process_images(p)
        self.assertEqual(self.runtime.generate.call_args.args[1]['model'], '/weights/model.safetensors')

    def test_unmarked_physical_checkpoint_routes(self):
        self.models.checkpoints_list['physical'] = SimpleNamespace(filename='/weights/model.safetensors')
        self.assertTrue(self.forge.selected(self.request(forge_preset='hunyuan3', sd_model_checkpoint='physical')))

    def test_markers_with_wrong_preset_fail_closed(self):
        for value in (LABEL, '/weights/model.safetensors'):
            with self.assertRaises(ValueError):
                self.processing.process_images(self.request(sd_model_checkpoint=value))
        self.runtime.generate.assert_not_called()

    def test_foreign_checkpoint_with_hunyuan_preset_fails_closed(self):
        with self.assertRaises(ValueError):
            self.processing.process_images(self.request(forge_preset='hunyuan3'))

    def test_additional_modules_override_script_and_global_choices(self):
        self.opts.forge_additional_modules = ['/weights/vision.safetensors']
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL,
                         forge_additional_modules=['/weights/vae.safetensors'])
        p.script_args[2:5] = ['stale vae', 'stale vision', 'stale head']
        self.processing.process_images(p)
        values = self.runtime.generate.call_args.args[1]
        self.assertEqual(values['vae'], '/weights/vae.safetensors')
        self.assertEqual(values['vision'], 'Auto')
        self.assertEqual(values['head'], 'Auto')
        self.assertEqual([call.args[1] for call in self.assets.resolve.call_args_list], ['model', 'vae'])
        self.assertEqual(self.opts.forge_additional_modules, ['/weights/vision.safetensors'])

    def test_empty_additional_modules_does_not_use_stale_global_selection(self):
        self.opts.forge_additional_modules = ['foreign component']
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL, forge_additional_modules=[])
        self.assertEqual(self.processing.process_images(p), 'worker result')

    def test_unknown_module_and_rewrite_fail_before_worker(self):
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL, forge_additional_modules=['foreign'])
        with self.assertRaises(ValueError):
            self.processing.process_images(p)
        p.override_settings['forge_additional_modules'] = []
        p.script_args[5] = 'invalid rewrite'
        with self.assertRaises(ValueError):
            self.processing.process_images(p)
        self.runtime.generate.assert_not_called()

    def test_reference_positions_are_kept_only_for_img2img(self):
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL)
        ref2, ref3 = object(), object()
        p.script_args[5:] = ['think + rewrite', True, True, ref2, ref3]
        values = self.forge.options(p.scripts, p)
        self.assertIsNone(values['reference2'])
        self.assertEqual(values['refs'], [])
        edit = self.processing.StableDiffusionProcessingImg2Img()
        edit.__dict__.update(p.__dict__)
        values = self.forge.options(edit.scripts, edit)
        self.assertIs(values['reference2'], ref2)
        self.assertIs(values['reference3'], ref3)
        self.assertEqual(values['refs'], [ref2, ref3])
        self.assertEqual([values[key] for key in ('rewrite', 'spectrum', 'keep_loaded')], ['think + rewrite', True, True])

    def test_switch_releases_before_existing_callback(self):
        events = []
        self.runtime.release.side_effect = lambda: events.append('release')
        self.opts.data_labels['forge_preset'].onchange = lambda: events.append('existing')
        self.forge.selection_release()
        self.opts.set('forge_preset', 'hunyuan3')
        self.assertEqual(events, ['release', 'existing'])

    def test_unchanged_selection_retains_warm_worker(self):
        self.forge.selection_release()
        self.opts.data_labels['forge_preset'].onchange()
        self.runtime.release.assert_not_called()

    def test_loader_rejects_marker_in_current_or_pending_selection(self):
        self.opts.sd_model_checkpoint = LABEL
        with self.assertRaises(RuntimeError):
            self.models.forge_model_reload()
        self.opts.sd_model_checkpoint = 'ordinary'
        self.models.model_data.forge_loading_parameters = {'checkpoint_info': self.models.checkpoint_aliases[LABEL]}
        with self.assertRaises(RuntimeError):
            self.models.forge_model_reload()
        self.models.model_data.forge_loading_parameters = {}
        self.assertEqual(self.models.forge_model_reload(), 'native model')

    def test_checkpoint_selection_bypasses_native_loader_and_preserves_modules(self):
        self.opts.forge_additional_modules = ['/weights/vae.safetensors']
        self.assertTrue(self.entry.checkpoint_change(LABEL, 'hunyuan3'))
        self.assertEqual(self.opts.forge_additional_modules, ['/weights/vae.safetensors'])
        self.opts.save.assert_called_once()
        with self.assertRaises(ValueError):
            self.entry.checkpoint_change(LABEL, 'sd')

    def test_reinstall_does_not_duplicate_hooks(self):
        process = self.processing.process_images
        self.forge.install()
        self.assertIs(self.processing.process_images, process)
        self.callbacks.on_app_started.assert_called_once()

    def test_preset_clones_defaults_without_mutating_template(self):
        self.preset.install()
        self.preset.install()
        self.assertEqual(self.presets.PresetArch.choices(), ['sd', ('HunyuanImage 3.0', 'hunyuan3')])
        self.assertEqual(self.opts.data_labels['forge_checkpoint_hunyuan3'].default, LABEL)
        self.opts.data_labels['forge_additional_modules_hunyuan3'].default.append('test')
        self.assertEqual(self.template['forge_additional_modules_sd'].default, [])
        self.assertNotIn('unrelated', self.opts.data_labels)

    def test_native_preset_defaults(self):
        self.preset.install()
        for mode in ('t2i', 'i2i'):
            for suffix, expected in {'sampler': 'Euler', 'scheduler': 'Simple', 'step': 8,
                                     'cfg': 1.0, 'dcfg': 2.5, 'width': 1024, 'height': 1024}.items():
                self.assertEqual(self.opts.data_labels[f'hunyuan3_{mode}_{suffix}'].default, expected)
        for suffix, expected in {'step': 8, 'cfg': 1.0, 'dcfg': 2.5}.items():
            self.assertEqual(self.opts.data_labels[f'hunyuan3_t2i_hr_{suffix}'].default, expected)
        self.assertEqual(self.opts.data_labels['hunyuan3_i2i_denoising_strength'].default, 1.0)
        self.assertEqual(self.template['sd_t2i_sampler'].default, 'Euler a')

    def test_distilled_guidance_visibility_is_hunyuan_only(self):
        self.assertTrue(self.entry.use_distill('hunyuan3'))
        self.assertTrue(self.entry.use_distill('flux'))
        self.assertFalse(self.entry.use_distill('sd'))

    def test_optional_auto_selection_is_deferred_to_variant_matching(self):
        self.inventory['vision'] = ['/weights/distil-vision.safetensors', '/weights/instruct-vision.safetensors']
        self.inventory['head'] = ['/weights/distil-head.safetensors', '/weights/instruct-head.safetensors']
        p = self.request(forge_preset='hunyuan3', sd_model_checkpoint=LABEL)
        values = self.forge.options(p.scripts, p)
        self.assertEqual((values['vision'], values['head']), ('Auto', 'Auto'))
        p.script_args[3] = self.inventory['vision'][1]
        self.assertEqual(self.forge.options(p.scripts, p)['vision'], self.inventory['vision'][1])

    def test_worker_marker_resource_matches_registration(self):
        import json
        marker = Path(self.models.checkpoint_aliases[LABEL].filename)
        self.assertTrue(marker.is_file())
        self.assertTrue(json.loads(marker.read_text())['_pi_hunyuan3'])

    def test_missing_hook_fails_before_registering_more_markers(self):
        self.models.forge_model_reload = None
        with self.assertRaises(RuntimeError):
            self.forge.install()

    def test_ui_contract_visibility_refresh_and_rebuild(self):
        created = []
        class Component:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                self.change, self.click = Mock(), Mock()
                created.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        gr = module('gradio', update=lambda **kwargs: kwargs)
        for name in ('Accordion', 'Dropdown', 'Button', 'Checkbox', 'Radio', 'Row', 'Image'):
            setattr(gr, name, Component)
        with patch.dict(sys.modules, {'gradio': gr}):
            engine = load('hunyuan_test_engine', 'scripts/engine.py')
            first = Component()
            self.forge.capture(first, elem_id='forge_ui_preset')
            denoise = Component()
            self.forge.capture(denoise, elem_id='img2img_denoising_strength')
            controls = engine.Script().ui(False)
            edit_controls = engine.Script().ui(True)
            self.assertEqual(len(controls), 12)
            self.assertEqual(controls[4].kwargs['value'], 'off')
            self.assertEqual(controls[7].kwargs['type'], 'pil')
            self.assertFalse(self.forge.PANELS[0].kwargs['visible'])
            self.forge.bind_panels()
            callback = first.change.call_args.args[0]
            self.assertEqual(callback('hunyuan3'), [{'visible': True}, {'visible': True}, {'value': 1.0}])
            self.assertEqual(callback('sd'), [{'visible': False}, {'visible': False}, {}])
            self.assertTrue(all(control.kwargs['visible'] is False for control in controls[:4]))
            self.assertFalse(any(control.args == ('Refresh local files',) for control in created))
            self.forge.reset_ui()
            self.assertEqual(self.forge.PANELS, [])
            self.assertEqual(self.forge.BOUND, set())

    def test_native_component_refresh_preserves_existing_models(self):
        self.entry.module_list = {'vae.safetensors': '/ordinary/vae.safetensors'}
        self.entry.refresh_models = Mock(return_value=(['ordinary'], ['vae.safetensors']))
        self.inventory['vae'] = ['/hunyuan/vae.safetensors']
        self.inventory['vision'] = ['/hunyuan/vision.safetensors']
        self.forge.install_component_refresh()
        hook = self.entry.refresh_models
        self.forge.install_component_refresh()
        self.assertIs(self.entry.refresh_models, hook)
        checkpoints, modules = hook()
        self.assertEqual(checkpoints, ['ordinary'])
        self.assertEqual(self.entry.module_list['vae.safetensors'], '/ordinary/vae.safetensors')
        self.assertIn('/hunyuan/vae.safetensors', modules)
        self.assertEqual(self.entry.module_list['vision.safetensors'], '/hunyuan/vision.safetensors')


if __name__ == '__main__':
    unittest.main()
