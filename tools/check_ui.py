"""Build real Gradio controls with simulated Forge interfaces; no server."""
from pathlib import Path
import faulthandler
faulthandler.dump_traceback_later(30, exit=True)
import importlib.util
import sys
import gradio as gr

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('test_forge_ui_smoke', root / 'tests' / 'test_forge.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
case = tests.IntegrationTests()
case.setUp()
try:
    engine = tests.load('pi_hunyuan_gradio_smoke', 'scripts/engine.py')
    with gr.Blocks(analytics_enabled=False) as ui:
        select = gr.Dropdown(choices=case.presets.PresetArch.choices(), value='sd', elem_id='forge_ui_preset')
        case.forge.capture(select, elem_id='forge_ui_preset')
        txt = engine.Script().ui(False)
        edit = engine.Script().ui(True)
        case.forge.bind_panels()
    config = ui.get_config_file()
    assert len(txt) == 12 and len(edit) == 12
    assert any(component.get('props', {}).get('elem_id') == 'pi_hunyuan3_txt2img_panel' for component in config['components'])
    assert len(config['dependencies']) >= 1
    preset_control = next(component for component in config['components'] if component.get('props', {}).get('elem_id') == 'forge_ui_preset')
    assert ['HunyuanImage 3.0', 'hunyuan3'] in [list(choice) for choice in preset_control['props']['choices']]
    print('UI_BUILD_OK: real Gradio image controls and preset events; simulated Forge interfaces, no browser acceptance.')
finally:
    case.doCleanups()
    faulthandler.cancel_dump_traceback_later()
