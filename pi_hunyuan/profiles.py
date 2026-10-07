"""Conservative total-device budgets; a selected smaller card can be simulated."""
import math

PROFILES = {
    '8':  {'budget_gib': 6.0, 'tiled': True,  'recommended_size': [768, 768]},
    '10': {'budget_gib': 8.0, 'tiled': True,  'recommended_size': [1024, 768]},
    '12': {'budget_gib': 9.5, 'tiled': True,  'recommended_size': [1024, 1024]},
    '16': {'budget_gib': 13.0, 'tiled': False, 'recommended_size': [1024, 1024]},
    '24': {'budget_gib': 20.0, 'tiled': False, 'recommended_size': [1024, 1536]},
    '32': {'budget_gib': 27.5, 'tiled': False, 'recommended_size': [1024, 1536]},
}


def normalize(value):
    value = str(value or 'auto').lower().strip().replace(' gb', '')
    if value == 'automatic':
        value = 'auto'
    if value not in ('auto', *PROFILES):
        raise ValueError('VRAM profile must be Automatic, 8, 10, 12, 16, 24 or 32 GB.')
    return value


def plan(total_gib, profile='auto', mode='auto'):
    total = float(total_gib)
    if not math.isfinite(total) or total < 4:
        raise ValueError('This model needs at least a 4 GiB addressable GPU; 8/10 GiB profiles are experimental.')
    selected = normalize(profile)
    nominal = selected
    if selected == 'auto':
        nominal = next((name for name in reversed(PROFILES) if total >= int(name) * 0.92), '8')
    recipe = dict(PROFILES[nominal])
    budget = min(recipe['budget_gib'], max(2, total - 1.5))
    if mode == 'speed' and selected == 'auto':
        budget = total
    elif mode == 'low-memory':
        budget = min(budget, 8, total * 0.65)
    return dict(recipe, selected=selected, nominal_gb=int(nominal), budget_gib=round(budget, 3),
                headroom_gib=round(max(0, total-budget), 3),
                tiled=recipe['tiled'] or mode == 'low-memory',
                simulation=selected != 'auto' and int(selected) < total * 0.92)
