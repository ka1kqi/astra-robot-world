"""Catalog behavior: physical definitions, retrieval isolation, and local imports."""
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from astra_world.catalog import get_asset, load_catalog, search_assets


def test_every_procedural_asset_compiles_and_has_finite_physics():
    catalog = load_catalog()
    assert len(catalog) >= 20
    for name, asset in catalog.items():
        assert asset['id'] == name
        assert np.isfinite(asset['bounds']).all() and min(asset['bounds']) > 0
        assert asset['default_mass'] > 0
        geoms = []
        for part in asset['geometries']:
            attrs = ' '.join(f'{key}="{" ".join(map(str, part[key]))}"' for key in ('size', 'pos', 'euler'))
            geoms.append(f'<geom type="{part["type"]}" {attrs}/>')
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>' + ''.join(geoms) + '</body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all(), name


def test_search_matches_tags_and_returns_independent_definitions():
    assert 'bin' in {item['id'] for item in search_assets('container')}
    assert len(search_assets(limit=2)) == 2
    assert search_assets('no_such_asset_abcdef') == []
    asset = get_asset('box')
    asset['geometries'][0]['size'][0] = -1
    assert get_asset('box')['geometries'][0]['size'][0] > 0
    with pytest.raises(KeyError):
        get_asset('not_an_asset')


def test_import_cli_recenters_scales_and_records_license(tmp_path):
    mesh = tmp_path / 'tetra.obj'
    mesh.write_text('v 10 0 0\nv 12 0 0\nv 10 2 0\nv 10 0 2\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
    destination = tmp_path / 'local'
    command = [sys.executable, 'scripts/import_asset.py', str(mesh), '--id', 'tetra', '--license', 'CC0-1.0', '--units', 'cm', '--output-dir', str(destination)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    imported = load_catalog(custom_catalog_path=destination / 'catalog.json')['tetra']
    assert imported['bounds'] == pytest.approx([.02, .02, .02])
    assert imported['license'] == 'CC0-1.0'
    assert imported['collision_mode'] == 'convex_hull'
    assert Path(imported['geometries'][0]['file']).exists()
    assert imported['readiness'] == 'imported'
    assert (destination / 'tetra' / 'source.obj').read_bytes() == mesh.read_bytes()
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0


@pytest.mark.parametrize('flag,value', [('--scale', 'nan'), ('--mass', '-1'), ('--id', '../escape')])
def test_import_cli_rejects_unsafe_metadata(tmp_path, flag, value):
    mesh = tmp_path / 'empty.obj'
    mesh.write_text('')
    result = subprocess.run([sys.executable, 'scripts/import_asset.py', str(mesh), '--id', 'safe', '--license', 'CC0', '--output-dir', str(tmp_path / 'out'), flag, value], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / 'out').exists()
