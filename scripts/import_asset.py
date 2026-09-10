#!/usr/bin/env python3
"""Import a local OBJ/STL with explicit provenance and convex-hull collision.

Example: uv run python scripts/import_asset.py model.obj --id sample --license CC0-1.0 --units cm
Imports are geometry only: no grasp poses, joints, or robot controllers are inferred.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--id', required=True, dest='asset_id')
    parser.add_argument('--license', required=True)
    parser.add_argument('--source', help='Source URL or attribution; defaults to local filename')
    parser.add_argument('--units', choices=['m', 'cm', 'mm'], default='m')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--mass', type=float, default=1.0)
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'assets' / 'custom')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', args.asset_id):
        parser.error('--id must be a lowercase identifier (letters, digits, underscores)')
    if not args.license.strip():
        parser.error('--license must identify the permission to use this mesh')
    if any(not math.isfinite(value) or value <= 0 for value in [args.scale, args.mass]):
        parser.error('--scale and --mass must be finite positive values')
    if not args.mesh.is_file() or args.mesh.suffix.lower() not in {'.obj', '.stl'}:
        parser.error('mesh must be an existing local OBJ or STL file')
    from astra_world.catalog import load_catalog
    output = args.output_dir.resolve()
    catalog_path = output / 'catalog.json'
    if args.asset_id in load_catalog(custom_catalog_path=catalog_path):
        parser.error(f'asset id already registered: {args.asset_id}')
    import numpy as np
    import trimesh
    try:
        mesh = trimesh.load(args.mesh, force='mesh', process=True)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4 or not len(mesh.faces):
            raise ValueError('mesh must contain a three-dimensional surface')
        if not np.isfinite(mesh.vertices).all():
            raise ValueError('mesh coordinates must be finite')
        mesh.apply_scale(args.scale * {'m': 1, 'cm': .01, 'mm': .001}[args.units])
        dimensions = mesh.extents
        if not np.isfinite(dimensions).all() or (dimensions <= 0).any():
            raise ValueError('mesh must have finite positive extent on every axis')
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        hull = mesh.convex_hull
        if hull.volume <= 0:
            raise ValueError('mesh must have nonzero three-dimensional convex volume')
        visual_obj = mesh.export(file_type='obj', include_texture=False)
        collision_obj = hull.export(file_type='obj', include_texture=False)
    except Exception as exc:
        parser.error(f'could not import mesh: {exc}')
    document = json.loads(catalog_path.read_text()) if catalog_path.exists() else {'schema_version': 1, 'assets': []}
    folder = output / args.asset_id
    if folder.exists():
        parser.error(f'asset directory already exists: {folder}')
    folder.mkdir(parents=True)
    shutil.copyfile(args.mesh, folder / f'source{args.mesh.suffix.lower()}')
    visual_path = folder / 'visual.obj'
    collision_path = folder / 'collision.obj'
    visual_path.write_text(visual_obj)
    collision_path.write_text(collision_obj)
    asset = dict(
        id=args.asset_id, description=f'Locally imported {args.mesh.name}; convex-hull collision approximation',
        tags=['imported', 'mesh'], source=args.source or f'local:{args.mesh.name}',
        license=args.license.strip(), readiness='imported', origin='bbox_center',
        bounds=dimensions.tolist(), default_mass=args.mass, default_fixed=args.fixed,
        collision_mode='convex_hull', visual_file=str(visual_path),
        geometries=[dict(type='mesh', file=str(collision_path), size=[1, 1, 1], pos=[0, 0, 0], euler=[0, 0, 0])],
        import_settings=dict(units=args.units, scale=args.scale),
    )
    document['assets'].append(asset)
    temporary = catalog_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(catalog_path)
    print(json.dumps(asset, indent=2))


if __name__ == '__main__':
    main()
