"""Project-authored props and explicitly imported local mesh assets.

Bounds are full dimensions in metres. Every asset origin is its bounding-box
center; geom sizes follow MuJoCo conventions and geom Euler angles are degrees.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / 'assets' / 'catalog.json'
CUSTOM_CATALOG_PATH = PROJECT_ROOT / 'assets' / 'custom' / 'catalog.json'


def load_catalog(*, custom_catalog_path: Path | None = None) -> dict[str, dict]:
    """Load fresh definitions, including optional developer-imported local props."""
    catalog = {}
    for path in (CATALOG_PATH, custom_catalog_path or CUSTOM_CATALOG_PATH):
        if not path.exists():
            if path == CATALOG_PATH:
                raise FileNotFoundError(f'Procedural catalog missing: {path}')
            continue
        document = json.loads(path.read_text())
        for asset in document['assets']:
            if asset['id'] in catalog:
                raise ValueError(f'Duplicate asset id: {asset["id"]}')
            catalog[asset['id']] = asset
    return catalog


def get_asset(asset_id: str) -> dict:
    """Get an independent asset definition; unknown IDs raise KeyError."""
    return load_catalog()[asset_id]


def search_assets(query: str = '', limit: int = 20) -> list[dict]:
    """Search IDs, descriptions and tags; expose provenance and physical sizes."""
    terms = query.lower().split()
    results = []
    for asset in load_catalog().values():
        text = ' '.join([asset['id'], asset['description'], *asset['tags']]).lower()
        if all(term in text for term in terms):
            results.append({key: value for key, value in asset.items() if key != 'geometries'})
    return results[:max(0, limit)]
