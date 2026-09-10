import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANDA = Path(
    os.environ.get(
        "ASTRA_PANDA_PATH", ROOT / "assets/menagerie/franka_emika_panda/panda.xml"
    )
)
CATALOG = [
    {
        "id": "panda",
        "kind": "robot",
        "source": "MuJoCo Menagerie",
        "license": "Apache-2.0",
    },
    {
        "id": "block",
        "kind": "prop",
        "dimensions": [0.04, 0.04, 0.04],
        "mass": 0.04,
        "colors": ["red", "blue", "green"],
    },
    {
        "id": "bin",
        "kind": "container",
        "interior": [0.25, 0.25],
        "source": "procedural",
    },
    {"id": "obstacle", "kind": "barrier", "source": "procedural"},
]
