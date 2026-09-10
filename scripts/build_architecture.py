"""Refresh the standalone architecture tool catalog from local public contracts."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astra_world.astra import TOOLS, EXPERIMENT_TOOLS  # noqa: E402

path = ROOT / "web" / "architecture.html"
data = [{**entry["function"], "budget": "experiment" if entry["function"]["name"] in EXPERIMENT_TOOLS else "live"} for entry in TOOLS]
encoded = json.dumps(data, indent=2).replace("<", "\\u003c")
page = path.read_text()
pattern = r'(<script id="tool-catalog" type="application/json">)\n.*?\n(</script>)'
updated, count = re.subn(pattern, lambda match: match[1] + "\n" + encoded + "\n" + match[2], page, flags=re.S)
if count != 1:
    raise SystemExit("Expected exactly one embedded tool catalog.")
if "--check" in sys.argv:
    if updated != page:
        raise SystemExit("Architecture catalog is stale; run scripts/build_architecture.py.")
    print(f"Architecture catalog matches all {len(data)} tools and exact parameter schemas.")
else:
    path.write_text(updated)
    print(f"Embedded {len(data)} tool contracts in web/architecture.html.")
