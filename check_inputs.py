import json
from pathlib import Path

nb = json.loads(Path('name_that_llm/public_notebooks/areebbb_0643/0-643-same-stack-sharper-blend-enjoy.ipynb').read_text(encoding='utf-8'))
src = "".join(nb['cells'][7]['source'])
ns = {}
exec(src, ns)
checks = ns.get('REQUIRED_INPUT_CHECKS', [])
print(f"Total required input checks: {len(checks)}")
for c in checks:
    print(c)
