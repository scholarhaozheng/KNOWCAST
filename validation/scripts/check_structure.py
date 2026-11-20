"""Check pseudocode module paths and class/procedure declarations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def check_modules(repo: Path) -> dict:
    modules = json.loads((repo / 'validation/modules.json').read_text(encoding='utf-8'))
    expected_paths = {module['path'] for module in modules}
    actual_paths = {path.relative_to(repo).as_posix() for path in repo.rglob('*.pseudo')}
    errors = [f'Missing module: {path}' for path in sorted(expected_paths - actual_paths)]
    errors += [f'Module absent from index: {path}' for path in sorted(actual_paths - expected_paths)]
    declaration_count = 0
    for module in modules:
        path = repo / module['path']
        if not path.is_file():
            continue
        declarations = re.findall(r'^\s*(CLASS|PROCEDURE)\s+([A-Za-z_][\w.]*)',
                                  path.read_text(encoding='utf-8'), re.M)
        declaration_count += len(declarations)
        expected = Counter((item['kind'], item['name']) for item in module['declarations'])
        actual = Counter(declarations)
        for (kind, name), count in (expected - actual).items():
            errors.append(f'Missing declaration: {module["path"]}: {kind} {name} ({count})')
        for (kind, name), count in (actual - expected).items():
            errors.append(f'Declaration absent from index: {module["path"]}: {kind} {name} ({count})')
    return {'check_kind': 'module_declarations', 'modules': len(actual_paths),
            'declarations': declaration_count, 'errors': errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check_modules(Path(__file__).resolve().parents[2])
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    raise SystemExit(bool(result['errors']))
