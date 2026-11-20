"""Verify real pseudocode declarations and recorded file hashes without source markers."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import re


def audit(repo):
    inventory=json.loads((repo/'audit/source_inventory.json').read_text('utf-8'))
    recorded={row['pseudocode_path']:row for row in
              json.loads((repo/'audit/source_coverage.json').read_text('utf-8'))}
    errors=[]
    original_count=covered_count=0
    algorithm_files=list(repo.rglob('*.pseudo'))
    actual_by_path={}
    for path in algorithm_files:
        content=path.read_text('utf-8')
        rel=path.relative_to(repo).as_posix()
        actual_by_path[rel]=Counter(re.findall(
            r'^\s*(?:CLASS|PROCEDURE)\s+([A-Za-z_][\w.]*)',content,re.M))
        if re.search(r'\b(?:SOURCE_FILE|ORIGINAL_SOURCE|SOURCE_SYMBOL|ADDED_SYMBOL)\b',content):
            errors.append(f'Provenance annotation in algorithm: {rel}')
    for item in inventory:
        rel=Path(item['path']).with_suffix('.pseudo').as_posix()
        path=repo/rel
        expected=Counter(s['name'] for s in item['symbols'])
        actual=actual_by_path.get(rel,Counter())
        original_count+=sum(expected.values())
        covered_count+=sum((expected&actual).values())
        if not path.exists(): errors.append(f'Missing module: {rel}')
        if expected-actual: errors.append(f'Missing declarations: {rel}: {dict(expected-actual)}')
        if path.exists():
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if recorded.get(rel,{}).get('pseudocode_sha256')!=digest:
                errors.append(f'Coverage hash is stale: {rel}')
    return {'check_kind':'actual_declaration_and_coverage_hash_check',
            'source_files':len(inventory),'algorithm_files':len(algorithm_files),
            'source_declarations':original_count,'covered_declarations':covered_count,
            'executed_pseudocode':False,'behavioral_correctness_proven':False,'errors':errors}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(Path(__file__).resolve().parents[2])
    rendered=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output: args.output.write_text(rendered+'\n',encoding='utf-8')
    print(rendered)
    raise SystemExit(bool(result['errors']))
