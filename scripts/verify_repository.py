"""Verify the standalone file selection without Git, Julia, training, or network access."""
from pathlib import Path
from collections import defaultdict
import ast
import csv
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path, delimiter='\t'):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def equal_event_mean(rows, metric='mse_h'):
    events = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        assert math.isfinite(value), 'Nonfinite saved metric'
        events[row['event']].append(value)
    assert len(events) == 18, 'Incomplete event coverage'
    return statistics.mean(statistics.mean(values) for values in events.values())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14), (actual, expected)


def main():
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts
             and '__pycache__' not in p.parts and not p.is_relative_to(ROOT / 'docs/figures')]
    assert not (ROOT / 'thesis').exists(), 'Thesis directory must not be distributed'
    assert not any((ROOT / name).exists() for name in ['tmp', '.local', 'tools', 'experiments'])
    required = [
        'Starting_networks/Project.toml', 'Starting_networks/Manifest.toml',
        'thesis_experiments/shared/conservative_pilot_utils.jl',
        'thesis_experiments/shared/reviewer_followup_runtime.jl',
        'Data/24_hour_source.jld2', 'Trained_models/flux_correction_pretrained.bson',
        *['Data/thesis_v4/' + name + '.jld2' for name in ['uniform', 'interior', 'boundary', 'control']],
    ]
    assert all((ROOT / name).is_file() for name in required)
    for name in required:
        if name.endswith('.jld2'):
            assert (ROOT / name).stat().st_size > 1_000_000, 'LFS data not downloaded: ' + name
    includes = 0
    for p in files:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        if p.suffix == '.jl':
            text = p.read_text(encoding='utf-8-sig')
            for expr in re.findall(r'include\((joinpath\([^\n]+?\)|"[^"]+")\)', text):
                if expr.startswith('"'):
                    target = p.parent / expr.strip('"')
                else:
                    base = re.match(r'joinpath\((@__DIR__|repository|root),', expr)
                    assert base, 'Unresolved include expression: ' + expr
                    target = p.parent if base[1] == '@__DIR__' else ROOT
                    for part in re.findall(r'"([^"]+)"', expr):
                        target = target / part
                assert target.resolve().is_file(), 'Missing include: ' + str(target)
                includes += 1
    lfs = (ROOT / '.gitattributes').read_text()
    assert '*.jld2 filter=lfs' in lfs and '*.bson filter=lfs' in lfs
    oversize = [p.relative_to(ROOT).as_posix() for p in files
                if p.stat().st_size > 100 * 1024 * 1024 and p.suffix not in ['.jld2', '.bson']]
    assert not oversize, 'Ordinary-Git file exceeds GitHub limit: ' + str(oversize)
    rules = {
        'GitHub token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})',
        'AWS access key': rb'(?:AKIA|ASIA)[A-Z0-9]{16}',
        'Private key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'Credential in URL': rb'https?://[^\s/@:]{2,}:[^\s/@]{3,}@',
        'Assigned secret': rb'(?i)(?:password|api_key|access_token|client_secret)\s*[:=]\s*[\x22\x27][A-Za-z0-9_+/=-]{12,}[\x22\x27]',
    }
    findings = []
    for p in files:
        assert p.name not in ['.env', 'credentials', 'id_rsa', 'id_ed25519']
        with p.open('rb') as stream:
            overlap = b''
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                data = overlap + block
                lowered = data.lower()
                prefixes = {
                    'GitHub token': (b'ghp_', b'gho_', b'ghu_', b'ghs_', b'ghr_', b'github_pat_'),
                    'AWS access key': (b'akia', b'asia'),
                    'Private key': (b'-----begin ',),
                    'Credential in URL': (b'http://', b'https://'),
                    'Assigned secret': (b'password', b'api_key', b'access_token', b'client_secret'),
                }
                for label, pattern in rules.items():
                    if any(prefix in lowered for prefix in prefixes[label]) and re.search(pattern, data):
                        findings.append((p.relative_to(ROOT).as_posix(), label))
                overlap = data[-256:]
    assert not findings, 'Potential secrets (values withheld): ' + str(sorted(set(findings)))

    base = 'results/thesis/thesis_flux_correction_v4/'
    events = read(base + 'event_metrics.tsv')
    summaries = read(base + 'summary.tsv')
    for summary in summaries:
        group = [r for r in events if all(r[k] == summary[k] for k in ['stage', 'condition', 'model', 'from_hours', 'to_hours'])]
        assert len(group) == 54
        for metric in ['mse_h', 'mse_u', 'anomaly_mse']:
            close(equal_event_mean(group, metric), float(summary[metric]))
    jobs = read(base + 'run_index.tsv')
    for job in jobs:
        directory = ROOT / job['directory']
        assert directory.is_relative_to(ROOT)
        for name in ['result.jld2', 'best_model.bson', 'training_history.tsv', 'diagnostics/metrics.tsv']:
            assert (directory / name).is_file(), str(directory / name)

    follow = 'results/thesis/reviewer_followups_initialfix_v1/'
    fevents = read(follow + 'event_metrics.tsv')
    fsummaries = read(follow + 'summary.tsv')
    allowed = {'gauge_one', 'curriculum_fast', 'gauges_two', 'gauges_four', 'multi128', 'curriculum_original'}
    assert {r['case'] for r in fsummaries} <= allowed
    for summary in fsummaries:
        group = [r for r in fevents if all(r[k] == summary[k] for k in ['case', 'model', 'from_hours', 'to_hours'])]
        assert len(group) == 54
        close(math.sqrt(equal_event_mean(group)), float(summary['rmse_h']))

    source = 'results/thesis/thesis_source_temporal_controls/'
    sevents = read(source + 'event_metrics.tsv')
    ssummaries = read(source + 'summary.tsv')
    assert {r['objective'] for r in sevents} == {'long128'}
    for summary in ssummaries:
        group = [r for r in sevents if all(r[k] == summary[k] for k in ['objective', 'model', 'from_hours', 'to_hours'])]
        assert len(group) == 54
        close(equal_event_mean(group), float(summary['mse_h']))
    target = 'results/thesis/source_target_evaluation/'
    tevents = read(target + 'event_metrics.tsv')
    tsummaries = read(target + 'summary.csv', ',')
    for summary in tsummaries:
        group = [r for r in tevents if r['model'] == summary['model'] and float(r['from_hours']) == float(summary['from_hours'])]
        assert len(group) == 54
        close(equal_event_mean(group), float(summary['mse_h']))
    for start, expected in [(0, 99.55), (48, 127.81)]:
        row = next(r for r in tsummaries if r['model'] == 'adapted' and float(r['from_hours']) == start)
        assert round(1000 * math.sqrt(float(row['mse_h'])), 2) == expected

    checksums = ROOT / 'docs/file_checksums.sha256'
    if checksums.is_file():
        for line in checksums.read_text().splitlines():
            digest, rel = line.split('  ', 1)
            path = ROOT / rel
            assert path.is_file(), 'Missing inventoried file: ' + rel
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()
            assert actual == digest, 'File changed since migration audit: ' + rel
    print(json.dumps({'status': 'passed', 'local_include_paths': includes, 'main_summary_groups': len(summaries),
                      'indexed_main_entries': len(jobs), 'distinct_main_fits': len({r['directory'] for r in jobs}),
                      'followup_summary_groups': len(fsummaries), 'source_summary_groups': len(ssummaries),
                      'target_summary_groups': len(tsummaries), 'credential_pattern_findings': len(findings),
                      'non_LFS_files_over_100_MiB': oversize}, indent=2))


if __name__ == '__main__':
    main()
