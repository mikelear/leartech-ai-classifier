"""Eval CronJob — run test cases against the deployed classifier daily.

Clones leartech-llm-training-data for test cases and baseline,
calls the classifier /predict endpoint, compares results.
Posts a GitHub Issue if accuracy drops below floor or regressions found.

Usage:
  python app/cron_eval.py \
    --endpoint http://leartech-ai-classifier:8080 \
    --repo mikelear/leartech-llm-training-data \
    --accuracy-floor 0.8
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def clone_repo(repo_url: str, target: str) -> bool:
    """Clone a repo. Returns True on success."""
    result = subprocess.run(
        ['git', 'clone', '--depth', '1', repo_url, target],
        capture_output=True,
    )
    return result.returncode == 0


def predict(endpoint: str, diff: str) -> dict:
    """Call classifier /predict endpoint."""
    payload = json.dumps({'diff': diff}).encode()
    req = urllib.request.Request(
        f'{endpoint}/predict',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def main() -> None:
    """Run eval suite against deployed classifier."""
    parser = argparse.ArgumentParser(description='Eval CronJob')
    parser.add_argument('--endpoint', required=True, help='Classifier endpoint URL')
    parser.add_argument('--repo', default='mikelear/leartech-llm-training-data')
    parser.add_argument('--accuracy-floor', type=float, default=0.8)
    parser.add_argument('--cluster-id', default='unknown')
    args = parser.parse_args()

    # Token from env (optional — needed for posting Issues on failure)
    token = os.environ.get('GIT_TOKEN', '')

    # Health check
    try:
        health = json.loads(urllib.request.urlopen(f'{args.endpoint}/health', timeout=5).read())
        print(f"Classifier: {health.get('status')} | params={health.get('parameters')} | accuracy={health.get('accuracy')}")
    except Exception as e:
        print(f'ERROR: Classifier unreachable at {args.endpoint}: {e}')
        sys.exit(1)

    # Clone test cases (public repo — token optional for clone, needed for Issues)
    clone_url = f'https://x-access-token:{token}@github.com/{args.repo}.git' if token else f'https://github.com/{args.repo}.git'
    evals_dir = Path('/tmp/eval-data')

    if not clone_repo(clone_url, str(evals_dir)):
        print('ERROR: Failed to clone training data repo')
        sys.exit(1)

    # Load manifest + baseline
    manifest_path = evals_dir / 'evals' / 'manifest.yaml'
    baseline_path = evals_dir / 'evals' / 'baseline.json'

    if not manifest_path.exists():
        print('ERROR: manifest.yaml not found')
        sys.exit(1)

    # Simple YAML parsing (avoid pyyaml dependency)
    test_cases = []
    current = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if line.startswith('- file:'):
            if current:
                test_cases.append(current)
            current = {'file': line.split(':', 1)[1].strip()}
        elif line.startswith('verdict:') and current:
            current['verdict'] = line.split(':', 1)[1].strip()
    if current:
        test_cases.append(current)

    print(f'\nLoaded {len(test_cases)} test cases')

    # Load baseline
    baseline_map = {}
    if baseline_path.exists():
        for b in json.loads(baseline_path.read_text()):
            baseline_map[b['file']] = b

    # Run predictions
    results = []
    for tc in test_cases:
        diff_path = evals_dir / 'evals' / tc['file']
        if not diff_path.exists():
            print(f'  SKIP: {tc["file"]} not found')
            continue

        diff = diff_path.read_text()
        pred = predict(args.endpoint, diff)
        match = pred['verdict'] == tc['verdict']

        results.append({
            'file': tc['file'],
            'expected': tc['verdict'],
            'actual': pred['verdict'],
            'probability': pred['probability'],
            'match': match,
        })

        icon = '✓' if match else '✗'
        print(f'  {icon} {tc["file"]:50s} expected={tc["verdict"]:4s} actual={pred["verdict"]:4s} prob={pred["probability"]:.3f}')

    # Calculate metrics
    accuracy = sum(r['match'] for r in results) / len(results) if results else 0
    regressions = []
    improvements = []

    for r in results:
        b = baseline_map.get(r['file'])
        if not b:
            continue
        if b.get('match') and not r['match']:
            regressions.append(r['file'])
        elif not b.get('match') and r['match']:
            improvements.append(r['file'])

    print(f'\n{"="*60}')
    print(f'Eval Results [{args.cluster_id}]')
    print(f'  Accuracy:     {accuracy:.0%} ({sum(r["match"] for r in results)}/{len(results)})')
    print(f'  Floor:        {args.accuracy_floor:.0%}')
    print(f'  Improvements: {len(improvements)}')
    print(f'  Regressions:  {len(regressions)}')

    # Gate check
    failed = False
    reasons = []

    if accuracy < args.accuracy_floor:
        failed = True
        reasons.append(f'Accuracy {accuracy:.0%} < floor {args.accuracy_floor:.0%}')

    if regressions:
        failed = True
        reasons.append(f'{len(regressions)} regression(s): {", ".join(regressions)}')

    if failed:
        print(f'\n  EVAL FAILED:')
        for r in reasons:
            print(f'    ✗ {r}')

        # Post GitHub Issue if token available
        if token and args.repo:
            _post_issue(token, args.repo, accuracy, regressions, improvements, args.cluster_id)
    else:
        print(f'\n  EVAL PASSED ✓')


def _post_issue(token: str, repo: str, accuracy: float, regressions: list, improvements: list, cluster: str) -> None:
    """Post or update a GitHub Issue for eval failures."""
    issue_label = 'eval-regression'
    classifier_repo = 'mikelear/leartech-ai-classifier'

    # Check for existing open issue
    check_url = f'https://api.github.com/repos/{classifier_repo}/issues?labels={issue_label}&state=open&per_page=1'
    req = urllib.request.Request(check_url, headers={
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    })
    existing = json.loads(urllib.request.urlopen(req).read())

    title = f'[eval] Classifier accuracy dropped to {accuracy:.0%}'
    body = f"""## Classifier Eval Regression [{cluster}]

Daily eval detected a problem with the deployed classifier.

**Accuracy:** {accuracy:.0%}
**Regressions:** {len(regressions)}

### Regressions (were correct, now wrong)
{"".join(f"- {r}" + chr(10) for r in regressions) if regressions else "None"}

### Improvements
{"".join(f"- {i}" + chr(10) for i in improvements) if improvements else "None"}

---
*Generated by classifier eval CronJob [{cluster}]*
"""

    if existing:
        # Update existing
        issue_num = existing[0]['number']
        req = urllib.request.Request(
            f'https://api.github.com/repos/{classifier_repo}/issues/{issue_num}',
            data=json.dumps({'title': title, 'body': body}).encode(),
            headers={
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json',
            },
            method='PATCH',
        )
        urllib.request.urlopen(req)
        print(f'  Updated issue #{issue_num}')
    else:
        # Create new
        req = urllib.request.Request(
            f'https://api.github.com/repos/{classifier_repo}/issues',
            data=json.dumps({'title': title, 'body': body, 'labels': [issue_label]}).encode(),
            headers={
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json',
            },
            method='POST',
        )
        resp = json.loads(urllib.request.urlopen(req).read())
        print(f'  Created issue #{resp["number"]}')


if __name__ == '__main__':
    main()
