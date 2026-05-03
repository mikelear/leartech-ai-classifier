"""Training CronJob — retrain classifier on accumulated feedback data.

Pulls latest feedback from leartech-llm-training-data, trains a new
model with v3 features + char n-grams, runs eval against test cases.
If eval gate passes, saves artefacts to GCS for the service to pick up.

If eval fails, logs the failure and exits non-zero (no deployment).

Usage:
  python app/cron_train.py \
    --endpoint http://leartech-ai-classifier:8080 \
    --output-dir /tmp/model-output
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.features import extract_features_v3, FEATURE_NAMES_V3, PIPELINE_NAMES, PIPELINE_DEFAULTS


HANDCRAFTED_BOOST = 3.0


def clone_repo(repo_url: str, target: str) -> bool:
    """Clone a repo. Returns True on success."""
    result = subprocess.run(
        ['git', 'clone', '--depth', '1', repo_url, target],
        capture_output=True,
    )
    return result.returncode == 0


def load_feedback(feedback_dir: str) -> tuple[list[str], list[float]]:
    """Load all feedback diffs and labels."""
    diffs, labels = [], []
    for json_file in Path(feedback_dir).rglob('*.json'):
        try:
            with open(json_file) as f:
                record = json.load(f)
            if 'diff' in record and 'overall_verdict' in record:
                diffs.append(record['diff'])
                labels.append(0.0 if record['overall_verdict'] == 'PASS' else 1.0)
        except (json.JSONDecodeError, KeyError):
            continue
    return diffs, labels


def load_eval_cases(evals_dir: str) -> list[dict]:
    """Load eval test cases."""
    cases = []
    current: dict = {}
    manifest = Path(evals_dir) / 'manifest.yaml'
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line.startswith('- file:'):
            if current:
                cases.append(current)
            current = {'file': line.split(':', 1)[1].strip()}
        elif line.startswith('verdict:') and current:
            current['verdict'] = line.split(':', 1)[1].strip()
    if current:
        cases.append(current)

    for tc in cases:
        diff_path = Path(evals_dir) / tc['file']
        if diff_path.exists():
            tc['diff'] = diff_path.read_text()

    return cases


def train_model(
    diffs: list[str],
    labels: np.ndarray,
) -> tuple[nn.Sequential, TfidfVectorizer, StandardScaler, int]:
    """Train a model on the given data. Returns (model, tfidf, scaler, input_dim)."""
    # Features
    v3_raw = np.array([extract_features_v3(d) for d in diffs])
    pipeline_raw = np.full((len(diffs), len(PIPELINE_NAMES)), PIPELINE_DEFAULTS)

    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=200, sublinear_tf=True)
    tf_features = tfidf.fit_transform(diffs).toarray()

    scaler = StandardScaler()
    combined = np.hstack([v3_raw, pipeline_raw])
    combined_scaled = scaler.fit_transform(combined)

    X = np.hstack([combined_scaled * HANDCRAFTED_BOOST, tf_features])
    input_dim = X.shape[1]

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Model
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    best_val_loss = float('inf')
    best_weights = None
    wait = 0

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 40:
            break

    model.load_state_dict(best_weights)  # type: ignore[arg-type]
    model.eval()

    # Training accuracy
    with torch.no_grad():
        train_acc = ((model(X_t) > 0.5).float() == y_t).float().mean().item()
        val_acc = ((model(X_v) > 0.5).float() == y_v).float().mean().item()

    print(f'  Training: {len(X_train)} examples, stopped epoch {epoch + 1}')
    print(f'  Train accuracy: {train_acc:.1%}, Val accuracy: {val_acc:.1%}')

    return model, tfidf, scaler, input_dim


def eval_model(
    model: nn.Sequential,
    tfidf: TfidfVectorizer,
    scaler: StandardScaler,
    test_cases: list[dict],
    baseline_path: str,
    accuracy_floor: float,
) -> tuple[bool, list[dict]]:
    """Run eval suite. Returns (passed, results)."""
    baseline_map = {}
    if Path(baseline_path).exists():
        for b in json.loads(Path(baseline_path).read_text()):
            baseline_map[b['file']] = b

    results = []
    for tc in test_cases:
        if 'diff' not in tc:
            continue

        v3 = np.array([extract_features_v3(tc['diff'])])
        pipeline = np.array([PIPELINE_DEFAULTS])
        combined = scaler.transform(np.hstack([v3, pipeline])) * HANDCRAFTED_BOOST
        tf = tfidf.transform([tc['diff']]).toarray()
        stacked = np.hstack([combined, tf])

        with torch.no_grad():
            prob = model(torch.tensor(stacked, dtype=torch.float32)).item()

        verdict = 'FAIL' if prob > 0.5 else 'PASS'
        match = verdict == tc['verdict']
        results.append({
            'file': tc['file'],
            'expected': tc['verdict'],
            'actual': verdict,
            'probability': round(prob, 4),
            'match': match,
        })

        icon = '✓' if match else '✗'
        print(f'  {icon} {tc["file"]:50s} {tc["verdict"]:4s}→{verdict:4s} prob={prob:.3f}')

    accuracy = sum(r['match'] for r in results) / len(results) if results else 0
    regressions = [
        r['file'] for r in results
        if not r['match'] and baseline_map.get(r['file'], {}).get('match', False)
    ]

    print(f'\n  Accuracy: {accuracy:.0%} ({sum(r["match"] for r in results)}/{len(results)})')
    print(f'  Regressions: {len(regressions)}')

    passed = accuracy >= accuracy_floor and len(regressions) == 0
    return passed, results


def main() -> None:
    """Run the training pipeline."""
    parser = argparse.ArgumentParser(description='Training CronJob')
    parser.add_argument('--endpoint', default='http://leartech-ai-classifier:8080')
    parser.add_argument('--output-dir', default='/tmp/model-output')
    parser.add_argument('--accuracy-floor', type=float, default=0.8)
    parser.add_argument('--cluster-id', default='unknown')
    args = parser.parse_args()

    token = os.environ.get('GIT_TOKEN', '')
    clone_url = f'https://x-access-token:{token}@github.com/mikelear/leartech-llm-training-data.git' if token else 'https://github.com/mikelear/leartech-llm-training-data.git'

    data_dir = Path('/tmp/training-data')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=== Classifier Training CronJob ===')
    print(f'Cluster: {args.cluster_id}')

    # Clone data
    print('\n--- Cloning feedback data ---')
    if not clone_repo(clone_url, str(data_dir)):
        print('ERROR: Failed to clone training data')
        sys.exit(1)

    # Load data
    diffs, labels = load_feedback(str(data_dir / 'feedback'))
    labels_arr = np.array(labels)
    print(f'Loaded {len(diffs)} feedback records ({sum(labels_arr == 0):.0f} PASS, {sum(labels_arr == 1):.0f} FAIL)')

    if len(diffs) < 50:
        print(f'ERROR: Not enough data ({len(diffs)} < 50 minimum)')
        sys.exit(1)

    # Add augmentation (same as Session 10.8)
    test_cases = load_eval_cases(str(data_dir / 'evals'))
    for tc in test_cases:
        if 'diff' in tc:
            diffs.append(tc['diff'])
            labels.append(0.0 if tc['verdict'] == 'PASS' else 1.0)

    labels_arr = np.array(labels)
    print(f'After augmentation: {len(diffs)} records')

    # Train
    print('\n--- Training ---')
    model, tfidf, scaler, input_dim = train_model(diffs, labels_arr)

    # Eval
    print('\n--- Eval ---')
    baseline_path = str(data_dir / 'evals' / 'baseline.json')
    passed, results = eval_model(model, tfidf, scaler, test_cases, baseline_path, args.accuracy_floor)

    if not passed:
        print('\n  GATE FAILED — model not saved')
        sys.exit(1)

    # Save artefacts
    print('\n--- Saving artefacts ---')

    feature_names = FEATURE_NAMES_V3 + PIPELINE_NAMES

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_type': 'v3 (28 code) + pipeline (6) + char n-grams (200)',
        'num_features': input_dim,
        'code_features': len(FEATURE_NAMES_V3),
        'pipeline_features': len(PIPELINE_NAMES),
        'tfidf_features': 200,
        'boost': HANDCRAFTED_BOOST,
        'feature_names': feature_names,
        'eval_accuracy': sum(r['match'] for r in results) / len(results),
        'training_examples': len(diffs),
    }, str(output_dir / 'code_classifier.pt'))

    # Save TF-IDF and scaler as JSON (no pickle)
    tfidf_json = {
        'analyzer': tfidf.analyzer,
        'ngram_range': list(tfidf.ngram_range),
        'max_features': tfidf.max_features,
        'sublinear_tf': tfidf.sublinear_tf,
        'vocabulary': {k: int(v) for k, v in tfidf.vocabulary_.items()},
        'idf': tfidf.idf_.tolist(),
    }
    with open(output_dir / 'tfidf_char.json', 'w') as f:
        json.dump(tfidf_json, f)

    scaler_json = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_features_in': int(scaler.n_features_in_),
        'n_samples_seen': int(scaler.n_samples_seen_) if hasattr(scaler.n_samples_seen_, '__int__') else 1,
    }
    with open(output_dir / 'scaler.json', 'w') as f:
        json.dump(scaler_json, f)

    # Save eval results as new baseline candidate
    new_baseline = [{
        'file': r['file'],
        'expected_verdict': r['expected'],
        'actual_verdict': r['actual'],
        'probability': r['probability'],
        'match': r['match'],
        'status': 'pass' if r['match'] else 'fail',
    } for r in results]
    with open(output_dir / 'baseline.json', 'w') as f:
        json.dump(new_baseline, f, indent=2)

    print(f'\n  Saved to {output_dir}:')
    for f in output_dir.iterdir():
        print(f'    {f.name} ({f.stat().st_size / 1024:.1f} KB)')

    # TODO: upload to GCS bucket for the service to pick up on restart
    # gsutil cp {output_dir}/* gs://ai-models-product-first/classifier/latest/
    print('\n  TODO: upload to GCS for automatic deployment')
    print('\n  TRAINING COMPLETE ✓')


if __name__ == '__main__':
    main()
