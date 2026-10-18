"""Serial, resumable experiment queue; never uses clean accuracy for selection."""
import argparse
import ctypes
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from engine import Data, atomic_json, train


def run_suite(args):
    root = Path(args.runs).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(args.submission).resolve()
    analysis = Path(args.analysis).resolve()
    analysis.mkdir(parents=True, exist_ok=True)
    status = analysis / 'status.json'
    atomic_json(status, {'status': 'loading_data', 'pid': os.getpid(), 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    data = Data(args.data)
    summaries = {}
    specs = [('noisy_baseline', 'ce', 'noisy_label'),
             ('clean_reference', 'ce', 'clean_label'),
             ('robust_elr', 'elr', 'noisy_label'),
             ('robust_gce', 'gce', 'noisy_label'),
             ('robust_label_smoothing', 'label_smoothing', 'noisy_label')]
    for name, method, label in specs:
        atomic_json(status, {'status': 'training', 'current_run': name, 'completed_runs': list(summaries), 'pid': os.getpid()})
        summaries[name] = train(data, root, name, method, label, args.seed, args.epochs,
                                args.batch_size, dynamics=name == 'noisy_baseline')
        atomic_json(analysis / 'run_summary.json', summaries)
    # Choice is exclusively the best HUMAN-NOISY validation accuracy. Ties use queue order.
    candidates = [s for n, s in summaries.items() if n.startswith('robust_')]
    best = max(candidates, key=lambda s: s['best_acc'])
    atomic_json(analysis / 'selection.json', {'best_method': best['method'], 'selection_label': 'noisy_label',
                                             'validation_accuracy_percent': best['best_acc'],
                                             'checkpoint_epoch': best['best_epoch'],
                                             'clean_labels_used_for_selection': False})
    for name, method in [('synthetic_baseline', 'ce'), ('synthetic_best', best['method'])]:
        atomic_json(status, {'status': 'training', 'current_run': name, 'completed_runs': list(summaries), 'pid': os.getpid()})
        summaries[name] = train(data, root, name, method, 'synthetic_label', args.seed,
                                args.epochs, args.batch_size)
        atomic_json(analysis / 'run_summary.json', summaries)
    # Package the core runs before computing cross-validation scores.
    # Task 3 signal is predefined early loss until ALL cross-validation folds finish.
    from finalize import finalize
    finalize(args.data, root, out, analysis, args.name, args.roll, summaries,
             official_check=args.official_check, use_oof=False)
    atomic_json(status, {'status': 'core_submission_verified', 'completed_runs': list(summaries),
                         'best_method': best['method'], 'pid': os.getpid()})
    if args.folds:
        assert 2 <= args.folds <= 5
        # Do not launch optional cross-validation if it threatens the submission buffer.
        deadline = dt.datetime.fromisoformat(args.deadline).timestamp() if args.deadline else float('inf')
        completed = [s for s in summaries.values() if s['elapsed_seconds'] > 0]
        # Conservative estimate includes the slower full-baseline diagnostics and a 4-hour buffer.
        duration = max(s['elapsed_seconds'] for s in completed) * .75 * args.fold_epochs / args.epochs
        remaining_folds = sum(not (root / f'noise_fold_{k+1}' / 'complete.json').exists() for k in range(args.folds))
        enough_time = time.time() + remaining_folds * duration + 4 * 3600 < deadline
        if not enough_time:
            atomic_json(analysis / 'cross_validation.json', {'status': 'omitted_to_preserve_deadline',
                        'required_estimated_seconds': remaining_folds * duration + 4*3600,
                        'remaining_seconds': deadline-time.time(), 'submitted_score': 'early_loss_epochs_1_to_10'})
        else:
            rng = np.random.default_rng(args.seed)
            shuffled = rng.permutation(data.train)
            folds = np.array_split(shuffled, args.folds)
            for k, heldout in enumerate(folds):
                training = np.concatenate([f for j, f in enumerate(folds) if j != k])
                name = f'noise_fold_{k+1}'
                atomic_json(status, {'status': 'training', 'current_run': name, 'completed_runs': list(summaries), 'pid': os.getpid()})
                summaries[name] = train(data, root, name, 'ce', 'noisy_label', args.seed + k + 1,
                                       args.fold_epochs, args.batch_size, training_indices=training,
                                       score_indices=heldout)
                atomic_json(analysis / 'run_summary.json', summaries)
            finalize(args.data, root, out, analysis, args.name, args.roll, summaries,
                     official_check=args.official_check, use_oof=True)
            atomic_json(analysis / 'cross_validation.json', {'status': 'complete', 'runs': args.folds,
                                                            'submitted_score': 'out_of_fold_negative_log_probability'})
    atomic_json(status, {'status': 'complete_verified', 'completed_runs': list(summaries),
                         'best_method': best['method'], 'pid': os.getpid(),
                         'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    print('Complete: submission folder and archive passed format validation.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--submission', required=True)
    p.add_argument('--analysis', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--roll', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=2133)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--fold-epochs', type=int, default=30)
    p.add_argument('--deadline', default='')
    p.add_argument('--official-check', required=True)
    args = p.parse_args()
    try:
        run_suite(args)
    except BaseException as exc:
        Path(args.analysis).mkdir(parents=True, exist_ok=True)
        atomic_json(Path(args.analysis) / 'status.json', {'status': 'failed', 'error': repr(exc), 'pid': os.getpid()})
        traceback.print_exc()
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
