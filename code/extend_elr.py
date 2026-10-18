"""Compare two ELR seeds and their equal-probability ensemble.

Only noisy validation labels choose the human candidate. Synthetic runs reproduce
the chosen human policy. No clean label or Task 3 audit metric enters this module.
"""
import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
from engine import Data, ResNet18, atomic_json, probabilities, train

BASE_NAMES = ['noisy_baseline', 'clean_reference', 'robust_elr', 'robust_gce',
              'robust_label_smoothing', 'synthetic_baseline', 'synthetic_best',
              'noise_fold_1', 'noise_fold_2', 'noise_fold_3']
CHOICES = ['original', 'second_seed', 'ensemble']


def candidate_scores(first, second, labels):
    assert first.shape == second.shape == (len(labels), 10)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    arrays = [first, second, (first + second) * .5]
    # Integer counts prevent floating-point artifacts from changing ties.
    counts = [int((p.argmax(1) == labels).sum()) for p in arrays]
    winner = max(range(3), key=lambda i: counts[i])
    return CHOICES[winner], dict(zip(CHOICES, [100 * n / len(labels) for n in counts]))


def get_probs(data, runs, name, test=False):
    """Persist probabilities with a signature of the checkpoint being evaluated."""
    import hashlib
    checkpoint = runs/name/'best.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path = runs/name/('extension_test_probs.npy' if test else 'extension_val_probs.npy')
    signature = path.with_suffix('.json')
    if path.exists() and signature.exists() and json.loads(signature.read_text())['checkpoint_sha256'] == digest:
        return np.load(path)
    net = ResNet18().to(data.device, memory_format=torch.channels_last)
    net.load_state_dict(torch.load(checkpoint, map_location=data.device, weights_only=True))
    ids = np.arange(10000) if test else data.val
    p = probabilities(net, data, ids, test=test).cpu().numpy()
    if test:
        assert np.array_equal(p.argmax(1), np.load(runs/name/'test_best.npy'))
    del net
    np.save(path, p)
    atomic_json(signature, {'checkpoint_sha256': digest, 'test': test})
    return p


def selected_summary(data, runs, first, second, choice, label, combined_name):
    if choice == 'original':
        result = dict(first)
        result.update(variant='single_seed_2133', network_count=1, member_runs=[first['name']], member_seeds=[2133])
        return result
    if choice == 'second_seed':
        result = dict(second)
        result.update(variant='single_seed_2134', network_count=1, member_runs=[second['name']], member_seeds=[2134])
        return result
    assert choice == 'ensemble'
    valp = (get_probs(data, runs, first['name']) + get_probs(data, runs, second['name'])) * .5
    y = np.array([int(row[label]) for row in data.rows])[data.val]
    testp = (get_probs(data, runs, first['name'], True) + get_probs(data, runs, second['name'], True)) * .5
    directory = runs/combined_name
    directory.mkdir(exist_ok=True)
    np.save(directory/'test_best.npy', testp.argmax(1))
    result = dict(first)
    result.update(name=combined_name, variant='equal_probability_ensemble', network_count=2,
                  member_runs=[first['name'], second['name']], member_seeds=[2133, 2134],
                  member_epochs=[first['best_epoch'], second['best_epoch']],
                  weights=[.5, .5], best_acc=float((valp.argmax(1) == y).mean()*100),
                  best_epoch=[first['best_epoch'], second['best_epoch']], final_acc=None,
                  elapsed_seconds=first['elapsed_seconds']+second['elapsed_seconds'],
                  seed=[2133, 2134])
    # This derived prediction artifact is not an additional trained network/run.
    atomic_json(directory/'ensemble.json', result)
    return result


def guard_run(runs, name, label):
    settings = runs/name/'settings.json'
    if settings.exists():
        s = json.loads(settings.read_text())
        assert (s['method'], s['training_label'], s['seed'], s['epochs'], s['batch_size']) == ('elr', label, 2134, 60, 128), s


def run_extension(args):
    runs, analysis, folder = map(lambda p: Path(p).resolve(), (args.runs, args.analysis, args.submission))
    analysis.mkdir(parents=True, exist_ok=True)
    base = {n: json.loads((runs/n/'complete.json').read_text()) for n in BASE_NAMES}
    for n in ['robust_elr', 'synthetic_best']:
        assert base[n]['method'] == 'elr' and base[n]['seed'] == 2133 and base[n]['epochs'] == 60
    status_file = analysis/'status.json'
    completed = list(BASE_NAMES)
    def status(state, **kw):
        atomic_json(status_file, dict(status=state, phase='elr_extension', pid=os.getpid(), completed_runs=completed, **kw))
    policy = {'second_seed':2134, 'epochs_per_network':60, 'batch_size':128,
              'candidates_in_tie_order':CHOICES, 'ensemble_weights':[.5,.5],
              'selection_label':'noisy_label', 'clean_label_usage':'none',
              'synthetic_policy':'Reproduce the human-selected configuration; no synthetic policy re-selection.',
              'maximum_additional_networks':2, 'task3_unchanged':True}
    policy_path = runs/'extension_policy.json'
    if policy_path.exists():
        assert json.loads(policy_path.read_text()) == policy, 'Policy changed after experiments started'
    else:
        atomic_json(policy_path, policy)
    status('loading_data')
    data = Data(args.data)
    # No new seed/hyperparameter trials beyond the declared candidates.
    if args.finalize_only:
        second = json.loads((runs/'elr_seed2/complete.json').read_text())
    else:
        guard_run(runs, 'elr_seed2', 'noisy_label')
        status('training', current_run='elr_seed2')
        second = train(data, runs, 'elr_seed2', 'elr', 'noisy_label', 2134, 60, 128)
    completed.append('elr_seed2')
    status('evaluating_extension')
    p1 = get_probs(data, runs, 'robust_elr')
    p2 = get_probs(data, runs, 'elr_seed2')
    y = np.array([int(row['noisy_label']) for row in data.rows])[data.val]
    choice, scores = candidate_scores(p1, p2, y)
    assert abs(scores['original']-base['robust_elr']['best_acc']) < 1e-4
    assert abs(scores['second_seed']-second['best_acc']) < 1e-4
    extension = dict(policy=policy, choice=choice, human_validation_scores=scores,
                     human_runs=[base['robust_elr'], second], synthetic_runs=[base['synthetic_best']])
    atomic_json(analysis/'extension_selection.json', extension)
    print(f'ELR extension selection: {choice}; noisy validation: {scores}', flush=True)
    selected_human = selected_summary(data, runs, base['robust_elr'], second, choice, 'noisy_label', 'elr_combined')
    if choice == 'original':
        selected_synth = selected_summary(data, runs, base['synthetic_best'], None, choice, 'synthetic_label', 'synthetic_elr_combined')
    else:
        if args.finalize_only:
            second_synth = json.loads((runs/'synthetic_elr_seed2/complete.json').read_text())
        else:
            guard_run(runs, 'synthetic_elr_seed2', 'synthetic_label')
            status('training', current_run='synthetic_elr_seed2', selected_human_policy=choice)
            second_synth = train(data, runs, 'synthetic_elr_seed2', 'elr', 'synthetic_label', 2134, 60, 128)
        completed.append('synthetic_elr_seed2')
        extension['synthetic_runs'].append(second_synth)
        selected_synth = selected_summary(data, runs, base['synthetic_best'], second_synth, choice, 'synthetic_label', 'synthetic_elr_combined')
    extension.update(selected_human=selected_human, selected_synthetic=selected_synth)
    atomic_json(analysis/'extension_selection.json', extension)
    # Replace the ELR slot; never create a fourth submitted robust method.
    summaries = dict(base)
    summaries['robust_elr'] = selected_human
    summaries['synthetic_best'] = selected_synth
    status('packaging_extension', selected_human_policy=choice)
    del data
    torch.cuda.empty_cache()
    # Include the source used to produce the selected predictions.
    code = folder/'code'
    code.mkdir(exist_ok=True)
    for source in Path(__file__).resolve().parent.iterdir():
        if source.is_file() and source.suffix in ('.py','.txt'):
            target = code/source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
    from finalize import finalize
    finalize(args.data, runs, folder, analysis, args.name, args.roll, summaries,
             args.official_check, use_oof=True, extension=extension)
    atomic_json(analysis/'run_summary.json', summaries)
    atomic_json(analysis/'selection.json', {'best_method':'elr','selection_label':'noisy_label',
                'validation_accuracy_percent':selected_human['best_acc'],
                'checkpoint_epoch':selected_human['best_epoch'], 'variant':selected_human['variant'],
                'clean_labels_used_for_selection':False})
    status('extension_complete_verified', selected_human_policy=choice,
           finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    print('ELR comparison complete. Submission archive passed format validation.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ['data','runs','submission','analysis','name','roll','official-check']:
        p.add_argument('--'+arg, required=True)
    p.add_argument('--finalize-only', action='store_true')
    args = p.parse_args()
    try:
        run_extension(args)
    except BaseException as exc:
        atomic_json(Path(args.analysis)/'status.json', {'status':'failed','phase':'elr_extension','error':repr(exc),'pid':os.getpid()})
        traceback.print_exc()
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
