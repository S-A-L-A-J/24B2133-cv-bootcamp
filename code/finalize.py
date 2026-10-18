"""Post-hoc diagnostics, short report, integrity checks, and reproducible packaging.

Clean labels are used here ONLY for Task 1 noise statistics, Task 2 dynamics,
and the permitted Task 3 diagnostic audit. They do not select a submitted score.
"""
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from engine import atomic_json


def read_labels(data):
    with (Path(data) / 'labels.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    labels = {key: np.array([int(row[key]) for row in rows])
              for key in ['clean_label', 'noisy_label', 'synthetic_label']}
    tr = np.loadtxt(Path(data) / 'train_idx.txt', dtype=np.int64)
    return labels, tr


def noise_statistics(data, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    y, tr = read_labels(data)
    clean, noisy = y['clean_label'][tr], y['noisy_label'][tr]
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (clean, noisy), 1)
    names = (Path(data) / 'classes.txt').read_text().splitlines()
    wrong = int((clean != noisy).sum())
    pairs = sorted([(int(cm[i, j] + cm[j, i]), i, j) for i in range(10) for j in range(i+1, 10)], reverse=True)
    with (destination / 'noise_confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['clean_class / noisy_class'] + names)
        writer.writerows([[names[i]] + row.tolist() for i, row in enumerate(cm)])
    summary = {'training_images':len(tr), 'wrong_labels':wrong,
               'noise_rate':wrong/len(tr), 'orientation':'rows=clean; columns=noisy; training split only',
               'confusion_matrix':cm.tolist(),
               'top_combined_pairs':[{'pair':[names[i], names[j]], 'count':n} for n,i,j in pairs[:5]]}
    atomic_json(destination / 'noise_statistics.json', summary)
    fig, ax = plt.subplots(figsize=(10,8), layout='constrained')
    im = ax.imshow(cm / cm.sum(1, keepdims=True) * 100, cmap='Blues', vmin=0, vmax=100)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=names, yticklabels=names,
           xlabel='Human-provided label', ylabel='Clean class',
           title=f'Human annotation transitions: training split ({100*wrong/len(tr):.2f}% wrong)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if cm[i,j]/cm[i].sum()>.5 else '#182435')
    fig.colorbar(im, ax=ax, label='Percentage of clean-class images')
    fig.savefig(destination / 'noise_confusion_matrix.png', dpi=160)
    plt.close(fig)
    return summary


def write_predictions(path, values):
    values = np.asarray(values)
    assert values.shape == (10000,)
    assert np.issubdtype(values.dtype, np.integer) and ((0 <= values) & (values <= 9)).all()
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','pred'])
        writer.writerows(enumerate(values.tolist()))


def memorization(history):
    """Diagnostic operational onset, independent of checkpoint selection."""
    wrong = np.array([r['train_acc_mislabeled'] for r in history])
    clean = np.array([r['val_acc_clean'] for r in history])
    for end in range(5, len(history)):
        rise = wrong[end-2:end+1].mean() - wrong[end-5:end-2].mean()
        fall = clean[end-5:end-2].mean() - clean[end-2:end+1].mean()
        if rise >= 3 and fall >= 1:
            return history[end]['epoch'], float(rise), float(fall)
    return 0, 0., 0.


def audit_scores(scores, wrong):
    k = int(wrong.sum())
    ranked = np.argsort(-scores, kind='stable')[:k]
    return {'auroc':float(roc_auc_score(wrong, scores)),
            'precision_at_k':float(wrong[ranked].mean()), 'k':k}


def validate(folder, data, official_check):
    spec = importlib.util.spec_from_file_location('official_submission_checker', official_check)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validation = module.check_submission(str(folder), module.load_train_idx(str(data)))
    assert all(x == 1.0 for x in validation['completion'].values()), validation
    assert not validation['general'], validation
    assert not validation['warnings'], validation
    return validation


def finalize(data, runs, folder, analysis, name, roll, summaries, official_check, use_oof=False, extension=None):
    data, runs, folder, analysis = map(Path, (data,runs,folder,analysis))
    analysis.mkdir(parents=True, exist_ok=True)
    preds = folder / 'predictions'
    preds.mkdir(parents=True, exist_ok=True)
    extra = folder / 'analysis'
    extra.mkdir(parents=True, exist_ok=True)
    stats = noise_statistics(data, extra)
    for source, target, checkpoint in [('clean_reference','t1_clean_ref','final'),
                                        ('noisy_baseline','t1_noisy_ce','final'),
                                        ('noisy_baseline','t2_early_stop','best'),
                                        ('synthetic_baseline','t5_baseline_synth','best'),
                                        ('synthetic_best','t5_best_synth','best')]:
        write_predictions(preds / (target+'.csv'), np.load(runs/summaries[source]['name']/f'test_{checkpoint}.npy'))
    robust = [summaries[n] for n in ['robust_elr','robust_gce','robust_label_smoothing']]
    best = max(robust, key=lambda s:s['best_acc'])
    for method in robust:
        write_predictions(preds / f't4_{method["method"]}.csv', np.load(runs/method['name']/'test_best.npy'))
    history = json.loads((runs/'noisy_baseline/history.json').read_text())
    columns = ['epoch','train_acc_correct','train_acc_mislabeled','val_acc_noisy','val_acc_clean']
    with (preds / 't2_dynamics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(history)
    labels, tr = read_labels(data)
    wrong = labels['clean_label'][tr] != labels['noisy_label'][tr]
    early_loss = np.load(runs/'noisy_baseline/early_losses.npy')
    final_loss = np.load(runs/'noisy_baseline/final_losses.npy')
    scores = early_loss
    score_name = 'mean_noisy_label_loss_epochs_1_to_10'
    audits = {'early_loss_epochs_1_to_10':audit_scores(early_loss, wrong),
              'final_epoch_loss':audit_scores(final_loss, wrong)}
    if use_oof:
        folds = sorted(runs.glob('noise_fold_*/heldout_scores.npz'))
        assert 2 <= len(folds) <= 5
        index = np.concatenate([np.load(f)['index'] for f in folds])
        values = np.concatenate([np.load(f)['score'] for f in folds])
        assert len(index) == len(set(index)) == 45000 and set(index) == set(tr)
        lookup = dict(zip(index.tolist(), values.tolist()))
        scores = np.array([lookup[int(i)] for i in tr])
        score_name = 'out_of_fold_negative_log_noisy_label_probability'
        audits['out_of_fold_negative_log_probability'] = audit_scores(scores, wrong)
    assert scores.shape == (45000,) and np.isfinite(scores).all()
    with (preds / 't3_noise_scores.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index','score'])
        writer.writerows(zip(tr.tolist(), scores.tolist()))
    atomic_json(extra/'task3_posthoc_audit.json', {'submitted_signal':score_name,
                'selection_rule':'OOF if all predefined folds complete within time budget; otherwise early loss. Clean audit never selects the submitted signal.',
                'metrics':audits})
    onset, rise, fall = memorization(history)
    noisy = summaries['noisy_baseline']
    clean = summaries['clean_reference']
    synthce, synthbest = summaries['synthetic_baseline'], summaries['synthetic_best']
    settings = {key:noisy[key] for key in ['batch_size','epochs']}
    settings.update(optimizer=noisy['optimizer'], learning_rate_schedule=noisy['schedule'],
                    augmentation=noisy['augmentation'], gpu_used=noisy['gpu'], precision=noisy['precision'],
                    normalization=noisy['normalization'], cudnn_benchmark=True,
                    reproducibility='Seeded and epoch-resumable; GPU kernels are not guaranteed bit-exact across machines.')
    results = {'name':name,'roll_no':roll,'seed':noisy['seed'],'training_setup':settings,
        'task1':{'noise_rate':stats['noise_rate'], 'noise_rate_split':'training only',
                 'confusion_matrix_clean_rows_noisy_columns':stats['confusion_matrix'],
                 'val_acc_noisy_labels__clean_ref_model':clean['final_acc'],
                 'val_acc_noisy_labels__noisy_ce_model':noisy['final_acc']},
        'task2':{'memorization_start_epoch':onset,'early_stop_epoch':noisy['best_epoch'],
                 'val_acc_noisy_labels_at_early_stop':noisy['best_acc'],
                 'memorization_definition':'First ending epoch of a 3-epoch window with wrong-label train accuracy rising >=3 percentage points and clean val accuracy falling >=1 point relative to preceding 3 epochs. 0 means no clear onset.',
                 'train_accuracy_target':'Assigned noisy label, on both subsets',
                 'checkpoint_rule':'Highest noisy validation accuracy; earliest epoch wins ties'},
        'task3':{'method':score_name,'clean_label_usage':'Post-hoc audit only',
                 'additional_training_runs':len([n for n in summaries if n.startswith('noise_fold_')])},
        'task4':{'methods':[{'name':m['method'],'file':f't4_{m["method"]}.csv',
                            'val_acc_noisy_labels':m['best_acc'],'epoch':m['best_epoch'],
                            'network_count':m.get('network_count',1),
                            'variant':m.get('variant','single_network')} for m in robust],
                 'best_method':best['method'],'selection_rule':'Highest noisy validation accuracy; no clean-label selection'},
        'task5':{'val_acc_synthetic_labels__baseline':synthce['best_acc'],
                 'val_acc_synthetic_labels__best_method':synthbest['best_acc'],
                 'baseline_epoch':synthce['best_epoch'],'best_method_epoch':synthbest['best_epoch'],
                 'checkpoint_rule':'Highest synthetic validation accuracy, separately per model',
                 'real_comparison_baseline':'Task 2 early-stopped cross-entropy, for matching selection policy'},
        'method_parameters':{'label_smoothing':{'epsilon':.1},'gce':{'q':.7},'elr':{'beta':.7,'lambda':3}},
        'accuracy_units':'percent (0-100); noise_rate is a fraction (0-1)'}
    if extension:
        results['elr_extension'] = extension
        results['task5']['network_count_best_method'] = synthbest.get('network_count',1)
        atomic_json(extra/'extension_review.json', extension)
    atomic_json(folder/'results.json', results)
    top = stats['top_combined_pairs'][:3]
    q1 = ('The largest bidirectional human-label confusions were ' + '; '.join(
        f'{p["pair"][0]}/{p["pair"][1]} ({p["count"]:,} images)' for p in top) +
        f'. Across the 45,000 training images, {stats["wrong_labels"]:,} labels were wrong ({stats["noise_rate"]*100:.2f}%). '
        'The matrix uses clean classes as rows and human labels as columns. These concentrated transitions suggest structured mistakes, although class-level counts alone do not establish the cause of individual errors.')
    if onset:
        selected_row = next(row for row in history if row['epoch'] == noisy['best_epoch'])
        last_row = history[-1]
        q2 = (f'The diagnostic criterion first flagged epoch {onset}: wrong-label training accuracy rose '
              f'{rise:.2f} points while clean validation accuracy fell {fall:.2f} points, comparing consecutive three-epoch means. '
              'This early warning does not establish a permanent transition. '
              f'From the noisy-selected epoch {noisy["best_epoch"]} to epoch {last_row["epoch"]}, '
              f'wrong-label training accuracy rose from {selected_row["train_acc_mislabeled"]:.2f}% to {last_row["train_acc_mislabeled"]:.2f}%, '
              f'while clean validation accuracy fell from {selected_row["val_acc_clean"]:.2f}% to {last_row["val_acc_clean"]:.2f}%. '
              'That later divergence provides stronger evidence of harmful memorization. Clean accuracy was diagnostic only; early stopping used noisy validation accuracy.')
    else:
        q2 = ('No clear memorization onset met the predefined joint criterion within 60 epochs: '
              'a three-epoch wrong-label training-accuracy rise of at least three percentage points paired with a clean-validation decline of at least one point. '
              'The reported epoch 0 denotes absence of clear evidence, not an observed onset. '
              f'Early stopping selected epoch {noisy["best_epoch"]} solely by noisy validation accuracy ({noisy["best_acc"]:.2f}%). '
              'Full curves are provided rather than asserting memorization from training accuracy alone.')
    auditwinner = max(audits, key=lambda key:audits[key]['auroc'])
    winning = audits[auditwinner]
    submitted = audit_scores(scores, wrong)
    signal_titles = {'early_loss_epochs_1_to_10': 'Mean loss over epochs 1-10',
                     'final_epoch_loss': 'Final training loss',
                     'out_of_fold_negative_log_probability': 'Out-of-fold negative log probability'}
    q3 = (f'{signal_titles[auditwinner]} performed best in the post-hoc comparison '
          f'(AUROC {winning["auroc"]:.4f}). '
          f'The submitted scores achieved precision@{submitted["k"]} of {100 * submitted["precision_at_k"]:.2f}%. '
          'The score and checkpoint rules were fixed before clean-label evaluation. '
          'A difficult, correctly labeled image can receive a high score because the model assigns little probability to its class. '
          'Conversely, systematic annotation errors can receive low scores when the model learns the same confusion.')
    if use_oof:
        q3 += ' Held-out scoring reduces the influence of memorizing an image during training.'
    mechanisms = {'elr':'a temporal prediction target discourages replacing earlier consistent predictions with annotation mistakes',
                  'gce':'the loss reduces the influence of examples assigned very low probability',
                  'label_smoothing':'less absolute targets reduce pressure to become fully confident in an assigned label'}
    delta = best['best_acc'] - noisy['best_acc']
    outcome = 'improved' if delta > 0 else ('matched' if delta == 0 else 'fell below')
    q4 = (f'{best["method"]} was selected using noisy validation accuracy ({best["best_acc"]:.2f}%). '
          f'It {outcome} the equally selected early-stopped cross-entropy baseline ({noisy["best_acc"]:.2f}%), a difference of {delta:+.2f} percentage points. '
          f'Its mechanism is that {mechanisms[best["method"]]}. '
          'This is a plausible interpretation rather than proof of causality. Architecture, initialization seed, optimizer, schedule, augmentation, and 60-epoch budget were shared. Test accuracy is unavailable because test labels are withheld.')
    if extension:
        variant = best.get('variant', 'single_network').replace('_', ' ')
        final_evidence = ''
        if best.get('final_acc') is not None:
            final_evidence = (f'At epoch 60, ELR retained {best["final_acc"]:.2f}% accuracy '
                              f'versus {noisy["final_acc"]:.2f}% for cross-entropy, consistent with reduced late memorization. ')
        q4 = (f'ELR ({variant}) reached {best["best_acc"]:.2f}% noisy validation accuracy, '
              f'{delta:+.2f} points over early-stopped cross-entropy. '
              'Its temporal prediction targets resist fitting incorrect annotations. '
              + final_evidence +
              'Two seeds and their equal-probability ensemble were compared using noisy validation only. '
              'Architecture, optimizer, augmentation and schedule were shared, with 60 epochs per network. '
              'The extra seed increased training cost. These results support the mechanism but do not establish hidden-test performance.')
    realdelta = delta
    synthdelta = synthbest['best_acc'] - synthce['best_acc']
    sign = lambda x: (x > 0) - (x < 0)
    ranking = 'changed' if sign(realdelta) != sign(synthdelta) else 'did not change'
    q5 = (f'The cross-entropy versus ELR ranking {ranking}: ELR improved noisy validation accuracy by '
          f'{realdelta:+.2f} points with human labels and {synthdelta:+.2f} with synthetic labels. '
          'Only these two methods were retrained on synthetic labels; this does not establish the ranking of all robust methods. '
          'Human errors concentrate in specific class pairs, whereas synthetic corruption is uniform. '
          'Despite matched overall noise rates, these different error structures can change the benefit of regularization. '
          'The configuration selected on human-noisy validation was retained; synthetic checkpoints used synthetic validation. '
          'The two noisy validation accuracies are not directly comparable measures of clean accuracy.')
    answers = [q1,q2,q3,q4,q5]
    assert all(len(q.split()) <= 100 for q in answers), [(i+1,len(q.split())) for i,q in enumerate(answers)]
    (folder/'report.md').write_text('\n\n'.join(f'Q{i+1}. {q}' for i,q in enumerate(answers))+'\n', encoding='utf-8')
    plot_results(history, robust, summaries, extra)
    write_readme(folder, name, roll, best['method'], use_oof,
                 len([n for n in summaries if n.startswith('noise_fold_')]), noisy['batch_size'], noisy['seed'], extension)
    from evidence import detection_figure, add_results_overview
    detection_figure(wrong, early_loss, final_loss, scores, use_oof, extra/'detection_comparison.png')
    add_results_overview(folder, summaries, extension, audits, submitted, use_oof)
    # Record experiment settings and selected checkpoints.
    atomic_json(extra/'experiment_manifest.json', summaries)
    hashes = {str(p.relative_to(folder)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'sha256.json'}
    atomic_json(extra/'sha256.json', hashes)
    check = validate(folder, data, official_check)
    atomic_json(analysis/'format_check.json', check)
    archive = folder.parent/f'{roll}_cv_bootcamp.zip'
    temporary = archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                z.write(f, f.relative_to(folder).as_posix())
    temporary.replace(archive)
    # Validate the archive with the supplied checker.
    temp_root = analysis / 'checker_temp'
    temp_root.mkdir(parents=True, exist_ok=True)
    checker_env = os.environ.copy()
    checker_env.update(TEMP=str(temp_root.resolve()), TMP=str(temp_root.resolve()))
    proc = subprocess.run([sys.executable,str(Path(official_check).resolve()),str(archive.resolve())],
                          capture_output=True,text=True,encoding='utf-8',env=checker_env)
    (analysis/'official_zip_check.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    assert proc.returncode == 0 and 'All good.' in proc.stdout, proc.stdout+proc.stderr
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert json.loads(z.read('results.json')) == results
    (analysis/'results_summary.md').write_text(
        f'# Experiment results\n\nBest method: **{best["method"]}**, selected on noisy validation labels.\n\n'
        f'- Noisy baseline, final: {noisy["final_acc"]:.2f}%\n'
        f'- Noisy baseline, selected early checkpoint: {noisy["best_acc"]:.2f}% (epoch {noisy["best_epoch"]})\n'
        f'- Best robust method: {best["best_acc"]:.2f}% (epoch {best["best_epoch"]})\n'
        f'- Synthetic baseline: {synthce["best_acc"]:.2f}%\n'
        f'- Synthetic best method: {synthbest["best_acc"]:.2f}%\n'
        f'- Submitted noise detector AUROC: {submitted["auroc"]:.4f}\n'
        f'- Official ZIP checker: All good.\n\nTest labels are withheld; no test accuracy is claimed.\n',encoding='utf-8')
    print(f'VERIFIED {archive}: score={score_name}, best={best["method"]}',flush=True)


def write_readme(folder, name, roll, best_method, use_oof, folds, batch_size, seed, extension=None):
    fold_count = folds if use_oof else 0
    text = f'''# CV Selection Camp: Learning with Noisy Labels

{name} | Roll number: {roll}

## Environment and supplied data

The execution environment was Windows, Python 3.13.7, PyTorch 2.11.0+cu128,
and an RTX 3050 4 GB laptop GPU. Dependencies are pinned in requirements.txt.
No pretrained weights or external datasets are used. The supplied model.py is
byte-for-byte unchanged.

Arrange the supplied assignment files in a sibling folder named `provided`:
`../provided/data/` contains the NPY files, labels.csv, classes.txt and index files;
`../provided/check_format.py` is the unmodified supplied checker. Extract this
submission ZIP into its own folder, and run the following commands from that
folder.

```
python -m pip install -r code/requirements.txt
python code/verify_implementation.py
```

## Exact full-suite command (all Tasks 1-5)

```
python code/run_all.py --data ../provided/data --runs ../reproduction_runs --submission . --analysis analysis --name "{name}" --roll {roll} --batch-size {batch_size} --epochs 60 --seed {seed} --folds {fold_count} --fold-epochs 30 --official-check ../provided/check_format.py
```

The full-suite workflow writes results.json, report.md, all prediction files and plots,
then runs the official checker on the folder and final ZIP. Completed runs are
reused; interrupted runs resume from their last complete epoch. For a fresh
reproduction, use an empty reproduction_runs folder. Seeded experiments can
still differ numerically across GPU kernels and software/hardware; cuDNN
benchmarking is enabled and bit-exact cross-machine reproduction is not claimed.

## Commands for each task

Task 1 noisy baseline and Task 2 dynamics/early stopping share this single run:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run noisy_baseline --method ce --label noisy_label --epochs 60 --batch-size {batch_size} --seed {seed} --dynamics
```

Task 1 clean reference:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run clean_reference --method ce --label clean_label --epochs 60 --batch-size {batch_size} --seed {seed}
```

Task 3 uses {'out-of-fold assigned-label negative log probability from three 30-epoch folds' if use_oof else 'mean unaugmented assigned-label loss over epochs 1-10 of the noisy baseline'}.
The full-suite command reproduces its scoring policy. Additional Task 3 training
runs in this submission: {fold_count}. Clean-label AUROC and precision audits are
post-hoc diagnostics and do not choose the submitted scoring rule.

Task 4 methods:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run robust_elr --method elr --label noisy_label --epochs 60 --batch-size {batch_size} --seed {seed}
python code/train.py --data ../provided/data --runs ../reproduction_runs --run robust_gce --method gce --label noisy_label --epochs 60 --batch-size {batch_size} --seed {seed}
python code/train.py --data ../provided/data --runs ../reproduction_runs --run robust_label_smoothing --method label_smoothing --label noisy_label --epochs 60 --batch-size {batch_size} --seed {seed}
```

Task 5 baseline and selected method ({best_method}); method parameters are frozen:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run synthetic_baseline --method ce --label synthetic_label --epochs 60 --batch-size {batch_size} --seed {seed}
python code/train.py --data ../provided/data --runs ../reproduction_runs --run synthetic_best --method {best_method} --label synthetic_label --epochs 60 --batch-size {batch_size} --seed {seed}
```

To regenerate only packaging/report/plots from completed run checkpoints:

```
python code/finalize.py --data ../provided/data --runs ../reproduction_runs --submission . --analysis analysis --name "{name}" --roll {roll} --official-check ../provided/check_format.py{' --oof' if use_oof else ''}
python ../provided/check_format.py .
```

## Shared training and selection rules

SGD: learning rate 0.1, momentum 0.9, Nesterov, weight decay 0.0005. Cosine
schedule over each run's epoch budget. Reflection padding by four pixels,
random 32x32 crop and horizontal flip with probability 0.5. Normalization is
computed from the 45,000 training images alone. All submitted classifiers have
the same 60-epoch budget and initialization seed. Each method uses one network.
Optional Task 3 folds use 30 epochs each and disjoint held-out training images.

Robust parameters: label smoothing epsilon=0.1; GCE q=0.7; ELR beta=0.7,
lambda=3. ELR uses detached per-image exponential moving-average probabilities.
All parameters were fixed before seeing clean-label diagnostic results.

Task 1 prediction files use the FINAL epoch. Task 2 and Task 4 use the highest
human-noisy validation accuracy, with earliest-epoch tie breaking. Task 5 uses
the highest synthetic validation accuracy. Baseline-versus-robust comparisons
in the report match these best-checkpoint policies. Training accuracy on the
mislabeled subset means agreement with its assigned WRONG labels; extra plots
also show clean-label agreement for diagnosis. Validation images never train
any network. Clean labels never filter/relabel data or select models.

Noise confusion statistics use the training split, with clean rows and human
noisy columns. Memorization onset is an explicitly documented diagnostic
criterion; an epoch of 0 indicates no clear onset under that criterion.
No test accuracy is asserted: the test labels are withheld.

## Method references

- Zhang and Sabuncu, NeurIPS 2018: https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf
- Liu et al., NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/file/ea89621bee7c88b2c5be6681c8ef4906-Paper.pdf
- Wei et al., ICLR 2022 (dataset context): https://arxiv.org/abs/2110.12088
'''
    if extension:
        extension_command = (f'python code/extend_elr.py --data ../provided/data --runs ../reproduction_runs '
                             f'--submission . --analysis analysis --name "{name}" --roll {roll} '
                             '--official-check ../provided/check_format.py')
        base_command = (f'python code/run_all.py --data ../provided/data --runs ../reproduction_runs --submission . '
                        f'--analysis analysis --name "{name}" --roll {roll} --batch-size {batch_size} --epochs 60 '
                        f'--seed {seed} --folds {fold_count} --fold-epochs 30 --official-check ../provided/check_format.py')
        text = text.replace(base_command, base_command+'\n'+extension_command)
        text = text.replace('the same 60-epoch budget and initialization seed. Each method uses one network.',
                            'the same 60-epoch budget per network. Base runs use seed 2133. The ELR extension uses a second seed, 2134, and at most two networks per ELR predictor.')
        regen = (f'python code/finalize.py --data ../provided/data --runs ../reproduction_runs --submission . '
                 f'--analysis analysis --name "{name}" --roll {roll} --official-check ../provided/check_format.py --oof')
        text = text.replace(regen, extension_command+' --finalize-only')
        text += f'''
## ELR seed and ensemble comparison

The full-suite commands above first reproduce the original experiments, then
the extension. Human ELR seeds are 2133 and 2134, each trained from scratch for
60 epochs with beta=0.7 and lambda=3. Each network uses its own highest human
noisy-validation checkpoint. Candidates, in tie order, are the original seed,
the second seed alone, and the fixed arithmetic mean of their class probabilities.
No weights, temperatures, extra seeds or clean-label criteria are searched.
The selected policy was `{extension['choice']}`. It is reproduced on synthetic
labels with separately synthetic-selected component checkpoints. The synthetic
policy itself is not reselected. If the original seed wins, no second synthetic
network is needed. These are the same ELR family, not an additional Task 4 file.

`analysis/extension_review.json` records every component run, selected epoch,
candidate validation score and the deployed variant. Task 3 remains the original
three-fold detector and is not retrained. Additional runs use unique directories
`elr_seed2` and, if needed, `synthetic_elr_seed2`, so the original checkpoints are
preserved. The two-network ensemble uses twice the inference work of a single
network. Do not reuse a completed run directory for a changed configuration.

To run the second human component alone:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run elr_seed2 --method elr --label noisy_label --epochs 60 --batch-size 128 --seed 2134
```

The extension command manages the corresponding second synthetic component and
regenerates all selected predictions, reports, hashes and the verified ZIP.
'''
    (Path(folder)/'code/README.md').write_text(text,encoding='utf-8')


def plot_results(history, methods, summaries, folder):
    e = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    for key,label in [('train_acc_correct','Correct subset: assigned label'),
                      ('train_acc_mislabeled','Mislabeled subset: assigned WRONG label'),
                      ('train_acc_mislabeled_clean','Mislabeled subset: clean label (diagnostic)')]:
        axes[0].plot(e,[r[key] for r in history],label=label)
    axes[0].set(title='Training behavior (unaugmented evaluation)',xlabel='Epoch',ylabel='Accuracy (%)',ylim=(0,100))
    axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5,-.17))
    for key,label in [('val_acc_noisy','Noisy labels: checkpoint selection'),('val_acc_clean','Clean labels: diagnostic only')]:
        axes[1].plot(e,[r[key] for r in history],label=label)
    stop = summaries['noisy_baseline']['best_epoch']
    axes[1].axvline(stop,ls='--',color='#444',label=f'Selected epoch {stop}')
    axes[1].set(title='Validation behavior',xlabel='Epoch',ylabel='Accuracy (%)',ylim=(0,100))
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5,-.17))
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(folder/'training_dynamics.png',dpi=180)
    plt.close(fig)
    names = ['CE early stop'] + [m['method'] for m in methods]
    values = [summaries['noisy_baseline']['best_acc']] + [m['best_acc'] for m in methods]
    fig, ax = plt.subplots(figsize=(8,4.5),layout='constrained')
    bars = ax.bar(names,values,color=['#758397']+['#267e83']*len(methods))
    ax.bar_label(bars,fmt='%.2f%%',padding=4)
    ax.set(title='Human noisy-label validation: same checkpoint-selection policy',ylabel='Accuracy (%)',ylim=(0,100))
    ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'robust_comparison.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--runs',required=True)
    p.add_argument('--submission',required=True)
    p.add_argument('--analysis',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--roll',required=True)
    p.add_argument('--official-check',required=True)
    p.add_argument('--oof',action='store_true')
    args=p.parse_args()
    summaries={d.name:json.loads((d/'complete.json').read_text()) for d in Path(args.runs).iterdir()
               if (d/'complete.json').exists()}
    finalize(args.data,args.runs,args.submission,args.analysis,args.name,args.roll,summaries,args.official_check,args.oof)
