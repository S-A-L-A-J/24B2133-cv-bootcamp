"""Generate result summaries and post-hoc detection plots from completed runs."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def detection_figure(wrong, early, final, submitted, use_oof, destination):
    """Clean labels are used only to evaluate the fixed Task 3 scores."""
    signals = [('Final training loss', final, '#8b96a5'),
               ('Mean loss, epochs 1–10', early, '#b76a24')]
    if use_oof:
        signals.append(('Out-of-fold score', submitted, '#207b7f'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    k = int(wrong.sum())
    precisions = []
    for label, score, color in signals:
        fpr, tpr, _ = roc_curve(wrong, score)
        area = roc_auc_score(wrong, score)
        axes[0].plot(fpr, tpr, color=color, linewidth=2,
                     label=f'{label} ({area:.4f})')
        order = np.argsort(-score, kind='stable')
        precisions.append(100 * float(wrong[order[:k]].mean()))
    axes[0].plot([0, 1], [0, 1], '--', color='#666666', linewidth=1)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate',
                title='ROC curves (AUROC in legend)', xlim=(0, 1), ylim=(0, 1.01))
    axes[0].legend(loc='lower right', fontsize=8)
    bars = axes[1].bar(range(len(signals)), precisions,
                       color=[item[2] for item in signals], width=.65)
    axes[1].bar_label(bars, fmt='%.2f%%', padding=4)
    axes[1].set_xticks(range(len(signals)),
                       [label.replace(' ', '\n', 1) for label, _, _ in signals])
    axes[1].axhline(100 * wrong.mean(), color='#666666', linestyle='--',
                    label=f'Random ranking: {100 * wrong.mean():.2f}%')
    axes[1].set(ylabel='Precision (%)', title=f'Highest-scoring {k:,} images',
                ylim=(0, 100))
    axes[1].legend(loc='upper left', fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Task 3: post-hoc evaluation of fixed noise scores', fontsize=13)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def add_results_overview(folder, summaries, extension, audits, submitted, use_oof):
    """Place the measured outcomes and artifact guide before reproduction details."""
    folder = Path(folder)
    lines = [
        '## Results at a glance', '',
        'Classification accuracy below is measured against the indicated validation',
        'annotations. Test labels are withheld. Each checkpoint was selected using',
        'the permitted noisy labels; Task 1 exports the final epoch as required.', '',
        '| Experiment | Validation labels | Checkpoint | Accuracy |',
        '|---|---|---:|---:|',
    ]
    rows = [
        ('Clean-label reference (Task 1)', 'clean_reference', 'Human noisy', True),
        ('Noisy cross-entropy (Task 1)', 'noisy_baseline', 'Human noisy', True),
        ('Early-stopped cross-entropy (Task 2)', 'noisy_baseline', 'Human noisy', False),
        ('ELR (Task 4)', 'robust_elr', 'Human noisy', False),
        ('Generalized cross-entropy (Task 4)', 'robust_gce', 'Human noisy', False),
        ('Label smoothing (Task 4)', 'robust_label_smoothing', 'Human noisy', False),
        ('Synthetic cross-entropy (Task 5)', 'synthetic_baseline', 'Synthetic noisy', False),
        ('Synthetic selected method (Task 5)', 'synthetic_best', 'Synthetic noisy', False),
    ]
    for title, key, labels, final in rows:
        run = summaries[key]
        epoch = run['epochs'] if final else run['best_epoch']
        accuracy = run['final_acc'] if final else run['best_acc']
        lines.append(f'| {title} | {labels} | {epoch} | {accuracy:.2f}% |')
    if extension:
        scores = extension['human_validation_scores']
        lines += ['', 'The ELR seed comparison gave '
                  f"{scores['original']:.2f}% for seed 2133, "
                  f"{scores['second_seed']:.2f}% for seed 2134 and "
                  f"{scores['ensemble']:.2f}% for their equal-probability ensemble. "
                  f"The selected configuration was `{extension['choice']}`."]
    lines += ['', '### Noise detection', '',
              f'| Fixed score | AUROC | Precision at {submitted["k"]:,} |',
              '|---|---:|---:|']
    labels = {'early_loss_epochs_1_to_10': 'Mean loss, epochs 1–10',
              'final_epoch_loss': 'Final training loss',
              'out_of_fold_negative_log_probability': 'Out-of-fold score (submitted)'}
    for key, values in audits.items():
        lines.append(f"| {labels[key]} | {values['auroc']:.4f} | "
                     f"{100 * values['precision_at_k']:.2f}% |")
    if use_oof:
        lines += ['', 'Each training image is scored by a fold model that did not train on it.',
                  'Three disjoint folds cover all 45,000 training indices. Each fold model',
                  'uses 30,000 training images and 30 epochs, with its checkpoint selected',
                  'on the separate noisy validation split. The score is',
                  '`-log(max(p(noisy_label | image), 1e-8))`.', '',
                  'The clean-label comparison is diagnostic only: it did not select the',
                  'score, set a filtering threshold, or influence any classifier.']
    lines += ['', '### Files and implementation', '',
              '- [Report answers](../report.md): Q1–Q5, each under 100 words.',
              '- [Training dynamics](../analysis/training_dynamics.png): memorization and checkpoint selection.',
              '- [Method comparison](../analysis/robust_comparison.png): classification results.',
              '- [Noise detection](../analysis/detection_comparison.png): ROC curves and precision.',
              '- [Noise matrix](../analysis/noise_confusion_matrix.png): class transitions.',
              '- `engine.py`: data preparation, objectives, training and checkpointing.',
              '- `run_all.py` and `extend_elr.py`: experiment sequence and seed comparison.',
              '- `finalize.py` and `evidence.py`: reports, figures, prediction export and validation.',
              '- `analysis/experiment_manifest.json`: training settings and selected epochs.',
              '- `analysis/sha256.json`: checksums for the submitted files.', '',
              'Saved weights are not included in the submission. The commands below train',
              'from scratch and retain checkpoints in the specified runs directory.', '',
              'The comparison uses one supplied validation split. Differences between',
              'selected checkpoints do not establish hidden-test accuracy.', '',
              '## Environment and supplied data']
    path = folder / 'code/README.md'
    text = path.read_text(encoding='utf-8')
    text = text.replace('## Environment and supplied data', '\n'.join(lines), 1)
    path.write_text(text, encoding='utf-8')
