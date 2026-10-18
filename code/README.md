# CV Selection Camp: Learning with Noisy Labels

Salaj Sonar | Roll number: 24B2133

## Results at a glance

Classification accuracy below is measured against the indicated validation
annotations. Test labels are withheld. Each checkpoint was selected using
the permitted noisy labels; Task 1 exports the final epoch as required.

| Experiment | Validation labels | Checkpoint | Accuracy |
|---|---|---:|---:|
| Clean-label reference (Task 1) | Human noisy | 60 | 59.18% |
| Noisy cross-entropy (Task 1) | Human noisy | 60 | 50.18% |
| Early-stopped cross-entropy (Task 2) | Human noisy | 43 | 55.94% |
| ELR (Task 4) | Human noisy | 46 | 58.06% |
| Generalized cross-entropy (Task 4) | Human noisy | 55 | 56.82% |
| Label smoothing (Task 4) | Human noisy | 36 | 56.20% |
| Synthetic cross-entropy (Task 5) | Synthetic noisy | 41 | 49.88% |
| Synthetic selected method (Task 5) | Synthetic noisy | 56 | 53.26% |

The ELR seed comparison gave 58.06% for seed 2133, 57.66% for seed 2134 and 57.82% for their equal-probability ensemble. The selected configuration was `original`.

### Noise detection

| Fixed score | AUROC | Precision at 18,109 |
|---|---:|---:|
| Mean loss, epochs 1–10 | 0.8535 | 71.29% |
| Final training loss | 0.7493 | 62.72% |
| Out-of-fold score (submitted) | 0.9280 | 81.31% |

Each training image is scored by a fold model that did not train on it.
Three disjoint folds cover all 45,000 training indices. Each fold model
uses 30,000 training images and 30 epochs, with its checkpoint selected
on the separate noisy validation split. The score is
`-log(max(p(noisy_label | image), 1e-8))`.

The clean-label comparison is diagnostic only: it did not select the
score, set a filtering threshold, or influence any classifier.

### Files and implementation

- [Report answers](../report.md): Q1–Q5, each under 100 words.
- [Training dynamics](../analysis/training_dynamics.png): memorization and checkpoint selection.
- [Method comparison](../analysis/robust_comparison.png): classification results.
- [Noise detection](../analysis/detection_comparison.png): ROC curves and precision.
- [Noise matrix](../analysis/noise_confusion_matrix.png): class transitions.
- `engine.py`: data preparation, objectives, training and checkpointing.
- `run_all.py` and `extend_elr.py`: experiment sequence and seed comparison.
- `finalize.py` and `evidence.py`: reports, figures, prediction export and validation.
- `analysis/experiment_manifest.json`: training settings and selected epochs.
- `analysis/sha256.json`: checksums for the submitted files.

Saved weights are not included in the submission. The commands below train
from scratch and retain checkpoints in the specified runs directory.

The comparison uses one supplied validation split. Differences between
selected checkpoints do not establish hidden-test accuracy.

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
python code/run_all.py --data ../provided/data --runs ../reproduction_runs --submission . --analysis analysis --name "Salaj Sonar" --roll 24B2133 --batch-size 128 --epochs 60 --seed 2133 --folds 3 --fold-epochs 30 --official-check ../provided/check_format.py
python code/extend_elr.py --data ../provided/data --runs ../reproduction_runs --submission . --analysis analysis --name "Salaj Sonar" --roll 24B2133 --official-check ../provided/check_format.py
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
python code/train.py --data ../provided/data --runs ../reproduction_runs --run noisy_baseline --method ce --label noisy_label --epochs 60 --batch-size 128 --seed 2133 --dynamics
```

Task 1 clean reference:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run clean_reference --method ce --label clean_label --epochs 60 --batch-size 128 --seed 2133
```

Task 3 uses out-of-fold assigned-label negative log probability from three 30-epoch folds.
The full-suite command reproduces its scoring policy. Additional Task 3 training
runs in this submission: 3. Clean-label AUROC and precision audits are
post-hoc diagnostics and do not choose the submitted scoring rule.

Task 4 methods:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run robust_elr --method elr --label noisy_label --epochs 60 --batch-size 128 --seed 2133
python code/train.py --data ../provided/data --runs ../reproduction_runs --run robust_gce --method gce --label noisy_label --epochs 60 --batch-size 128 --seed 2133
python code/train.py --data ../provided/data --runs ../reproduction_runs --run robust_label_smoothing --method label_smoothing --label noisy_label --epochs 60 --batch-size 128 --seed 2133
```

Task 5 baseline and selected method (elr); method parameters are frozen:

```
python code/train.py --data ../provided/data --runs ../reproduction_runs --run synthetic_baseline --method ce --label synthetic_label --epochs 60 --batch-size 128 --seed 2133
python code/train.py --data ../provided/data --runs ../reproduction_runs --run synthetic_best --method elr --label synthetic_label --epochs 60 --batch-size 128 --seed 2133
```

To regenerate only packaging/report/plots from completed run checkpoints:

```
python code/extend_elr.py --data ../provided/data --runs ../reproduction_runs --submission . --analysis analysis --name "Salaj Sonar" --roll 24B2133 --official-check ../provided/check_format.py --finalize-only
python ../provided/check_format.py .
```

## Shared training and selection rules

SGD: learning rate 0.1, momentum 0.9, Nesterov, weight decay 0.0005. Cosine
schedule over each run's epoch budget. Reflection padding by four pixels,
random 32x32 crop and horizontal flip with probability 0.5. Normalization is
computed from the 45,000 training images alone. All submitted classifiers have
the same 60-epoch budget per network. Base runs use seed 2133. The ELR extension uses a second seed, 2134, and at most two networks per ELR predictor.
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

## ELR seed and ensemble comparison

The full-suite commands above first reproduce the original experiments, then
the extension. Human ELR seeds are 2133 and 2134, each trained from scratch for
60 epochs with beta=0.7 and lambda=3. Each network uses its own highest human
noisy-validation checkpoint. Candidates, in tie order, are the original seed,
the second seed alone, and the fixed arithmetic mean of their class probabilities.
No weights, temperatures, extra seeds or clean-label criteria are searched.
The selected policy was `original`. It is reproduced on synthetic
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
