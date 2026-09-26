# Source-Free Few-Shot Domain Adaptation for Channel-Robust RF Fingerprint Identification

**Source-Free Few-Shot Domain Adaptation for Channel-Robust Radio Frequency Fingerprint Identification**
Mingbo Jia, Jie Zhang, Tiantian Tang, Guan Gui, Tomoaki Ohtsuki, and Hikmet Sari. *IEEE Communications Letters* (under review).

This repository provides the implementation of a **class-center-guided source-free few-shot domain adaptation (SF-FSDA)** framework for **radio frequency fingerprint identification (RFFI)**. The framework adapts a source-pretrained RFFI model to an unseen target channel using only **a few labeled target samples and unlabeled target signals** — **without access to source data or extra annotation**.

## Method

The proposed framework consists of four core components:

1. **Few-Shot Class-Center Anchor Initialization.** Device prototypes (class-center anchors) are constructed from the scarce labeled target samples, establishing reliable class-level anchors in the feature space of the source-pretrained extractor.

2. **Contrastive Candidate Sampling & Prototype-Consistency Pseudo Labeling.** Informative unlabeled candidates are selected by a contrastive scoring function that combines the cross-round prediction residual and a class-level transferability term. Selected candidates are *not* directly labeled: a pseudo-label is confirmed only when the model prediction agrees with the nearest class-center prototype prediction (`confirm_by_prototype`), suppressing noisy pseudo labels under large cross-channel shifts.

3. **Persistent Class-Center RF Adaptation.** Class-center anchors are iteratively updated with an EMA scheme (momentum `gamma = 0.9`) and supervise remaining unlabeled samples through a persistence-guided alignment loss `L_pgra` (center-affinity, temperature `tau`), encouraging intra-class compactness and inter-class dispersion.

4. **Random RF Signal Masking.** To counteract overfitting from few-shot labels, continuous time-domain RF segments are randomly masked for online augmentation while device labels are preserved, enriching target-domain supervision diversity.

The overall objective is `L_total = L_ce + L_pgra`, trained iteratively over adaptation rounds (see Algorithm 1 in the paper). The active-adaptation loop is: contrastive candidate sampling → query labels → prototype-consistency pseudo labeling → train with persistent class-center guidance + Random RF Signal Masking.

## Results

Results on the **ORACLE** benchmark, **S1 → S2 (62ft)**, 16 shared devices, averaged over 5 seeds (mean ± std, %).

### Main results

Reported numbers use **validation-set model selection** (see *Model-selection protocol* below). Results on ORACLE **S1 → S2 (62ft)**, 16 shared devices, averaged over 5 seeds (mean ± std, %).

| Method | 5-shot Acc. (%) | 10-shot Acc. (%) |
| :--- | :---: | :---: |
| Source Only | 82.63 | — |
| CORAL [8] | 87.56 ± 0.56 | — |
| MMD [9] | 87.46 ± 1.15 | — |
| DANN [10] | 86.34 ± 0.32 | — |
| MixUp [16] | 81.79 ± 3.14 | — |
| Fine-tune [17] | 85.09 ± 0.77 | 85.78 ± 0.61 |
| Linear Probe | 87.25 ± 1.22 | 88.09 ± 0.60 |
| MME [13] | 94.91 ± 0.35 | 95.51 ± 0.76 |
| **Our proposed (SF-FSDA)** | **95.06 ± 0.84** | **97.40 ± 0.58** |

`—` denotes settings not reported in Table I of the paper; reference numbers follow that table.

CORAL [8], MMD [9], and DANN [10] improve over Source Only but remain limited under cross-channel shift. MixUp [16] does not explicitly address domain discrepancy, while fine-tuning [17] and linear probing are constrained by scarce target supervision. **MME [13] uses full labeled source data during adaptation and thus benefits from both source and target information.** Despite being source-free, our method achieves 95.06 ± 0.84 and 97.40 ± 0.58 under 5-shot and 10-shot, respectively. With identical few-shot splits, seeds, and validation-based model selection, it **performs comparably to MME at 5-shot** and **improves 10-shot accuracy by 1.89 pp** (p = 0.0077, paired t-test).

### Ablation study

| Method | 5-shot Acc. (%) | 10-shot Acc. (%) |
| :--- | :---: | :---: |
| w/o prototype consistency check | 89.91 ± 2.59 | 90.85 ± 2.31 |
| w/o model consistency check | 92.11 ± 2.50 | 95.16 ± 1.54 |
| w/o random masking | 94.93 ± 1.30 | 96.88 ± 0.66 |
| **Our proposed** | **95.06 ± 0.84** | **97.40 ± 0.58** |

**Confidence-threshold ablation (δ).** An explicit pseudo-label confidence threshold `max_c p_c(x) ≥ δ` is added on top of the proposed method; values are the accuracy change relative to the threshold-free baseline (δ = 0, i.e. our proposed), in percentage points, averaged over 3 seeds.

| δ | 0.0 (proposed) | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 5-shot | 95.06 (Table I) | −0.15 | −0.15 | +0.02 | +0.06 |
| 10-shot | 97.40 (Table I) | 0.00 | +0.17 | +0.19 | +0.40 |

The change stays within **0.40 pp** for all δ, confirming that the dual-consistency gate already filters low-confidence pseudo-labels and an explicit confidence threshold is unnecessary.

### Hyperparameter sensitivity

- Joint tuning of the contrastive strength `alpha` and the class-transferability weight `lambda` (Fig. 2): optimal pairs are **(alpha, lambda) = (0.9, 0.1)** for 5-shot and **(0.7, 0.9)** for 10-shot.
- Random RF Signal Masking ratio (Fig. 3): optimal values are **0.6 (5-shot)** and **0.4 (10-shot)**; performance is stable around the optima.

Full per-seed logs (`run_summary.json`, `acc_by_round.csv`, `train_strategy.log`, `raw_result.json`) for the main results, all ablations, the joint α/λ tuning, the masking-ratio sweep, and the validation-selected re-runs are stored under [`results/`](results/):

```
results/
├── proposed/                    # main results, mc×5 (5-shot & 10-shot)
├── val_selected/                # paper FINAL numbers (validation-set model selection)
│   ├── proposed/                # our method, mc×5 (5-shot & 10-shot)
│   └── mme/                     # MME baseline, mc×5 (5-shot & 10-shot)
├── ablation/
│   ├── w_o_prototype_consistency/   # w/o prototype consistency, mc×5
│   ├── w_o_model_consistency/       # w/o model consistency, mc×5
│   ├── w_o_random_masking/          # w/o random masking, mc×5
│   └── pseudo_threshold/            # δ confidence-threshold ablation, mc×3 per δ
└── tuning/
    ├── joint_alpha_lambda/      # joint alpha (contrastive) & lambda (transferability) tuning
    └── masking_ratio/           # random RF signal masking ratio sweep
```

## Requirements

Tested with `python=3.10` and a CUDA-enabled `torch=2.7.0+cu128` build. Install the dependencies:

```bash
pip install torch==2.7.0 torchvision==0.22.0 numpy scikit-learn Pillow
```

| Package | Version | Purpose |
| :--- | :--- | :--- |
| python | 3.10.20 | runtime |
| torch | 2.7.0+cu128 | model training & adaptation |
| torchvision | 0.22.0 | data transforms (matches torch 2.7.0) |
| numpy | ≥ 1.24 | RF-signal processing, metrics |
| scikit-learn | ≥ 1.3 | data splitting, evaluation metrics |
| Pillow | ≥ 10.0 | image-style dataset loading |

Notes:
- `OpenCV` (`cv2`) is imported only for image datasets (guarded import) and is **not** required for the RF pipeline.
- The optional WiSig branch (`wisig` / `wisig-eq` datasets) uses a separate `wisig_dataloader` that is not part of this repository.

## Dataset

We evaluate on the **ORACLE** RF fingerprint dataset (code path: `ORACLE-S`), source scenario `S1` → target scenario `S2`, `62ft` features, 16 device classes. During pretraining, 4800 source samples are used for training and 1200 for validation; during adaptation, only few-shot labeled (5-/10-shot) and unlabeled target samples are available. Adapt the paths in the run scripts (`rf_root`) to your local layout.

## Experimental setup

Mirrors Section IV-A of the paper; values below are the defaults used by the scripts in `code/`.

**Data.** ORACLE `S1 → S2` (62 ft) cross-channel task with 16 shared devices. Source pretraining uses 4800 training and 1200 validation samples. The target set comprises a 4800-sample adaptation pool and a 1600-sample independent test set; 30 samples/class (480 total, seed 2025) are held out from the pool for validation, leaving 4320. Under 5-/10-shot settings, 5/10 samples per class constitute the labeled set (80/160 total), yielding 4240/4160 unlabeled samples. Five Monte-Carlo runs adopt seeds 2025–2029. Validation guides hyperparameter tuning and checkpoint selection; the test set is reserved exclusively for final evaluation.

**Source pretraining.** MSCAN [15] backbone (the repository's `--net macnn` option), optimized by SGD with momentum 0.9, Nesterov and weight decay 1e-3 for 120 epochs at base LR 1e-4 with polynomial decay `(1 + 10 · iter/max_iter)^-0.75`.

**Adaptation.** Backbone/head learning rates 1e-5 / 5e-4 with the same optimizer settings and per-round decay; batch size 64; gradient clipping 1.0; label smoothing 0.05; eight adaptation rounds of eight inner epochs, freezing the backbone for the first two rounds. We set `B = 5/10` for 5-/10-shot, with fixed `tau = 0.5`, `beta = 1`, and a class-center EMA coefficient of `gamma = 0.9`. Random RF Signal Masking occludes continuous time-domain segments with probability 0.5 and ratio `rho = 0.6/0.4` for 5-/10-shot. Features and prototypes are L2-normalized for the cosine-similarity computation, whereas the EMA-updated centers keep raw feature means.

> **Note on `gamma`.** The paper's Eq. (15) writes `gamma = 0.9` as the weight on the *previous* center. The code passes `--gamma 0.1` (`mining/strategy.py`), where the coefficient weights the *current* center, so `0.1` in code is the same setting as `gamma = 0.9` in the paper.

> **Note on early stopping.** Early stopping is **disabled** for the reported runs (`early_stop_patience=0` in `run_final_valselect.py`); all eight rounds are kept and the best validation checkpoint is selected afterwards.

## Usage

### 1. Train the source model

```
python train_source.py \
    --dataset rf --rf_root <path/to/ORACLE-S> \
    --s_folder S1 --rf_ft 62ft \
    --output ckps/source/ --max_epoch 120
```

### 2. Source-free few-shot adaptation on the target

```
python run_cas_pgra_cutout.py
```

This runs both shot budgets with MC = 5 (seeds 2025–2029):

| Setting | `alpha` | `lambda` | `query_budget` | masking ratio |
| :--- | :--- | :--- | :--- | :--- |
| 5-shot | 0.9 | 0.1 | 5 | 0.6 |
| 10-shot | 0.7 | 0.9 | 10 | 0.4 |

Results are written to `exp_cas_pgra1_cutout/{shot}shot/` (`final_summary.csv`, per-`mcXX` logs). The name `cutout` in the script corresponds to the Random RF Signal Masking strategy in the paper.

### Model-selection protocol (validation set)

The numbers reported in this README and in the paper's Table I use **validation-set model selection**: a 10% per-class holdout (480 samples, seed 2025) is carved from the target train pool (4800 → 4320), the best adaptation round/epoch is selected **by validation accuracy**, and the reported number is the test accuracy of that validation-selected checkpoint. This avoids any test-set leakage during model selection. Baselines (CORAL, MMD, DANN, MixUp) follow the same protocol.

Run the validation-selected experiments with the dedicated scripts:

```bash
# proposed method (final numbers)
python run_final_valselect.py

# MME baseline
python run_mme_valselect.py

# δ confidence-threshold ablation
python run_pseudo_threshold_ablation.py
```

Set the dataset path via `--rf_root <path/to/ORACLE-S>` if your layout differs from the default.

## Citation

```
@article{jia2026sourcefree,
    title={Source-Free Few-Shot Domain Adaptation for Channel-Robust Radio Frequency Fingerprint Identification},
    author={Jia, Mingbo and Zhang, Jie and Tang, Tiantian and Gui, Guan and Ohtsuki, Tomoaki and Sari, Hikmet},
    journal={IEEE Communications Letters},
    year={2026},
    note={under review}
}
```
