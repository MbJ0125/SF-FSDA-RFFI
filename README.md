# Source-Free Few-Shot Domain Adaptation for Channel-Robust RF Fingerprint Identification

**Source-Free Few-Shot Domain Adaptation for Channel-Robust Radio Frequency Fingerprint Identification**
Mingbo Jia, Jie Zhang, Tiantian Tang, Guan Gui, Tomoaki Ohtsuki, and Hikmet Sari. *IEEE Communications Letters* (under review).

This repository provides the implementation of a **class-center-guided source-free few-shot domain adaptation (SF-FSDA)** framework for **radio frequency fingerprint identification (RFFI)**. The framework adapts a source-pretrained RFFI model to an unseen target channel using only **a few labeled target samples and unlabeled target signals** — **without access to source data or extra annotation**. It is built on top of [LFTL (ECCV 2024)](https://arxiv.org/abs/2407.18899).

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

| Method | 5-shot Acc. (%) | 10-shot Acc. (%) |
| :--- | :---: | :---: |
| **Our proposed (SF-FSDA)** | **95.50 ± 1.50** | **97.30 ± 0.77** |

The framework outperforms representative baselines (Source Only, CORAL, MMD, DANN, MixUp, Fine-tune, Linear Probe, MME) by **1.18% (5-shot)** and **0.48% (10-shot)**.

### Ablation study

| Method | 5-shot Acc. (%) | 10-shot Acc. (%) |
| :--- | :---: | :---: |
| w/o prototype consistency check | 89.91 ± 2.59 | 90.85 ± 2.31 |
| w/o model consistency check | 92.11 ± 2.50 | 95.16 ± 1.54 |
| w/o random masking | 94.93 ± 1.30 | 96.88 ± 0.66 |
| **Our proposed** | **95.50 ± 1.50** | **97.30 ± 0.77** |

### Hyperparameter sensitivity

- Joint tuning of the contrastive strength `alpha` and the class-transferability weight `lambda` (Fig. 2): optimal pairs are **(alpha, lambda) = (0.9, 0.1)** for 5-shot and **(0.7, 0.9)** for 10-shot.
- Random RF Signal Masking ratio (Fig. 3): optimal values are **0.6 (5-shot)** and **0.4 (10-shot)**; performance is stable around the optima.

Full per-seed logs (`run_summary.json`, `acc_by_round.csv`, `train_strategy.log`, `raw_result.json`) for the main results, all ablations, the joint α/λ tuning, and the masking-ratio sweep are stored under [`results/`](results/):

```
results/
├── proposed/                    # main results, mc×5 (5-shot & 10-shot)
├── ablation/                    # w/o prototype / w/o model consistency / w/o random masking, mc×5
└── tuning/
    ├── joint_alpha_lambda/      # joint alpha (contrastive) & lambda (transferability) tuning
    └── masking_ratio/           # random RF signal masking ratio sweep
```

## Requirements

- `python=3.10`
- `torch=2.3.0`
- `torchvision`
- `numpy`

## Dataset

We evaluate on the **ORACLE** RF fingerprint dataset (code path: `ORACLE-S`), source scenario `S1` → target scenario `S2`, `62ft` features, 16 device classes. During pretraining, 4800 source samples are used for training and 1200 for validation; during adaptation, only few-shot labeled (5-/10-shot) and unlabeled target samples are available. Adapt the paths in the run scripts (`rf_root`) to your local layout.

## Usage

### 1. Train the source model

```
python train_source.py \
    --dataset rf --rf_root <path/to/ORACLE-S> \
    --s_folder S1 --rf_ft 62ft \
    --output ckps/source/ --max_epoch 10
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

## Acknowledgment

The codebase is based on the SF-DA work [tim-learn/SHOT](https://github.com/tim-learn/SHOT) and [LFTL (ECCV 2024)](https://arxiv.org/abs/2407.18899). We thank the authors for releasing their code.

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
