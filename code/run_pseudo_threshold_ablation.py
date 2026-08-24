#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Confidence-threshold (delta) ablation for pseudo-labeling
==========================================================
Purpose
-------
The academician's comment asks: "If no confidence threshold is used, please
provide an ablation study demonstrating why it is unnecessary."

Our current pseudo-label rule (strategy.py `confirm_by_prototype`) accepts a
candidate only when model prediction and prototype prediction AGREE:
    y_tilde = y_hat_m  iff  y_hat_m == y_hat_p
There is no explicit confidence threshold (the code's `pseudo_conf_threshold`
defaults to 0.0, i.e. only `max p_c > 0`, which always holds).

This script runs the FULL proposed method (CAS + PGRA(beta=1.0) + Random RF
Signal Masking) while additionally requiring `max_c p_c(r)(x) > delta`, for
delta in {0.3, 0.5, 0.7, 0.9} (5-shot and 10-shot, seeds 2025-2029).  If the
accuracy stays essentially flat as delta grows, it demonstrates that the
dual-consistency gate already subsumes confidence screening and an explicit
threshold is unnecessary.

delta = 0.0 (no threshold) is the proposed method itself, whose results are
already stored under the proposed run (e.g. exp_cas_pgra1_cutout/ or the repo
results/).  Pass --proposed_dir to merge that baseline into the summary table.

Recommended environment (verified on this machine):
  conda activate lftl        # torch 2.7.0+cu128, CUDA

Usage (Git Bash)
----------------
  cd /d/pycharm/code/lftl-main/supp_experiments
  /d/anaconda/envs/lftl/python.exe run_pseudo_threshold_ablation.py --dry_run
  /d/anaconda/envs/lftl/python.exe run_pseudo_threshold_ablation.py --shots 5,10
  /d/anaconda/envs/lftl/python.exe run_pseudo_threshold_ablation.py \
      --deltas 0.3 0.7 --mc_runs 3                       # quick look

Output (absolute, auto-created)
-------------------------------
  D:/pycharm/code/lftl-main/exp_ablation_pseudo_threshold/{shot}shot/delta{δ}/mc{nn}/raw_result.json
  D:/pycharm/code/lftl-main/exp_ablation_pseudo_threshold/{shot}shot/final_summary.csv
"""
import os
import sys
import json
import csv
import argparse

import numpy as np
import torch

# ---- repo paths (this script lives in <repo>/supp_experiments/) -----------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(_REPO, "_experiments", "exps_oracle", "uda_rf_62ft")  # source ckpts
RF_ROOT = "D:/pydata/Datasets/ORACLE-S"                                   # dataset
OUT_ROOT = os.path.join(_REPO, "exp_ablation_pseudo_threshold")            # save dir

SEED_BASE = 2025

# ---- make repo modules importable (this script lives in <repo>/supp_experiments/) --
sys.path.insert(0, _REPO)
from adapt_target import adapt_target, set_global_seed
from mining import strategy as strat

# Shot configs: (shot, cd_ratio, uct_lambda, query_budget, cutout_ratio)
# cutout_ratio = Random RF Signal Masking ratio (paper Fig. 3 optima).
SHOT_CFGS = [
    (5,  0.9, 0.1, 5,  0.6),
    (10, 0.7, 0.9, 10, 0.4),
]
DEFAULT_DELTAS = [0.3, 0.5, 0.7, 0.9]


def make_cutout_transform(ratio):
    """Training transform with Random RF Signal Masking (mirrors run_cas_pgra_cutout)."""
    if ratio <= 0:
        from dataset import iq_train_transform
        return iq_train_transform()

    def transform(t):
        if torch.rand(1) < 0.5:
            cut_len = int(t.shape[-1] * ratio)
            if cut_len > 0 and t.shape[-1] > cut_len:
                start = torch.randint(0, t.shape[-1] - cut_len, (1,))
                t = t.clone()
                t[:, start:start + cut_len] = 0
        return t
    return transform


def build_args(shot, cd, lam, qb, cut_ratio, seed, mc, delta, out_root,
               rf_root, base):
    return argparse.Namespace(
        dataset="rf", rf_root=rf_root,
        s_folder="S1", t_folder="S2", rf_ft="62ft",
        class_num=16, in_channels=2, channels=16, bottleneck=512,
        layer="wn", classifier="bn",
        batch_size=64, max_epoch=8, num_round=8,
        ratio_per_round=0.1, freeze_f_rounds=2,
        lr_backbone=1e-5, lr_head=5e-4, gamma=0.1, smooth=0.05, clip_grad=1.0,
        init_pool="source", source_dir=base,
        shots=shot, query_budget=qb, shot_schedule=None, ul_ratio=100,
        sfada_ubl=1, mem_momentum=0.1,
        beta_im=0.0, beta_vpa=0.0, beta_pgra=1.0,
        lambda_mixup=0.0, mixup_prob=0.0, mixup_alpha=0.1,
        tau=0.5, beta_im_gate=0.0,
        abl_cas="cas", cd_ratio=cd, uct_lambda=lam,
        uct_kappa=-1, warmup_rounds=0, ucm_off=0,
        use_uct_loss=False, lent_off=False,
        label_universe_frac=1.0, label_universe_idx_file=None,
        label_universe_seed=seed, use_val=False,
        make_val_from_train=False, val_ratio=0.10, val_seed=seed,
        # ★ the only difference from the proposed run: explicit confidence gate
        pseudo_conf_threshold=float(delta),
        output=f"{out_root}/{shot}shot/delta{delta}",
        output_dir=f"{out_root}/{shot}shot/delta{delta}/mc{mc:02d}",
        name=f"pseudo_delta{delta}_{shot}s_mc{mc:02d}",
        mc_runs=1, mc_seed_base=seed, gpu_id="0",
        early_stop_patience=3,
    )


def check_env():
    print("  [env] repo     :", _REPO)
    print("  [env] dataset  :", RF_ROOT)
    print("  [env] ckpts    :", BASE)
    ok = True
    for sub in ["S1", "S2"]:
        good = os.path.isdir(os.path.join(RF_ROOT, sub))
        print(f"  [env] {'OK ' if good else 'X  '} ORACLE-S/{sub} exists")
        ok = ok and good
    for n in "FBC":
        fp = os.path.join(BASE, f"source_{n}.pt")
        good = os.path.isfile(fp)
        print(f"  [env] {'OK ' if good else 'X  '} source_{n}.pt")
        ok = ok and good
    dev = torch.cuda.is_available()
    print(f"  [env] CUDA     : {'AVAILABLE' if dev else 'NOT available (will be very slow / may fail)'}")
    print(f"  [env] save dir : {OUT_ROOT}")
    if not ok:
        raise SystemExit("[env] 路径检查失败，请修正 RF_ROOT / BASE 后再跑。")
    print("  [env] all paths OK\n")


def run_one_shot(shot, cd, lam, qb, cut_ratio, deltas, mc_runs, seeds,
                 force, out_root, rf_root, base):
    shot_dir = f"{out_root}/{shot}shot"
    os.makedirs(shot_dir, exist_ok=True)
    rows = []  # (delta, mean, std, all_best)

    orig_train = strat.Strategy.train_one_round_with_unlabel

    def patched_train(self, rd):
        self.train_transform = getattr(self.args, '_cutout_fn', lambda t: t)
        orig_train(self, rd)

    try:
        for delta in deltas:
            delta_str = f"{float(delta):.1f}"
            bests = []
            for mc in range(mc_runs):
                seed = seeds[mc]
                out_dir = f"{shot_dir}/delta{delta_str}/mc{mc:02d}"
                os.makedirs(out_dir, exist_ok=True)
                res_file = os.path.join(out_dir, "raw_result.json")
                if os.path.exists(res_file) and not force:
                    with open(res_file) as f:
                        best_acc = float(json.load(f)["best_acc"])
                    print(f"  [SKIP] {shot}-shot delta={delta_str} mc={mc} "
                          f"best={best_acc:.2f}%")
                    bests.append(best_acc)
                    continue

                set_global_seed(seed)
                args = build_args(shot, cd, lam, qb, cut_ratio, seed, mc,
                                  delta, out_root, rf_root, base)
                strat.Strategy.train_one_round_with_unlabel = patched_train
                # Wire the masking transform into this round's training.
                args._cutout_fn = make_cutout_transform(cut_ratio)

                print(f"  [RUN] {shot}-shot delta={delta_str} mc={mc} seed={seed}")
                best_acc, acc_history = adapt_target(args)
                best_acc = float(best_acc)
                acc_history = [float(x) for x in acc_history]
                with open(res_file, "w") as f:
                    json.dump({
                        "shot": shot, "delta": float(delta), "mc": mc, "seed": seed,
                        "cutout_ratio": cut_ratio, "beta_pgra": 1.0,
                        "best_acc": best_acc, "acc_history": acc_history,
                    }, f, indent=2)
                bests.append(best_acc)
                print(f"  [DONE] mc={mc}: best={best_acc:.2f}%")

            mean_b = float(np.mean(bests))
            std_b = float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0
            rows.append((float(delta), mean_b, std_b, bests))
            print(f"\n  --- {shot}-shot delta={delta_str} ---")
            print(f"    per-seed: {', '.join(f'{x:.2f}' for x in bests)}")
            print(f"    mean ± std: {mean_b:.2f} ± {std_b:.2f}")
    finally:
        strat.Strategy.train_one_round_with_unlabel = orig_train

    return rows


def write_summary(shot, rows, proposed_dir, out_root):
    shot_dir = f"{out_root}/{shot}shot"

    # Baseline delta=0 (the proposed method itself).
    baseline = None
    if proposed_dir:
        p_accs = []
        for mc in range(5):
            rj = os.path.join(proposed_dir, f"{shot}shot", f"mc{mc:02d}",
                              "raw_result.json")
            if os.path.exists(rj):
                with open(rj) as f:
                    p_accs.append(float(json.load(f)["best_acc"]))
        if len(p_accs) == 5:
            baseline = (0.0, float(np.mean(p_accs)),
                        float(np.std(p_accs, ddof=1)), p_accs)

    with open(os.path.join(shot_dir, "final_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "mean_best", "std_best", "all_best"])
        if baseline is not None:
            d, m, s, b = baseline
            w.writerow([d, round(m, 4), round(s, 4),
                        ",".join(f"{x:.4f}" for x in b)])
        for d, m, s, b in rows:
            w.writerow([d, round(m, 4), round(s, 4),
                        ",".join(f"{x:.4f}" for x in b)])

    print(f"\n  === {shot}-shot 汇总 (delta sweep) ===")
    print(f"    {'delta':>5s}  {'mean±std':>14s}  {'all best':>30s}")
    if baseline is not None:
        print(f"    {0.0:>5.1f}  {baseline[1]:.2f}±{baseline[2]:.2f}  "
              f"{[round(x,2) for x in baseline[3]]}")
    for d, m, s, b in rows:
        print(f"    {d:>5.1f}  {m:.2f}±{s:.2f}  {[round(x,2) for x in b]}")


def main():
    p = argparse.ArgumentParser(description="Confidence-threshold (delta) ablation")
    p.add_argument("--shots", type=str, default="5,10")
    p.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS,
                   help="confidence thresholds to sweep (default: 0.3 0.5 0.7 0.9)")
    p.add_argument("--mc_runs", type=int, default=5)
    p.add_argument("--seeds_base", type=int, default=SEED_BASE)
    p.add_argument("--rf_root", type=str, default=RF_ROOT,
                   help="ORACLE dataset root (contains S1/ and S2/)")
    p.add_argument("--base", type=str, default=BASE,
                   help="dir holding source_F/B/C.pt")
    p.add_argument("--out_root", type=str, default=OUT_ROOT,
                   help="absolute save dir (default: <repo>/exp_ablation_pseudo_threshold)")
    p.add_argument("--force", action="store_true",
                   help="re-run even if raw_result.json already exists")
    p.add_argument("--proposed_dir", type=str, default=None,
                   help="path to the proposed results dir (to merge the delta=0 "
                        "baseline row into the summary table)")
    p.add_argument("--dry_run", action="store_true",
                   help="only check env paths + print where results will be saved")
    a = p.parse_args()

    globals()["RF_ROOT"] = a.rf_root
    globals()["BASE"] = a.base
    globals()["OUT_ROOT"] = a.out_root

    shots = [int(s) for s in a.shots.split(",")]
    seeds = [a.seeds_base + i for i in range(a.mc_runs)]
    deltas = sorted(set(a.deltas))
    if 0.0 in deltas:
        print("[warn] delta=0.0 IS the proposed method; it will be run as-is "
              "and serves as the no-threshold baseline.")

    check_env()
    if a.dry_run:
        for shot, cd, lam, qb, cut_ratio in SHOT_CFGS:
            if shot not in shots:
                continue
            for delta in deltas:
                print(f"  [dry] {shot}-shot delta={float(delta):.1f} -> "
                      f"{OUT_ROOT}/{shot}shot/delta{float(delta):.1f}/mc{{00..{a.mc_runs-1:02d}}}/raw_result.json")
        print("  [dry] nothing was run. Remove --dry_run to start.")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"===== Pseudo-label confidence-threshold ablation =====")
    print(f"  deltas = {deltas}, mc = {a.mc_runs}, seeds = {seeds[0]}..{seeds[-1]}")

    for shot, cd, lam, qb, cut_ratio in SHOT_CFGS:
        if shot not in shots:
            continue
        print(f"\n{'='*60}\n  {shot}-shot\n{'='*60}")
        rows = run_one_shot(shot, cd, lam, qb, cut_ratio, deltas, a.mc_runs,
                            seeds, a.force, OUT_ROOT, RF_ROOT, BASE)
        write_summary(shot, rows, a.proposed_dir, OUT_ROOT)
    print(f"\nDone. Results saved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
