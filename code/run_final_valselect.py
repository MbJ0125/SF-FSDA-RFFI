#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final re-run with validation-set model selection
================================================
Purpose
-------
The paper's final numbers (95.50 / 97.30) were produced by `adapt_target`
selecting the best round ON THE S2 TEST SET (adapt_target.py:818-831, 984).
That is a second test-set-leakage point (the first is hyperparameter selection,
handled by run_hp_validation.py). To make the revision honest and consistent,
this script re-runs the FINAL experiments with a single, clean protocol:

  * a per-class 10% validation set is carved out of the S2 adaptation pool
    (480 samples, val_seed=2025), disjoint from the few-shot labeled sets and
    from the test set;
  * hyperparameters are FIXED at the paper's operating points
    (alpha, lambda, rho) = (0.9, 0.1, 0.6) for 5-shot and (0.7, 0.9, 0.4) for
    10-shot (already shown to be on the validation plateau in
    exp_hp_validation);
  * the best round is selected BY VALIDATION ACCURACY (Strategy.predict is
    patched to evaluate on the validation set);
  * the selected checkpoint is then evaluated ONCE on the held-out S2 test
    set; that test accuracy is the reported number.

This mirrors run_hp_validation.py but runs ONLY the chosen operating points,
with mc_runs=5 (seeds 2025-2029), and additionally records the TEST accuracy
of the validation-selected checkpoint (and of every saved round, so the
sensitivity of the result to the selection rule can be reported).

Recommended environment (verified on this machine):
  conda activate lftl        # torch 2.7.0+cu128, CUDA

Usage (Git Bash)
----------------
  cd /d/pycharm/code/lftl-main/supp_experiments
  /d/anaconda/envs/lftl/python.exe run_final_valselect.py --dry_run
  /d/anaconda/envs/lftl/python.exe run_final_valselect.py --shots 5,10 --mc_runs 5
  /d/anaconda/envs/lftl/python.exe run_final_valselect.py --smoke       # 1 round x 1 epoch

Output (absolute, auto-created)
-------------------------------
  D:/pycharm/code/lftl-main/exp_final_valselect/{shot}shot/mc{nn}/raw_result.json
  D:/pycharm/code/lftl-main/exp_final_valselect/{shot}shot/final_summary.csv
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
sys.path.insert(0, _REPO)
BASE = os.path.join(_REPO, "_experiments", "exps_oracle", "uda_rf_62ft")  # source ckpts
RF_ROOT = "D:/pydata/Datasets/ORACLE-S"                                   # dataset
OUT_ROOT = os.path.join(_REPO, "exp_final_valselect")                     # save dir

SEED_BASE = 2025

# ---- repo modules (never modified; only monkey-patched at runtime) --------
from adapt_target import (adapt_target, set_global_seed, load_target_splits,
                          build_model, cal_acc)
from mining import strategy as strat
from dataset import IQNPYDataset, iq_test_transform
from torch.utils.data import DataLoader

# ---- the fixed validation split (same as run_hp_validation.py) ------------
VAL_RATIO = 0.10
VAL_SEED = 2025

# ---- the paper's operating points (fixed; no tuning here) -----------------
CHOSEN = {5: (0.9, 0.1, 0.6), 10: (0.7, 0.9, 0.4)}


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


def build_args(shot, cd, lam, cut_ratio, seed, mc, out_root, rf_root, base,
               num_round, max_epoch):
    qb = 5 if shot == 5 else 10
    return argparse.Namespace(
        dataset="rf", rf_root=rf_root,
        s_folder="S1", t_folder="S2", rf_ft="62ft",
        class_num=16, in_channels=2, channels=16, bottleneck=512,
        layer="wn", classifier="bn",
        batch_size=64, max_epoch=max_epoch, num_round=num_round,
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
        make_val_from_train=True, val_ratio=VAL_RATIO, val_seed=VAL_SEED,
        pseudo_conf_threshold=0.0,
        output=f"{out_root}/{shot}shot",
        output_dir=f"{out_root}/{shot}shot/mc{mc:02d}",
        name=f"finalvs_{shot}s_mc{mc:02d}",
        mc_runs=1, mc_seed_base=seed, gpu_id="0",
        early_stop_patience=0,
    )


def build_val_list(rf_root):
    """Rebuild the EXACT same 10% validation split `adapt_target` creates
    when make_val_from_train=True, wrapped as the list Strategy.predict wants."""
    from adapt_target import stratified_split_from_train

    (train_x, train_y), _val_pairs, _test_pairs = \
        load_target_splits(rf_root, "S2", "62ft")
    _, (val_x, val_y) = stratified_split_from_train(
        train_x, train_y, VAL_RATIO, VAL_SEED)

    cache = os.path.join(OUT_ROOT, "_val_cache")
    os.makedirs(cache, exist_ok=True)
    vx = os.path.join(cache, f"val_x_{VAL_SEED}.npy")
    vy = os.path.join(cache, f"val_y_{VAL_SEED}.npy")
    np.save(vx, np.asarray(val_x))
    np.save(vy, np.asarray(val_y))

    ds_val = IQNPYDataset(vx, vy, transform=iq_test_transform())
    val_list = [(ds_val[i][0], ds_val[i][1], i) for i in range(len(ds_val))]
    print(f"  [val] validation set = {len(val_list)} samples "
          f"({VAL_RATIO:.0%} per-class of the adaptation pool, val_seed={VAL_SEED})")
    return val_list


def build_test_loader(rf_root, batch_size=128):
    """The held-out S2 test set (same construction as adapt_target)."""
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = \
        load_target_splits(rf_root, "S2", "62ft")
    ds_test = IQNPYDataset(test_x, test_y, transform=iq_test_transform())
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    print(f"  [test] held-out test set = {len(ds_test)} samples")
    return loader


def eval_checkpoint(test_loader, args, out_dir, best_round):
    """Load the model saved at `best_round` and evaluate on the S2 test set.

    `best_round` is the round whose checkpoint was saved by NetWrap.save when
    the VALIDATION accuracy improved (Strategy.predict patched to the val set);
    round 0's model is stored as best_F_adapt.pt.
    Returns (test_acc, mean_entropy).
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netF, netB, netC = build_model(args, dev)
    if best_round > 0:
        pF = os.path.join(out_dir, f"round{best_round}_F.pt")
        pB = os.path.join(out_dir, f"round{best_round}_B.pt")
        pC = os.path.join(out_dir, f"round{best_round}_C.pt")
    else:
        pF = os.path.join(out_dir, "best_F_adapt.pt")
        pB = os.path.join(out_dir, "best_B_adapt.pt")
        pC = os.path.join(out_dir, "best_C_adapt.pt")
    netF.load_state_dict(torch.load(pF, map_location=dev))
    netB.load_state_dict(torch.load(pB, map_location=dev))
    netC.load_state_dict(torch.load(pC, map_location=dev))
    netF.eval(); netB.eval(); netC.eval()
    acc, mean_ent = cal_acc(test_loader, netF, netB, netC)
    return float(acc), float(mean_ent)


def saved_rounds(out_dir, num_round):
    """Rounds (1..num_round) that have a checkpoint (i.e. every validation
    improvement). Returns sorted list; the last one is the validation-best."""
    rds = [rd for rd in range(1, num_round + 1)
           if os.path.exists(os.path.join(out_dir, f"round{rd}_F.pt"))]
    return sorted(rds)


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
    print(f"  [env] CUDA     : {'AVAILABLE' if dev else 'NOT available (very slow / may fail)'}")
    print(f"  [env] save dir : {OUT_ROOT}")
    if not ok:
        raise SystemExit("[env] 路径检查失败，请修正 RF_ROOT / BASE 后再跑。")
    print("  [env] all paths OK\n")


def run_one_mc(shot, cd, lam, cut_ratio, seed, mc, force, out_root, rf_root,
               base, num_round, max_epoch, val_list, test_loader):
    shot_dir = f"{out_root}/{shot}shot"
    out_dir = f"{shot_dir}/mc{mc:02d}"
    os.makedirs(out_dir, exist_ok=True)
    res_file = os.path.join(out_dir, "raw_result.json")
    if os.path.exists(res_file) and not force:
        with open(res_file) as f:
            d = json.load(f)
        print(f"  [SKIP] {shot}-shot mc={mc} seed={seed} "
              f"val_best={d['val_best_acc']:.2f}% test_best={d['test_best_acc']:.2f}%")
        return d["test_best_acc"]

    set_global_seed(seed)
    args = build_args(shot, cd, lam, cut_ratio, seed, mc, out_root, rf_root,
                      base, num_round, max_epoch)
    args._cutout_fn = make_cutout_transform(cut_ratio)

    # --- patch 1: round evaluation -> validation set (only call site of
    #     Strategy.predict in adapt_target is the round loop, line 984) ---
    orig_predict = strat.Strategy.predict
    def patched_predict(self, dataset, round=-1):
        return orig_predict(self, val_list, round)
    strat.Strategy.predict = patched_predict

    # --- patch 2: training transform with RF masking -----------------------
    orig_train = strat.Strategy.train_one_round_with_unlabel
    def patched_train(self, rd):
        self.train_transform = getattr(self.args, '_cutout_fn', lambda t: t)
        orig_train(self, rd)
    strat.Strategy.train_one_round_with_unlabel = patched_train

    try:
        print(f"  [RUN] {shot}-shot mc={mc} seed={seed} "
              f"(cd={cd}, lam={lam}, rho={cut_ratio}, val_selected)")
        val_best_acc, acc_history = adapt_target(args)
        val_best_acc = float(val_best_acc)
        acc_history = [float(x) for x in acc_history]

        # --- evaluate the saved checkpoints on the held-out TEST set -------
        rds = saved_rounds(out_dir, num_round)
        best_round = rds[-1] if rds else 0          # validation-best round
        test_best_acc, test_ent = eval_checkpoint(test_loader, args, out_dir,
                                                  best_round)
        # test accuracy of every saved round (for the selection-rule note)
        test_per_round = {}
        if best_round > 0:
            t0, _ = eval_checkpoint(test_loader, args, out_dir, 0)
            test_per_round[0] = t0
            for rd in rds:
                ta, _ = eval_checkpoint(test_loader, args, out_dir, rd)
                test_per_round[rd] = ta
        test_selected_best = max(test_per_round.values()) if test_per_round \
            else test_best_acc

        with open(res_file, "w") as f:
            json.dump({
                "shot": shot, "mc": mc, "seed": seed,
                "cd_ratio": float(cd), "uct_lambda": float(lam),
                "cutout_ratio": float(cut_ratio),
                "val_seed": VAL_SEED, "val_ratio": VAL_RATIO,
                "beta_pgra": 1.0,
                "val_best_acc": val_best_acc,
                "acc_history": acc_history,
                "best_round": best_round,
                "saved_rounds": rds,
                "test_best_acc": test_best_acc,      # final reported number
                "test_entropy": test_ent,
                "test_per_round": test_per_round,
                "test_selected_best": test_selected_best,
            }, f, indent=2)
        print(f"  [DONE] mc={mc}: val_best={val_best_acc:.2f}% | "
              f"TEST(validation-selected, round {best_round}) = "
              f"{test_best_acc:.2f}% | test-selected best = "
              f"{test_selected_best:.2f}%")
        return test_best_acc
    finally:
        strat.Strategy.predict = orig_predict
        strat.Strategy.train_one_round_with_unlabel = orig_train


def main():
    p = argparse.ArgumentParser(description="Final run with validation-set model selection")
    p.add_argument("--shots", type=str, default="5,10")
    p.add_argument("--mc_runs", type=int, default=5)
    p.add_argument("--seeds_base", type=int, default=SEED_BASE)
    p.add_argument("--num_round", type=int, default=8)
    p.add_argument("--max_epoch", type=int, default=8)
    p.add_argument("--smoke", action="store_true",
                   help="fast 1-round x 1-epoch check")
    p.add_argument("--rf_root", type=str, default=RF_ROOT)
    p.add_argument("--base", type=str, default=BASE)
    p.add_argument("--out_root", type=str, default=OUT_ROOT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    if a.smoke:
        a.num_round = 1
        a.max_epoch = 1
        a.mc_runs = min(a.mc_runs, 1)

    globals()["RF_ROOT"] = a.rf_root
    globals()["BASE"] = a.base
    globals()["OUT_ROOT"] = a.out_root

    shots = [int(s) for s in a.shots.split(",")]
    seeds = [a.seeds_base + i for i in range(a.mc_runs)]

    check_env()
    if a.dry_run:
        for shot in shots:
            cd, lam, rho = CHOSEN[shot]
            print(f"  [dry] {shot}-shot operating point (cd,lam,rho) = "
                  f"{CHOSEN[shot]} x mc 0..{a.mc_runs-1} -> "
                  f"{OUT_ROOT}/{shot}shot/mc{{00..{a.mc_runs-1:02d}}}/raw_result.json")
        print("  [dry] nothing was run. Remove --dry_run to start.")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    print("===== FINAL re-run: validation-set model selection =====")
    print(f"  val split = {VAL_RATIO:.0%} per-class holdout, val_seed={VAL_SEED}")
    print(f"  operating points fixed at {CHOSEN}")
    print(f"  shots = {shots}, mc = {a.mc_runs}, seeds = {seeds[0]}..{seeds[-1]}")
    print(f"  rounds/epochs = {a.num_round}/{a.max_epoch}, early-stop = OFF\n")

    for shot in shots:
        print(f"\n{'='*64}\n  {shot}-shot\n{'='*64}")
        val_list = build_val_list(RF_ROOT)
        test_loader = build_test_loader(RF_ROOT)
        cd, lam, rho = CHOSEN[shot]
        test_accs = []
        for mc in range(a.mc_runs):
            seed = seeds[mc]
            ta = run_one_mc(shot, cd, lam, rho, seed, mc, a.force, OUT_ROOT,
                            RF_ROOT, BASE, a.num_round, a.max_epoch,
                            val_list, test_loader)
            if ta is not None:
                test_accs.append(ta)

        # ---- summary for this shot ----
        shot_dir = f"{OUT_ROOT}/{shot}shot"
        with open(os.path.join(shot_dir, "final_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mc", "seed", "val_best_acc", "best_round",
                        "test_best_acc", "test_selected_best"])
            rows = []
            for mc in range(a.mc_runs):
                rj = os.path.join(shot_dir, f"mc{mc:02d}", "raw_result.json")
                if os.path.exists(rj):
                    with open(rj) as jf:
                        d = json.load(jf)
                    w.writerow([d["mc"], d["seed"], round(d["val_best_acc"], 4),
                                d["best_round"], round(d["test_best_acc"], 4),
                                round(d["test_selected_best"], 4)])
                    rows.append((d["val_best_acc"], d["test_best_acc"],
                                 d["test_selected_best"]))
        if rows:
            vals = np.array(rows)
            print(f"\n  === {shot}-shot summary (validation-set selection) ===")
            print(f"    val-selected TEST accuracy: "
                  f"{vals[:,1].mean():.2f} ± {vals[:,1].std(ddof=1):.2f}%")
            print(f"    val accuracy (for reference): "
                  f"{vals[:,0].mean():.2f} ± {vals[:,0].std(ddof=1):.2f}%")
            print(f"    test-selected best (for the sensitivity note): "
                  f"{vals[:,2].mean():.2f} ± {vals[:,2].std(ddof=1):.2f}%")
    print(f"\nDone. Results saved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
