#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MME baseline re-run with validation-set model selection
=======================================================
Purpose
-------
The paper's MME comparison (④, Table 1) must be FAIR with the proposed method.
The proposed method's final numbers are being re-run with validation-set model
selection (run_final_valselect.py): the same 10% per-class validation carve
(pool 4800 -> 4320 + 480 val, val_seed=2025), the few-shot labels sampled from
the 4320 pool, and the best model chosen BY VALIDATION ACCURACY (not the test
set). This script applies the IDENTICAL protocol to the standard MME baseline
(Saito et al., ICCV 2019: labeled S1 source + few-shot S2 labels + S2
unlabeled + domain-adversarial head):

  * carve the SAME validation set out of the S2 pool before the few-shot split;
  * train 30 epochs;
  * each epoch, evaluate on the validation set (selection) and the held-out
    S2 test set (report);
  * the reported number is the TEST accuracy at the epoch with the highest
    VALIDATION accuracy;
  * also record the best TEST accuracy across epochs (test-selection), so the
    paper can report the sensitivity to the selection rule.

Recommended environment (verified on this machine):
  conda activate lftl        # torch 2.7.0+cu128, CUDA

Usage (Git Bash)
----------------
  cd /d/pycharm/code/lftl-main/supp_experiments
  /d/anaconda/envs/lftl/python.exe run_mme_valselect.py --dry_run
  /d/anaconda/envs/lftl/python.exe run_mme_valselect.py --shots 5,10 --mc_runs 5

Output (absolute, auto-created)
-------------------------------
  D:/pycharm/code/lftl-main/exp_mme_valselect/{shot}shot/mc{nn}/result.json
  D:/pycharm/code/lftl-main/exp_mme_valselect/{shot}shot/final_summary.csv
"""
import os
import sys
import json
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- repo paths (this script lives in <repo>/supp_experiments/) -----------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)  # make the repo importable regardless of cwd

# ---- repo modules (must come after sys.path setup) ----
from adapt_target import set_global_seed, stratified_split_from_train   # noqa: E402
from mining.utils import lr_scheduler, op_copy                          # noqa: E402
from dataset import iq_train_transform                                  # noqa: E402

BASE = os.path.join(_REPO, "_experiments", "exps_oracle", "uda_rf_62ft")  # source ckpts
RF_ROOT = "D:/pydata/Datasets/ORACLE-S"                                   # dataset
OUT_ROOT = os.path.join(_REPO, "exp_mme_valselect")                        # save dir

SEED_BASE = 2025

# ---- the SAME validation carve as the proposed method (run_final_valselect) --
VAL_RATIO = 0.10
VAL_SEED = 2025


class IQD(Dataset):
    def __init__(self, X, Y, t=None):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long() if Y is not None else torch.zeros(len(X)).long()
        self.t = t

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        if self.t:
            x = self.t(x)
        return x, self.Y[i], i


class GRF(torch.autograd.Function):
    """Gradient-reversal layer."""
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return -g


def load_data(rf_root, shots, seed):
    """S1 source + S2 (val carve -> few-shot labeled from the 4320 pool -> unlabeled
    + held-out test). The val carve and the few-shot split use the SAME seed
    convention as the proposed method, so the labeled sets match exactly."""
    Xs = np.load(os.path.join(rf_root, 'S1', 'x_train_62ft.npy'))
    Ys = np.load(os.path.join(rf_root, 'S1', 'y_train_62ft.npy')).astype(np.int64)
    Xt = np.load(os.path.join(rf_root, 'S2', 'x_train_62ft.npy'))
    Yt = np.load(os.path.join(rf_root, 'S2', 'y_train_62ft.npy')).astype(np.int64)
    Xte = np.load(os.path.join(rf_root, 'S2', 'x_test_62ft.npy'))
    Yte = np.load(os.path.join(rf_root, 'S2', 'y_test_62ft.npy')).astype(np.int64)

    # 1) carve the SAME 10% per-class validation set out of the pool
    (Xt, Yt), (Xv, Yv) = stratified_split_from_train(Xt, Yt, VAL_RATIO, VAL_SEED)
    print(f"      pool after val carve = {len(Yt)} | val = {len(Yv)} "
          f"({VAL_RATIO:.0%}/class, val_seed={VAL_SEED})")

    # 2) few-shot labeled from the 4320 pool (same convention as proposed)
    rng = np.random.RandomState(seed)
    ti = []
    for c in np.unique(Yt):
        ic = np.where(Yt == c)[0]
        ti.extend(rng.choice(ic, min(shots, len(ic)), False).tolist())
    ti = np.array(ti)
    m = np.ones(len(Yt), bool)
    m[ti] = False
    return Xs, Ys, Xt[ti], Yt[ti], Xt[m], Xv, Yv, Xte, Yte


def eval_acc(Xarr, Yarr, nF, nB, nC, dev):
    """Top-1 accuracy on a numpy (x, y) pair, raw (no transform)."""
    nF.eval(); nB.eval(); nC.eval()
    cor = tot = 0
    with torch.no_grad():
        for i in range(0, len(Xarr), 128):
            x = torch.from_numpy(Xarr[i:i + 128]).float().to(dev)
            p = nC(nB(nF(x))).argmax(1).cpu().numpy()
            cor += int((p == Yarr[i:i + 128]).sum()); tot += len(p)
    return cor / max(tot, 1) * 100


def run_mme_standalone(nF, nB, nC, Xs, Ys, Xl, Yl, Xu, Xv, Yv, Xt, Yt, dev, ep=30):
    """Train one MME run; select the best epoch BY VALIDATION accuracy.

    Returns (test_at_best_val, best_val, test_selected_best).
    """
    dl_s = DataLoader(IQD(Xs, Ys, iq_train_transform()), batch_size=64, shuffle=True)
    dl_l = DataLoader(IQD(Xl, Yl, iq_train_transform()), batch_size=min(64, len(Xl)), shuffle=True)
    dl_u = DataLoader(IQD(Xu, None, iq_train_transform()), batch_size=64, shuffle=True)
    dc = nn.Linear(512, 2).to(dev)
    do = torch.optim.SGD(dc.parameters(), lr=1e-3, momentum=0.9)
    opt = torch.optim.SGD([
        {'params': nF.parameters(), 'lr': 1e-6, 'lr0': 1e-6},
        {'params': nB.parameters(), 'lr': 5e-4, 'lr0': 5e-4},
        {'params': nC.parameters(), 'lr': 5e-4, 'lr0': 5e-4}], momentum=0.9)

    best_val = 0.0
    test_at_best_val = 0.0
    test_selected_best = 0.0
    mi = ep * max(len(dl_s), len(dl_l), len(dl_u))
    ti = 0
    for e in range(ep):
        nF.train(); nB.train(); nC.train()
        it_s = iter(dl_s); it_l = iter(dl_l); it_u = iter(dl_u)
        for _ in range(max(len(dl_s), len(dl_l), len(dl_u))):
            try:
                xs, ys, _ = next(it_s)
            except StopIteration:
                it_s = iter(dl_s); xs, ys, _ = next(it_s)
            try:
                xl, yl, _ = next(it_l)
            except StopIteration:
                it_l = iter(dl_l); xl, yl, _ = next(it_l)
            try:
                xu, _, _ = next(it_u)
            except StopIteration:
                it_u = iter(dl_u); xu, _, _ = next(it_u)
            if xs.size(0) <= 1 or xl.size(0) <= 1 or xu.size(0) <= 1:
                continue
            ti += 1
            lr_scheduler(opt, ti, mi)
            xs, ys = xs.to(dev), ys.to(dev)
            xl, yl = xl.to(dev), yl.to(dev)
            xu = xu.to(dev)
            # CE on S1 (source)
            fs = nB(nF(xs)); ls = nC(fs)
            loss = F.cross_entropy(ls, ys)
            # CE on S2 labeled
            fl = nB(nF(xl)); ll = nC(fl)
            loss += F.cross_entropy(ll, yl)
            # Entropy min on S2 unlabeled
            fu = nB(nF(xu)); lu = nC(fu); pu = F.softmax(lu, 1)
            loss += 0.1 * -(pu * pu.clamp(1e-12).log()).sum(1).mean()
            # Domain adversarial: S1(source) vs S2(target, all)
            fa = torch.cat([fs, torch.cat([fl, fu])], 0)
            fr = GRF.apply(fa); dd = dc(fr)
            dlb = torch.cat([torch.zeros(fs.size(0)),
                             torch.ones(fl.size(0) + fu.size(0))]).long().to(dev)
            loss += 0.01 * F.cross_entropy(dd, dlb)
            opt.zero_grad(); do.zero_grad(); loss.backward(); opt.step(); do.step()

        # Eval: validation set for SELECTION, test set for REPORT.
        acc_val = eval_acc(Xv, Yv, nF, nB, nC, dev)
        acc_test = eval_acc(Xt, Yt, nF, nB, nC, dev)
        test_selected_best = max(test_selected_best, acc_test)
        if acc_val > best_val:
            best_val = acc_val
            test_at_best_val = acc_test
        if (e + 1) % 5 == 0 or e == ep - 1:
            print(f"    [val-select] Ep{e+1}/{ep}: val={acc_val:.2f}% "
                  f"test={acc_test:.2f}% | best_val={best_val:.2f}% "
                  f"test@best_val={test_at_best_val:.2f}%")
    return test_at_best_val, best_val, test_selected_best


def check_env(dev):
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
    print(f"  [env] device   : {dev}")
    print(f"  [env] save dir : {OUT_ROOT}")
    if not ok:
        raise SystemExit("[env] 路径检查失败，请修正 RF_ROOT / BASE 后再跑。")
    print("  [env] all paths OK\n")


def run_shot(shot, mc_runs, seeds, force, out_root, rf_root, base, dev):
    shot_dir = f"{out_root}/{shot}shot"
    os.makedirs(shot_dir, exist_ok=True)
    rows = []
    for mc in range(mc_runs):
        seed = seeds[mc]
        out_dir = f"{shot_dir}/mc{mc:02d}"
        os.makedirs(out_dir, exist_ok=True)
        res_file = os.path.join(out_dir, "result.json")
        if os.path.exists(res_file) and not force:
            with open(res_file) as f:
                d = json.load(f)
            print(f"  [SKIP] {shot}-shot mc={mc} (seed={seed}) "
                  f"test@best_val={d['test_best_acc']:.2f}%")
            rows.append((d["test_best_acc"], d["val_best_acc"],
                         d["test_selected_best"]))
            continue
        set_global_seed(seed)
        print(f"  [RUN] standalone(val-select) {shot}-shot mc={mc} seed={seed}")
        Xs, Ys, Xl, Yl, Xu, Xv, Yv, Xt, Yt = load_data(rf_root, shot, seed)
        print(f"      S1={len(Xs)} S2_labeled={len(Xl)} S2_unlabeled={len(Xu)} "
              f"S2_test={len(Yt)}")
        from adapt_target import MacnnBackbone
        from network import FeatureNeck, ClassifierHead
        nF = MacnnBackbone(2, 16, 16).to(dev)
        nB = FeatureNeck(nF.in_features, 512, 'bn').to(dev)
        nC = ClassifierHead(16, 512, 'wn').to(dev)
        for mm, n in [(nF, 'F'), (nB, 'B'), (nC, 'C')]:
            mm.load_state_dict(torch.load(os.path.join(base, f'source_{n}.pt'),
                                          map_location=dev))
        test_at_best_val, best_val, test_sel = run_mme_standalone(
            nF, nB, nC, Xs, Ys, Xl, Yl, Xu, Xv, Yv, Xt, Yt, dev)
        with open(res_file, "w") as f:
            json.dump({"variant": "standalone_valselect", "shot": shot, "mc": mc,
                       "seed": seed, "val_seed": VAL_SEED, "val_ratio": VAL_RATIO,
                       "val_best_acc": float(best_val),
                       "test_best_acc": float(test_at_best_val),
                       "test_selected_best": float(test_sel)}, f, indent=2)
        rows.append((float(test_at_best_val), float(best_val), float(test_sel)))
        print(f"  [DONE] mc={mc}: test@best_val={test_at_best_val:.2f}% "
              f"(best_val={best_val:.2f}%, test-selected={test_sel:.2f}%)")
    return rows


def main():
    p = argparse.ArgumentParser(description="MME re-run with validation-set model selection")
    p.add_argument("--shots", type=str, default="5,10")
    p.add_argument("--mc_runs", type=int, default=5)
    p.add_argument("--seeds_base", type=int, default=SEED_BASE)
    p.add_argument("--rf_root", type=str, default=RF_ROOT)
    p.add_argument("--base", type=str, default=BASE)
    p.add_argument("--out_root", type=str, default=OUT_ROOT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    globals()["RF_ROOT"] = a.rf_root
    globals()["BASE"] = a.base
    globals()["OUT_ROOT"] = a.out_root

    shots = [int(s) for s in a.shots.split(",")]
    seeds = [a.seeds_base + i for i in range(a.mc_runs)]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    check_env(dev)
    if a.dry_run:
        for shot in shots:
            print(f"  [dry] would write {OUT_ROOT}/{shot}shot/mc{{00..{a.mc_runs-1:02d}}}/result.json")
        print("  [dry] nothing was run. Remove --dry_run to start.")
        return

    print(f"===== MME standalone re-run (validation-set selection) =====")
    print(f"  val split = {VAL_RATIO:.0%} per-class holdout, val_seed={VAL_SEED}")
    print(f"  shots = {shots}, mc = {a.mc_runs}, seeds = {seeds[0]}..{seeds[-1]}")

    for shot in shots:
        print(f"\n{'='*64}\n  {shot}-shot\n{'='*64}")
        rows = run_shot(shot, a.mc_runs, seeds, a.force, OUT_ROOT, RF_ROOT, BASE, dev)
        t = np.array([r[0] for r in rows]) if rows else np.array([])
        v = np.array([r[1] for r in rows]) if rows else np.array([])
        ts = np.array([r[2] for r in rows]) if rows else np.array([])
        with open(os.path.join(f"{OUT_ROOT}/{shot}shot", "final_summary.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "shot", "test_mean", "test_std", "all_test",
                        "all_val", "all_test_selected"])
            w.writerow(["standalone_valselect", shot,
                        round(float(t.mean()), 4) if len(t) else "",
                        round(float(t.std(ddof=1)), 4) if len(t) > 1 else "",
                        ",".join(f"{x:.4f}" for x in t),
                        ",".join(f"{x:.4f}" for x in v),
                        ",".join(f"{x:.4f}" for x in ts)])
        print(f"\n  === MME(val-select) {shot}-shot ===")
        if len(t):
            print(f"    TEST @ best-validation epoch: "
                  f"{t.mean():.2f} ± {t.std(ddof=1):.2f}%")
            print(f"    (validation best: {v.mean():.2f} ± {v.std(ddof=1):.2f}% | "
                  f"test-selected best: {ts.mean():.2f} ± {ts.std(ddof=1):.2f}%)")
    print(f"\nDone. Results saved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
