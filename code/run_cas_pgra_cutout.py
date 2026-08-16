#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAS + PGRA(beta=1.0) + Cutout 实验
快速跑 5-shot 和 10-shot，每设置 MC=5 次

用法:
  python run_cas_pgra_cutout.py
"""
import os, sys, json, csv, numpy as np, torch
from adapt_target import adapt_target, set_global_seed

MC_RUNS = 5
SEED_BASE = 2025
BASE = "D:/Pycharm/code/lftl-main/_experiments/exps_oracle/uda_rf_62ft"
# Shot 配置：(shot, cd_ratio, uct_lambda, query_budget, cutout_ratio)
# cutout_ratio 来自 results_table.xlsx Table 6 的扫参最优结果
SHOT_CFGS = [
    (5,  0.9, 0.1, 5,  0.6),   # 5-shot best: cutout=0.6 → mean=95.50%
    (10, 0.7, 0.9, 10, 0.4),   # 10-shot best: cutout=0.4 → mean=97.30%
]

OUT_ROOT = "exp_cas_pgra1_cutout"


def make_cutout_transform(ratio):
    """带 cutout 的训练变换"""
    import torch
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


def build_args(shot, cd, lam, qb, seed, mc):
    return argparse.Namespace(
        dataset="rf", rf_root="D:/pydata/Datasets/ORACLE-S",
        s_folder="S1", t_folder="S2", rf_ft="62ft",
        class_num=16, in_channels=2, channels=16, bottleneck=512,
        layer="wn", classifier="bn",
        batch_size=64, max_epoch=8, num_round=8,
        ratio_per_round=0.1, freeze_f_rounds=2,
        lr_backbone=1e-5, lr_head=5e-4, gamma=0.1, smooth=0.05, clip_grad=1.0,
        init_pool="source", source_dir=BASE,
        shots=shot, query_budget=qb, shot_schedule=None, ul_ratio=100,
        sfada_ubl=1, mem_momentum=0.1,
        beta_im=0.0, beta_vpa=0.0, beta_pgra=1.0,   # PGRA=1.0 !
        lambda_mixup=0.0, mixup_prob=0.0, mixup_alpha=0.1,
        tau=0.5, beta_im_gate=0.0,
        abl_cas="cas", cd_ratio=cd, uct_lambda=lam,
        uct_kappa=-1, warmup_rounds=0, ucm_off=0,
        use_uct_loss=False, lent_off=False,
        label_universe_frac=1.0, label_universe_idx_file=None,
        label_universe_seed=seed, use_val=False,
        make_val_from_train=False, val_ratio=0.10, val_seed=seed,
        output=f"{OUT_ROOT}/{shot}shot",
        output_dir=f"{OUT_ROOT}/{shot}shot/mc{mc:02d}",
        name=f"cas_pgra1_cutout_{shot}s_mc{mc:02d}",
        mc_runs=1, mc_seed_base=seed, gpu_id="0",
        early_stop_patience=3,
    )


def main():
    import argparse  # for build_args

    # 提前 import，为 patch 做准备
    from mining import strategy as strat

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Patch: 训练时使用 cutout 变换（cutout_fn 每轮动态传入）
    orig_train = strat.Strategy.train_one_round_with_unlabel

    def patched_train(self, rd):
        self.train_transform = getattr(self.args, '_cutout_fn', lambda t: t)
        orig_train(self, rd)

    for shot, cd, lam, qb, cut_ratio in SHOT_CFGS:
        print(f"\n{'='*60}")
        print(f"  实验: {shot}-shot | CAS + PGRA(1.0) + Cutout({cut_ratio})")
        print(f"{'='*60}")

        shot_dir = f"{OUT_ROOT}/{shot}shot"
        final_csv = os.path.join(shot_dir, "final_summary.csv")

        if os.path.exists(final_csv):
            print(f"[SKIP] {shot}-shot 已完成，跳过。")
            # 读取之前的结果打印
            with open(final_csv) as f:
                for line in f:
                    print(f"  {line.strip()}")
            continue

        os.makedirs(shot_dir, exist_ok=True)
        run_best = []
        run_histories = []

        # Random RF Signal Masking 训练变换（对应论文的 masking ratio）
        cutout_fn = make_cutout_transform(cut_ratio)

        for mc in range(MC_RUNS):
            seed = SEED_BASE + mc
            set_global_seed(seed)
            args = build_args(shot, cd, lam, qb, seed, mc)
            args._cutout_fn = cutout_fn  # 应用 Random RF Signal Masking 变换
            os.makedirs(args.output_dir, exist_ok=True)

            # 应用 patch
            strat.Strategy.train_one_round_with_unlabel = patched_train

            print(f"\n  [RUN] mc={mc}, seed={seed}")
            best_acc, acc_history = adapt_target(args)

            best_acc = float(best_acc)
            acc_history = [float(x) for x in acc_history]
            run_best.append(best_acc)
            run_histories.append(acc_history)

            # 保存单次结果
            with open(os.path.join(args.output_dir, "raw_result.json"), "w") as f:
                json.dump({
                    "shot": shot, "mc": mc, "seed": seed,
                    "beta_pgra": 1.0, "cutout_ratio": cut_ratio,
                    "best_acc": best_acc,
                    "acc_history": acc_history,
                }, f, indent=2)

            print(f"  [DONE] mc={mc}: best={best_acc:.2f}%")

        # 恢复原始方法
        strat.Strategy.train_one_round_with_unlabel = orig_train

        # 汇总 MC 结果
        mean_best = float(np.mean(run_best))
        std_best = float(np.std(run_best, ddof=1)) if len(run_best) > 1 else 0.0
        max_best = float(np.max(run_best))

        print(f"\n  === {shot}-shot 汇总 ===")
        print(f"    MC 结果: {', '.join(f'{x:.2f}' for x in run_best)}")
        print(f"    均值: {mean_best:.2f}%  ±{std_best:.2f}  最高: {max_best:.2f}%")

        # 保存汇总 CSV
        with open(final_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["shot", "beta_pgra", "cutout_ratio",
                        "mean_best", "std_best", "max_best", "all_best"])
            w.writerow([shot, 1.0, cut_ratio,
                        round(mean_best, 4), round(std_best, 4), round(max_best, 4),
                        ",".join([f"{x:.4f}" for x in run_best])])

        # 写入纯文本结果
        with open(os.path.join(shot_dir, "result.txt"), "w") as f:
            f.write(f"Method: CAS + PGRA(beta=1.0) + Cutout(ratio={cut_ratio})\n")
            f.write(f"Shot: {shot}\n")
            f.write(f"MC runs: {MC_RUNS}\n")
            f.write(f"All best: {', '.join(f'{x:.2f}%' for x in run_best)}\n")
            f.write(f"Mean ± Std: {mean_best:.2f}% ± {std_best:.2f}\n")
            f.write(f"Max: {max_best:.2f}%\n")

    # 最终总表
    print(f"\n{'='*60}")
    print("  最终结果汇总")
    print(f"{'='*60}")
    print(f"  {'Shot':>5s}  {'Mean±Std':>12s}  {'Max':>8s}  {'All':>24s}")
    print(f"  {'-'*53}")
    for shot, cd, lam, qb, cut_ratio in SHOT_CFGS:
        csv_path = f"{OUT_ROOT}/{shot}shot/final_summary.csv"
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
                r = rows[0]
                print(f"  {shot:>5d}  {r['mean_best']}±{r['std_best']}  {r['max_best']:>8s}  {r['all_best']}")
        else:
            print(f"  {shot:>5d}  {'未完成':>20s}")
    print(f"  输出目录: {OUT_ROOT}/")


if __name__ == "__main__":
    main()
