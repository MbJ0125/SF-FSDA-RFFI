# -*- coding: utf-8 -*-
"""
文件：adapt_target.py（中文注释版，独立）
用途：在 ORACLE-S 的目标域（例如 S2）上进行主动学习 + 半监督的自适应（LFTL 思想）。

主要流程：
1）加载源域训练得到的最佳权重（best_* 或 source_*）
2）在目标域进行多轮（round）的主动学习：从未标注池里挑最不确定（熵最大）的样本加入标注集
3）每轮基于（标注集 + 未标注集）进行混合训练：监督 CE + 特征持续性对齐（VPA）+ 熵最小化（Ent）
4）每轮结束在目标测试集上评估，若更优则保存 best_*_adapt.pt；下一轮从 best 开始避免漂移

关键超参：
- ratio_per_round：每轮标注比例（主动学习的预算）
- freeze_f_rounds：前 K 轮冻结特征提取器 netF，避免早期未标注信号导致灾难性漂移
- ul_ratio：未标注:已标注 的 batch 倍率（半监督强度）
- beta1 / beta2：VPA 与 熵最小化 的损失权重（过大易漂移）
- lr_backbone / lr_head：骨干与头部的学习率（解冻后骨干的 lr 要小）

注释风格：
- 为函数/类提供 docstring（PyCharm 可在调用处/悬浮查看）
- 为关键逻辑提供行内中文注释（便于快速理解实现与论文的对应）
"""

import os
import os.path as osp
import argparse
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mining.contrastive_sampling import CASampling   # 你给的模块
from mining.strategy import Strategy
import math
import json, csv, time

# ----------------------------
# 项目内导入（根据你的工程结构）
# ----------------------------
from dataset import IQNPYDataset, iq_train_transform, iq_test_transform
try:
    from CNNmodel import MACNN  # 如果 CNNmodel.py 与本文件同目录
except ImportError:
    from model.CNNmodel import MACNN  # 如果放在 model/CNNmodel.py

try:
    from network import FeatureNeck, ClassifierHead
except Exception as e:
    raise RuntimeError("network.py 必须定义 FeatureNeck / ClassifierHead（分类颈部与分类头）")

# --- 损失（若你的 loss.py 命名不同，可在此适配）
try:
    from loss import lftl_total_loss, visual_persistence_loss, entropy_min_loss
except Exception:
    # 兜底实现：避免未找到 loss.py 时无法运行（可替换为你自己的实现）
    def entropy_min_loss(logits: torch.Tensor) -> torch.Tensor:
        """未标注样本的熵最小化项：降低预测分布的熵，使决策更自信。"""
        p = F.softmax(logits, dim=1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)  # H(p) >= 0
        return ent.mean()


    def visual_persistence_loss(f_u: torch.Tensor, f_l: torch.Tensor, gamma: float = 0.9) -> torch.Tensor:
        """特征持续性约束，让未标注特征与已标注特征保持结构一致。"""
        with torch.no_grad():
            f_l_ema = gamma * f_l + (1 - gamma) * f_l.detach()
        sim = F.cosine_similarity(f_u.unsqueeze(1), f_l_ema.unsqueeze(0), dim=-1)
        sim = F.softmax(sim, dim=1)
        ent_sim = -(sim * sim.clamp_min(1e-12).log()).sum(dim=1)  # 一致性熵
        return ent_sim.mean()


    def lftl_total_loss(logits_l, labels_l, logits_u, f_u, f_l, beta_vpa=0.5, beta_im=0.1):
        """beta_vpa -> 一致性模块  / beta_im -> 熵最小化模块"""
        Lce = F.cross_entropy(logits_l, labels_l)
        Lvpa = visual_persistence_loss(f_u, f_l)
        Lent = entropy_min_loss(logits_u)
        return Lce + beta_vpa * Lvpa + beta_im * Lent


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def materialize_split_to_npy(root_dir, tag, arr_x, arr_y):
    """
    将内存中的 (arr_x, arr_y) 保存为 .npy 文件，返回两个路径。
    root_dir: 保存目录
    tag: 字符串标记，比如 'train_after_split' / 'val_from_train'
    """
    import os, os.path as osp
    import numpy as np
    os.makedirs(root_dir, exist_ok=True)
    x_path = osp.join(root_dir, f"x_{tag}.npy")
    y_path = osp.join(root_dir, f"y_{tag}.npy")
    np.save(x_path, np.asarray(arr_x))
    np.save(y_path, np.asarray(arr_y).astype(np.int64))
    return x_path, y_path


def stratified_split_from_train(train_x, train_y, val_ratio=0.1, seed=42):
    """
    从 (train_x, train_y) 中按类别分层抽取 val_ratio 比例作为验证集，返回
    (new_train_x, new_train_y), (val_x, val_y)
    允许 train_x/train_y 是 np.ndarray 或 .npy 路径字符串
    """
    import numpy as np
    assert 0.0 < val_ratio < 1.0

    # ★ 若传进来是路径字符串，先加载为数组
    if isinstance(train_x, str):
        train_x = np.load(train_x, allow_pickle=False)
    if isinstance(train_y, str):
        train_y = np.load(train_y, allow_pickle=False)

    rng = np.random.default_rng(seed)
    y = np.asarray(train_y).astype(np.int64)
    x = np.asarray(train_x)

    classes = np.unique(y)
    train_idx_all, val_idx_all = [], []

    for c in classes:
        idx_c = np.where(y == c)[0]
        n_c = len(idx_c)
        if n_c == 0:
            continue
        n_val_c = int(np.round(val_ratio * n_c))
        n_val_c = max(1, n_val_c) if n_c > 1 else 0  # 单样本类不切
        idx_c_shuf = idx_c.copy()
        rng.shuffle(idx_c_shuf)
        val_idx_c = idx_c_shuf[:n_val_c]
        train_idx_c = idx_c_shuf[n_val_c:]
        val_idx_all.append(val_idx_c)
        train_idx_all.append(train_idx_c)

    if len(val_idx_all) == 0:
        return (train_x, train_y), (None, None)

    val_idx_all = np.concatenate(val_idx_all, axis=0)
    train_idx_all = np.concatenate(train_idx_all, axis=0)
    rng.shuffle(val_idx_all)
    rng.shuffle(train_idx_all)

    new_train_x = x[train_idx_all]
    new_train_y = y[train_idx_all]
    val_x = x[val_idx_all]
    val_y = y[val_idx_all]
    return (new_train_x, new_train_y), (val_x, val_y)



# === [新增函数] 统一随机种子（用于MC复现） ===
def set_global_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # 可选：更强可复现（略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === [新增函数] 单次 run 的结果写到 CSV / JSON / TXT ===
def write_run_summary(output_dir: str, meta: dict, acc_history: list, best_acc: float):
    """为单次 run 记录详细日志：每轮acc曲线CSV、run_summary.json、run_summary.txt。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1) 每轮Acc曲线：acc_by_round.csv
    csv_path = osp.join(output_dir, "acc_by_round.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "acc"])
        for rd, acc in enumerate(acc_history, start=0):  # 约定 round0 是 run0
            w.writerow([rd, f"{acc:.2f}"])

    # 2) 元信息 + 最终结果：run_summary.json
    meta_out = dict(meta)
    meta_out.update({
        "best_acc": round(float(best_acc), 2),
        "acc_history": [round(float(a), 2) for a in acc_history],
    })
    json_path = osp.join(output_dir, "run_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    # 3) 纯文本摘要：run_summary.txt
    txt_path = osp.join(output_dir, "run_summary.txt")
    lines = [
        f"name={meta_out.get('name')}",
        f"seed={meta_out.get('seed')}",
        f"shots={meta_out.get('shots')}",
        f"label_universe={meta_out.get('lu_size')} / pool={meta_out.get('n_pool')}",
        f"num_round={meta_out.get('num_round')} | B_per_round={meta_out.get('b_per_round')}",
        f"best_acc={best_acc:.2f}%",
        f"acc_history={', '.join(f'{a:.2f}' for a in acc_history)}",
        f"time_sec={meta_out.get('time_sec')}"
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class MacnnBackbone(nn.Module):
    """MACNN 的特征提取封装（作为 netF 使用）。

    Args:
        in_channels (int): 输入通道数（IQ 为 2）
        channels (int): MACNN 基础通道
        num_classes (int): 类别数（用于内部结构/兼容）

    Note:
        - 优先调用 core.forward_features(x) 直接得到特征；
        - 若 core.forward(x) 返回 (feat, logits) 也可兼容；
        - self.in_features 用于后续 FeatureNeck 构建。
    """
    def __init__(self, in_channels: int = 2, channels: int = 16, num_classes: int = 16):
        super().__init__()
        self.core = MACNN(in_channels=in_channels, channels=channels, num_classes=num_classes)
        self.in_features = getattr(self.core, "emb_dim", channels * 12)
        print(f"[LFTL] Using MACNN backbone (in_channels={in_channels}, channels={channels}, emb_dim={self.in_features})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.core, "forward_features"):
            return self.core.forward_features(x)
        out = self.core(x)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            feat, _ = out
            return feat
        raise RuntimeError("MACNN.forward 未返回 (emb, logits)，建议在 MACNN 中实现 forward_features().")


class IndexedSubset(Dataset):
    """（主动学习辅助）对基础数据集按索引进行子集包装，并返回 (x, y, idx)。

    - 兼容基础数据集 __getitem__ 返回 dict 或 (x,y,...) 的情况；
    - idx 方便在主动学习时回写“已标注/未标注”掩码。
    """
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.idxs = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self.idxs.shape[0]

    def __getitem__(self, i: int):
        j = int(self.idxs[i])
        item = self.base[j]
        # dict 情况
        if isinstance(item, dict):
            x = item.get('x') or item.get('data') or item.get('inputs') or next(iter(item.values()))
            for ky in ('y', 'label', 'labels', 'target', 'targets'):
                if ky in item:
                    y = item[ky]
                    break
            else:
                raise KeyError('字典样本未找到标签键')
            return x, y, j
        # tuple/list 情况
        if isinstance(item, (list, tuple)):
            if len(item) < 2:
                raise ValueError(f'索引 {j} 的样本长度 < 2')
            x, y = item[0], item[1]
            return x, y, j
        raise TypeError(f'不支持的数据类型: {type(item)}')

# === EMA 更新函数 ===
def update_ema_teacher(student_F, student_B, student_C, teacher_F, teacher_B, teacher_C, momentum=0.999):
    with torch.no_grad():
        for s, t in zip(student_F.parameters(), teacher_F.parameters()):
            t.data.mul_(momentum).add_(s.data * (1.0 - momentum))
        for s, t in zip(student_B.parameters(), teacher_B.parameters()):
            t.data.mul_(momentum).add_(s.data * (1.0 - momentum))
        for s, t in zip(student_C.parameters(), teacher_C.parameters()):
            t.data.mul_(momentum).add_(s.data * (1.0 - momentum))


def split_xy(batch):
    """通用 batch 解包：支持 dict 或 (x,y,...)。

    Returns:
        (x, y): 张量对
    """
    if isinstance(batch, dict):
        x = batch.get('x') or batch.get('data') or batch.get('inputs') or next(iter(batch.values()))
        for ky in ('y', 'label', 'labels', 'target', 'targets'):
            if ky in batch:
                y = batch[ky]
                break
        else:
            raise KeyError('batch 字典中未找到标签键')
        return x, y
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError('batch 长度 < 2')
        return batch[0], batch[1]
    raise TypeError(type(batch))

def load_target_splits(root_dir: str, t_folder: str, rf_ft: str):
    """
    返回目标域数据的文件路径（优先 S2C 的显式切分）。
    返回: (train_x_path, train_y_path), (val_x_path|None, val_y_path|None), (test_x_path, test_y_path)
    """
    import os.path as osp, os
    base = osp.join(root_dir, t_folder)
    xp = lambda name: osp.join(base, f"{name}_{rf_ft}.npy")

    train_x = xp("x_train"); train_y = xp("y_train")
    test_x  = xp("x_test");  test_y  = xp("y_test")

    val_x = val_y = None
    if os.path.exists(xp("x_val")) and os.path.exists(xp("y_val")):
        val_x = xp("x_val"); val_y = xp("y_val")

    # 简单存在性检查（可留可去）
    assert os.path.exists(train_x) and os.path.exists(train_y), f"Train files not found under {base}"
    assert os.path.exists(test_x)  and os.path.exists(test_y),  f"Test files not found under {base}"

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def _load_index_file(path: str, n_pool: int):
    """读取 .npy 或 .txt 索引，去重并裁剪到 [0, n_pool)。若 path 为空返回空数组。"""
    if not path:
        return np.array([], dtype=np.int64)
    if path.lower().endswith('.npy'):
        arr = np.load(path)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            tokens = f.read().replace(',', ' ').split()
            arr = np.array([int(t) for t in tokens], dtype=np.int64)
    arr = np.unique(arr[(arr >= 0) & (arr < n_pool)])
    return arr

def cal_acc(loader: DataLoader, netF: nn.Module, netB: nn.Module, netC: nn.Module,
            args=None, round_idx=None, debug: bool=False):



    """在给定 DataLoader 上计算分类准确率与平均熵。

    Returns:
        (acc, mean_entropy): (准确率%, 平均预测熵)
    """
    dev = next(netF.parameters()).device
    start_test = True
    with torch.no_grad():
        for batch in loader:
            x, y = split_xy(batch)
            x = x.to(dev); y = y.to(dev).long()
            logits = netC(netB(netF(x)))
            if start_test:
                all_out = logits.detach().cpu(); all_y = y.detach().cpu(); start_test = False
            else:
                all_out = torch.cat([all_out, logits.detach().cpu()], dim=0)
                all_y   = torch.cat([all_y,   y.detach().cpu()], dim=0)
    prob = torch.softmax(all_out.float(), dim=1)
    pred = prob.argmax(dim=1)
    acc = (pred == all_y).float().mean().item() * 100.0
    mean_ent = (-(prob.clamp_min(1e-12) * prob.clamp_min(1e-12).log()).sum(dim=1)).mean().item()

    if debug:
        import numpy as np
        K = all_out.shape[1]
        y_np = all_y.numpy()
        p_np = pred.numpy()

        cm = np.zeros((K, K), dtype=int)
        for y, p in zip(y_np, p_np):
            cm[y, p] += 1

        per_cls = (cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist()
        acc_from_cm = 100.0 * cm.diagonal().sum() / max(cm.sum(), 1)

        # === 保存到文件，不打印到终端 ===
        assert args is not None, "cal_acc() 需要传入 args"
        save_dir = os.path.join(args.output, "confusion_logs")
        os.makedirs(save_dir, exist_ok=True)

        # round_idx=round，必须是 int，否则写 unknown
        if isinstance(round_idx, int):
            filename = f"round_{round_idx}_cm.txt"
        else:
            filename = "round_unknown_cm.txt"

        log_path = os.path.join(save_dir, filename)

        with open(log_path, "w") as f:
            f.write("CONFUSION MATRIX:\n")
            f.write(str(cm) + "\n\n")

            f.write("PER-CLASS ACC:\n")
            f.write(str(["{:.2f}%".format(100 * x) for x in per_cls]) + "\n\n")

            f.write("CLASS COUNTS:\n")
            f.write(str(cm.sum(1).tolist()) + "\n\n")

            f.write(f"ACC FROM CM: {acc_from_cm:.2f}%\n\n")

            # sanity test
            perm = torch.randperm(K)
            acc_perm = (perm[pred].cpu() == all_y).float().mean().item() * 100.0
            f.write(f"SANITY permuted acc = {acc_perm:.2f}% (should be ~{100.0 / K:.2f}%)\n")

        print(f"[Round {round_idx}] Confusion matrix saved to {log_path}")

    return acc, mean_ent



def build_model(args, device: torch.device):
    """构建 netF/netB/netC，并从源域目录加载权重（优先 best_*，否则 source_*）。"""
    netF = MacnnBackbone(args.in_channels, args.channels, args.class_num).to(device)
    netB = FeatureNeck(feature_dim=netF.in_features, bottleneck_dim=args.bottleneck, type=getattr(args, 'classifier', 'bn')).to(device)
    netC = ClassifierHead(class_num=args.class_num, bottleneck_dim=args.bottleneck, type=getattr(args, 'layer', 'wn')).to(device)

    def _load_one(m: nn.Module, fname: str):
        p1 = osp.join(args.source_dir, f"best_{fname}.pt")
        p2 = osp.join(args.source_dir, f"source_{fname}.pt")
        ckpt = p1 if osp.exists(p1) else p2
        if not osp.exists(ckpt):
            raise FileNotFoundError(f"未找到权重：{p1} 或 {p2}")
        m.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[LOAD] {fname} <= {ckpt}")

    _load_one(netF, 'F'); _load_one(netB, 'B'); _load_one(netC,'C')
    return netF, netB, netC

def _try_wisig_file(root, sub, date_str):
    import os.path as osp
    p1 = osp.join(root, sub, f"rx_1-1_date{date_str}.pkl")
    p2 = osp.join(root, sub, f"rx_1_1_date{date_str}.pkl")
    return p1 if osp.exists(p1) else p2

def load_wisig_target(args):
    """
    载入 WiSig 目标域（支持多日期拼接），并按 wisig_test_ratio / wisig_val_ratio 切分为
    训练池（主动学习池）、验证集、测试集；返回 (ds_train, ds_val, ds_test)。
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    import os.path as osp
    # 延迟导入 WiSig（仅当实际使用 WiSig 数据时才需要）
    from wisig_dataloader import load_single_dataset, Dataset_

    dates = [d.strip() for d in str(args.tgt_dates).split(',') if d.strip()]
    Xs, Ys = [], []
    for d in dates:
        p = _try_wisig_file(args.dataset_path, args.wisig_sub, d)
        assert osp.exists(p), f"[WiSig] 找不到目标域文件：{p}"
        x, y = load_single_dataset(p, num_device=args.class_num)  # [N,2,L], [N]
        Xs.append(x); Ys.append(y)
    X_all = np.concatenate(Xs, axis=0)
    Y_all = np.concatenate(Ys, axis=0)

    # 先从全集切出 test，再从剩余切 val
    X_trainpool, X_test, Y_trainpool, Y_test = train_test_split(
        X_all, Y_all, test_size=float(args.wisig_test_ratio), random_state=42, stratify=Y_all)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainpool, Y_trainpool, test_size=float(args.wisig_val_ratio), random_state=42, stratify=Y_trainpool)

    ds_train = Dataset_('train', X_train, Y_train.astype(np.int64))
    ds_val   = Dataset_('val',   X_val,   Y_val.astype(np.int64))
    ds_test  = Dataset_('test',  X_test,  Y_test.astype(np.int64))
    #print(f"[DBG] split sizes: train_pool={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

    #print(f"[WiSig-Target] pool={len(ds_train)} val={len(ds_val)} test={len(ds_test)} from dates={dates}")
    return ds_train, ds_val, ds_test


def adapt_target(args):

    """目标域主动 + 半监督自适应主流程（加入 label_universe：只允许从小池子里取标注）"""
    start_time = time.time()  # [新增] 计时
    acc_history = []  # [新增] 收集各轮acc（含Round0

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.gpu_id) != "-1" else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print('[Device]:', device)
    # --- 分支哨兵 & 预定义，避免未定义局部变量 ---
    is_wisig = False
    train_x = train_y = val_x = val_y = test_x = test_y = None


    # 加载目标域数据（S2）
    #t_root = osp.join(args.rf_root, args.t_folder)
    #ft = args.rf_ft
    #xtr = osp.join(t_root, f"x_train_{ft}.npy");
    #ytr = osp.join(t_root, f"y_train_{ft}.npy")
    #xte = osp.join(t_root, f"x_test_{ft}.npy");
    #yte = osp.join(t_root, f"y_test_{ft}.npy")

    #ds_train = IQNPYDataset(xtr, ytr, transform=iq_train_transform())
    #ds_test = IQNPYDataset(xte, yte, transform=iq_test_transform())

    #n_pool = len(ds_train)
   # print(f"[RF] Target train pool ({args.t_folder} {ft}): {n_pool} samples | test: {len(ds_test)}")
    # 加载目标域数据（优先 S2C 显式切分）
    # 加载目标域数据（优先 S2C 显式切分；返回路径）
    # === 载入目标域数据（互斥分支） ===
    if getattr(args, 'dataset', 'wisig') == 'wisig':
        # 走 WiSig 原生 .pkl 分支
        ds_train, ds_val, ds_test = load_wisig_target(args)
        n_pool = len(ds_train)
        print(f"[RF] Target train pool (WiSig): {n_pool} | val: {len(ds_val)} | test: {len(ds_test)}")
        is_wisig = True

        # 构建 DataLoader
        from torch.utils.data import DataLoader
        dls = {
            "target_tr": DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False),
            "target_te": DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False),
            "test": DataLoader(ds_test, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, drop_last=False),
        }

    else:
        # 兼容旧的 ORACLE/NPY 分支
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_target_splits(args.rf_root, args.t_folder,
                                                                                  args.rf_ft)

        # 当 val 为空且用户要求从 train 切分时，做一次分层切分
        if args.make_val_from_train and ((val_x is None) or (val_y is None) or (len(val_x) == 0)):
            (train_x, train_y), (val_x, val_y) = stratified_split_from_train(
                train_x, train_y, val_ratio=float(args.val_ratio), seed=int(args.val_seed)
            )
            if (val_x is not None) and (val_y is not None):
                print(f"[VAL] created from train: ratio={args.val_ratio} | val_size={len(val_x)}")

        _cache_root = osp.join(args.output, "_cache_splits", f"{args.t_folder}-{args.rf_ft}")
        if isinstance(train_x, (list, tuple)) or hasattr(train_x, "shape"):
            train_x, train_y = materialize_split_to_npy(_cache_root, "train_after_split", train_x, train_y)
        if (val_x is not None) and (val_y is not None) and (hasattr(val_x, "shape")):
            val_x, val_y = materialize_split_to_npy(_cache_root, "val_from_train", val_x, val_y)

        ds_train = IQNPYDataset(train_x, train_y, transform=iq_train_transform())
        ds_test = IQNPYDataset(test_x, test_y, transform=iq_test_transform())
        ds_val = IQNPYDataset(val_x, val_y, transform=iq_test_transform()) if (
                    val_x is not None and val_y is not None) else None

        n_pool = len(ds_train)
        print(f"[RF] Target train pool ({args.t_folder} {args.rf_ft}): {n_pool} | "
              f"val: {0 if ds_val is None else len(ds_val)} | test: {len(ds_test)}")

        from torch.utils.data import DataLoader
        dls = {
            "target_tr": DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False),
            "target_te": DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    drop_last=False) if ds_val is not None else None,
            "test": DataLoader(ds_test, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, drop_last=False),
        }

    # === Net 包装器：把 netF/netB/netC 封装成 (logits, emb) 接口，并提供 get_embedding_dim ===
    class NetWrap(torch.nn.Module):
        def __init__(self, netF, netB, netC, device, sample_x):
            super().__init__()
            self.netF, self.netB, self.netC = netF, netB, netC
            self.device = device
            with torch.no_grad():
                xx = sample_x.unsqueeze(0).to(device)
                emb = self.netB(self.netF(xx))
            self._emb_dim = int(emb.shape[1])

        def forward(self, x):
            f = self.netB(self.netF(x))
            logits = self.netC(f)
            return logits, f

        def get_embedding_dim(self):
            return self._emb_dim

        # ★新增：Strategy 用它来创建分组优化器（分层学习率）
        def named_parameters(self, args):
            groups = []
            for _, p in self.netF.named_parameters():
                groups.append({'params': p, 'lr': args.lr_backbone})
            for _, p in self.netB.named_parameters():
                groups.append({'params': p, 'lr': args.lr_head})
            for _, p in self.netC.named_parameters():
                groups.append({'params': p, 'lr': args.lr_head})
            return groups

        # ★替换：真正保存 F/B/C，供 Strategy.save() 调用
        def save(self, round, args):
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(self.netF.state_dict(), os.path.join(args.output_dir, f'round{round}_F.pt'))
            torch.save(self.netB.state_dict(), os.path.join(args.output_dir, f'round{round}_B.pt'))
            torch.save(self.netC.state_dict(), os.path.join(args.output_dir, f'round{round}_C.pt'))

    # === 列表→Dataset 适配器：Strategy.handler 期望能把 list 转成 torch.utils.data.Dataset ===
    class ListHandler(Dataset):
        def __init__(self, data, transform=None):
            # ★ 关键：把传入的 list 存到 self.data，供 __getitem__ 使用
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            item = self.data[i]
            # 兼容 (x, y, idx) 或 (x, y)
            if isinstance(item, tuple):
                if len(item) >= 3:
                    x, y, _idx_global = item[:3]
                elif len(item) == 2:
                    x, y = item
                else:
                    raise ValueError(f"Unexpected tuple length in ListHandler: {len(item)}")
            else:
                # 如果你的代码构造的是 dict 形式，也做个兜底
                if isinstance(item, dict):
                    x = item.get('x')
                    y = item.get('y', -1)
                else:
                    raise TypeError(f"Unexpected item type in ListHandler: {type(item)}")

            # 标签转 int，DataLoader 会堆成 LongTensor
            y = int(y)

            # 本地索引（Strategy 期望第三个返回值是“本地 idx”）
            idx_local = i

            # 可选变换（把 numpy → tensor 等）
            x = self.transform(x) if self.transform is not None else x

            return x, y, idx_local  # 返回【局部】idx：0..len(self.lst)-1

    # ============== 构建“可标注全集”（label_universe） ==================
    # 优先使用 --shots（每类定量），否则保持原有 idx_file / frac 逻辑
    shots = getattr(args, 'shots', None)  # 每类 shots
    lu_file = getattr(args, 'label_universe_idx_file', None)
    lu_frac = float(getattr(args, 'label_universe_frac', 1.0))
    lu_seed = int(getattr(args, 'label_universe_seed', 42))

    # 取目标域 train 的标签数组
    if is_wisig:
        # WiSig 分支：直接从 ds_train 抽取标签
        y_np = np.array([ds_train[i][1] for i in range(len(ds_train))], dtype=np.int64)
    else:
        # 旧分支：train_y 是路径或数组
        if isinstance(train_y, str):
            y_np = np.load(train_y)
        else:
            y_np = np.asarray(train_y, dtype=np.int64)
    assert y_np.ndim == 1, "train_y / ds_train 应提供一维标签数组"

    # ★ 用 lu_seed 构造本地 RNG，不要再改全局 np.random 的种子
    rng = np.random.RandomState(lu_seed)
    all_idx = np.arange(n_pool, dtype=np.int64)

    # ===== 初始化 shot 可标注池（label universe） =====
    shots_per_class = args.shots  # 20
    label_universe = []

    # 按类别均匀采样，构成可标注池
    for c in range(args.class_num):  # 16类
        cls_idx = np.where(y_np == c)[0]  # 获取所有类为 c 的样本下标
        assert len(cls_idx) >= shots_per_class, \
            f"类 {c} 样本不足 shots={shots_per_class}"

        # ★ 使用 rng.choice，而不是全局 np.random.choice
        chosen = rng.choice(cls_idx, size=shots_per_class, replace=False)
        label_universe.extend(chosen.tolist())

    label_universe = np.array(sorted(label_universe), dtype=np.int64)
    label_universe_set = set(label_universe.tolist())

    print(f"[Init] Initial few-shot labeled set = {len(label_universe)} samples "
            f"({args.shots}-shot × {args.class_num} classes)")

    # === New SF-FSDA setting ===
    # label_universe is used as the initial few-shot labeled target set.
    initial_labeled_idx = label_universe.copy()
    args._initial_labeled_idx = initial_labeled_idx  # 用于 mixup_seed_only

    labeled_mask = np.zeros(n_pool, dtype=bool)
    labeled_mask[initial_labeled_idx] = True

    # 模型 + 源域权重
    netF, netB, netC = build_model(args, device)
    netF.eval(); netB.eval(); netC.eval()

    # -----------------------------------------------------
    # 在这里添加 UCT 的 EMA Teacher 初始化
    # -----------------------------------------------------
    if getattr(args, 'use_uct', False):
        print("[UCT] Initialize EMA teacher networks (momentum=0.999)")

        # ✅ 使用重新构建 + 加载参数，避免 weight_norm 引发 deepcopy 错误
        ema_F, ema_B, ema_C = build_model(args, device)

        # 从原网络复制参数
        ema_F.load_state_dict(netF.state_dict())
        ema_B.load_state_dict(netB.state_dict())
        ema_C.load_state_dict(netC.state_dict())

        # 冻结 Teacher 参数（不参与梯度更新）
        for p in list(ema_F.parameters()) + list(ema_B.parameters()) + list(ema_C.parameters()):
            p.requires_grad_(False)

    else:
        ema_F = ema_B = ema_C = None

    # 取一个样本用于估计 embedding 维度（放入 NetWrap 构造）
    #sample_x, _, _ = ds_train[0]
    sample_item = ds_train[0]
    sample_x = sample_item[0]

    net_wrap = NetWrap(netF, netB, netC, device, sample_x)

    # CAS 只允许在 Label-Universe 内选样：构造 CAS 专用的 “已标注掩码”
    cas_idxs_lb = np.zeros(n_pool, dtype=bool)
    not_in_lu = np.ones(n_pool, dtype=bool)
    not_in_lu[label_universe] = False
    cas_idxs_lb[not_in_lu] = True  # LU 之外一律置 True = 视为“不可选”（被锁死）

    # 给 Strategy/CAS 需要的若干默认参数（如果 argparse 没提供的话）
    if not hasattr(args, 'worker'): args.worker = 0
    if not hasattr(args, 'output_dir'): args.output_dir = getattr(args, 'output', '.')
    if not hasattr(args, 'seed'): args.seed = 42
    if not hasattr(args, 'cd_ratio'): args.cd_ratio = 1.0
    if not hasattr(args, 'lambda_topk'): args.lambda_topk = 0.0
    if not hasattr(args, 'topk'): args.topk = 500

    # === 在 al_strategy = CASampling(...) 这一段的正上方插入 ===
    # Strategy 训练会用到的一组关键超参（若已有同名字段则保留你现有值）
    _defaults = {
        'beta_im': 1.0,  # 信息最大化权重（>0 表示启用半监督 InfoMax）
        'beta_vpa': 0.5,  # VPA 注意力熵权重（未标注与标注原型的注意力分布更尖锐）
        'lambda_mixup': 0.3,  # Mixup 权重（小样本强烈建议开启）
        'mixup_alpha': 0.2,  # Mixup Beta 分布参数
        'sfada_ubl': 2,  # 每轮采样未标注样本的倍率（≈ 已标注的2倍；-1 表示用尽）
        'mem_momentum': 0.9,  # 记忆库动量（0.9~0.99 常用）
        'beta_pgra': 1.0,
        'gamma': 0.1,
        'tau': 0.5,
    }
    for k, v in _defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Strategy.print_log 需要一个可写的 out_file 句柄
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    if not hasattr(args, 'out_file') or args.out_file is None:
        args.out_file = open(os.path.join(args.output_dir, 'train_strategy.log'), 'a', buffering=1)

    n_pool = len(ds_train)

    # Initial few-shot labeled samples are marked as labeled.
    cas_idxs_lb = labeled_mask.copy()

    # 如果你确实有“先验已标注”的极少量样本（例如 warm start），就在这里仅标记那一小撮：
    # cas_idxs_lb[seed_labeled_indices] = True

    al_strategy = CASampling(
        train_dataset=ds_train,
        idxs_lb=cas_idxs_lb,  # CAS 内部维护的“已标注”掩码（含 LU 外锁死）
        net=net_wrap,
        handler=ListHandler,  # 把 list → Dataset
        train_transform=iq_train_transform(),
        test_transform=iq_test_transform(),
        args=args,
        source_dataset=None
    )

    al_strategy.label_universe = label_universe

    # Initialize few-shot class-center anchors
    center_counts = al_strategy.init_class_centers()
    print("[Center Init] class counts:", center_counts.tolist())
    print("[Center Init] class_centers shape:", tuple(al_strategy.class_centers.shape))

    # 把 Teacher 网络挂到 strategy 上
    if getattr(args, 'use_uct', False):
        al_strategy.ema_F = ema_F
        al_strategy.ema_B = ema_B
        al_strategy.ema_C = ema_C
    print("[DEBUG] init labeled:", int(al_strategy.idxs_lb.sum()),
          "unlabeled:", int((~al_strategy.idxs_lb).sum()))

    # 测试集加载器
    test_loader = DataLoader(ds_test, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False)
    print("[DBG] eval loader = test, size =", len(ds_test))

    # 优化器（按源域的分层 lr 配置）
    param_group = []
    for _, p in netF.named_parameters(): param_group += [{'params': p, 'lr': args.lr_backbone}]
    for _, p in netB.named_parameters(): param_group += [{'params': p, 'lr': args.lr_head}]
    for _, p in netC.named_parameters(): param_group += [{'params': p, 'lr': args.lr_head}]
    optimizer = torch.optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    def _restore_best_if_exists():
        """若存在上轮 best_*_adapt.pt，则在本轮开始前恢复以避免漂移。"""
        pF = osp.join(args.output_dir, 'best_F_adapt.pt')
        pB = osp.join(args.output_dir, 'best_B_adapt.pt')
        pC = osp.join(args.output_dir, 'best_C_adapt.pt')
        if osp.exists(pF) and osp.exists(pB) and osp.exists(pC):
            netF.load_state_dict(torch.load(pF, map_location=device))
            netB.load_state_dict(torch.load(pB, map_location=device))
            netC.load_state_dict(torch.load(pC, map_location=device))
            print('[RESTORE] loaded best_*_adapt.pt for a stable start of this round')

    # Round 0：直接在目标测试集上评估基线
    acc0, _ = cal_acc(
        test_loader, netF, netB, netC,
        args=args,
        round_idx=round,
        debug=False
    )

    best = acc0
    os.makedirs(args.output_dir, exist_ok=True)
    # 保存 label_universe，确保可复现
    np.save(osp.join(args.output_dir, 'label_universe.npy'), label_universe)

    print(f"[Round 0] Acc={acc0:.2f}% | save initial as best")
    acc_history.append(acc0)  # [新增] 记录Round0

    torch.save(netF.state_dict(), osp.join(args.output_dir, 'best_F_adapt.pt'))
    torch.save(netB.state_dict(), osp.join(args.output_dir, 'best_B_adapt.pt'))
    torch.save(netC.state_dict(), osp.join(args.output_dir, 'best_C_adapt.pt'))

    # 当前未标注索引（全集）
    unlabeled_idx = all_idx.copy()

    # 多轮主动学习 + 半监督自适应
    rounds_no_improve = 0
    for rd in range(1, args.num_round + 1):

        # ================================================================
        # Current method:
        # shots        = initial labeled samples per class
        # query_budget = number of CAS candidates selected per round
        # ================================================================
        remain_unlabeled = int(np.sum(al_strategy.idxs_lb == 0))

        query_budget = getattr(args, 'query_budget', None)
        if query_budget is None:
            query_budget = int(args.shots) if getattr(args, 'shots', None) is not None else max(
                1, int(round(args.ratio_per_round * n_pool))
            )

        B = min(int(query_budget), remain_unlabeled)

        print(f"[Round {rd}] CAS query budget B={B} | remaining unlabeled={remain_unlabeled}")

        if B <= 0:
            print(f"[Round {rd}] No unlabeled samples left. Stop adaptation.")
            break

        if getattr(args, 'abl_cas', 'cas') == 'cas':
            warmup_rounds = getattr(args, 'warmup_rounds', 2)

            if rd <= warmup_rounds:
                # 预热阶段：仅在 Label-Universe 内随机
                lu_unl = np.array([i for i in label_universe if not labeled_mask[i]], dtype=int)
                if len(lu_unl) == 0:
                    raise RuntimeError("LU 未标注已为空，无法预热选样")
                select = np.random.choice(lu_unl, size=min(B, len(lu_unl)), replace=False)
                print(f"[Round {rd}] Warmup(random-in-LU) select {len(select)}")
            else:
                # 正常阶段：CAS query
                select = al_strategy.query(B)
                if select is None or len(select) == 0:
                    # 兜底：CAS 出问题回退 LU 随机
                    lu_unl = np.array([i for i in label_universe if not labeled_mask[i]], dtype=int)
                    select = np.random.choice(lu_unl, size=min(B, len(lu_unl)), replace=False)
                    print(f"[Round {rd}] Fallback to random-in-LU {len(select)}")

            # === New: prototype-consistency pseudo labeling ===
            candidate_idx = np.array(select, dtype=int)

            confirmed_idx, pseudo_y = al_strategy.confirm_by_prototype(candidate_idx)

            if len(confirmed_idx) > 0:
                al_strategy.add_pseudo_labels(confirmed_idx, pseudo_y)
                labeled_mask[confirmed_idx] = True
                al_strategy.idxs_lb[confirmed_idx] = True

            print(f"[Round {rd}] CAS candidates={len(candidate_idx)} | "
                  f"confirmed pseudo labels={len(confirmed_idx)} | "
                  f"total labeled={int(al_strategy.idxs_lb.sum())}")


        else:
            # 仅在 Label-Universe(LU) 中挑选当前仍未标注的候选，保证 shots 公平
            lu_unl = np.array([i for i in label_universe if not labeled_mask[i]], dtype=int)
            if len(lu_unl) == 0:
                print(f"[Round {rd}] LU 未标注为空，停止选样")
                break
            B_eff = min(B, len(lu_unl))

            if args.abl_cas == 'random':
                # 在 LU 未标注集合内随机（与 CAS 同一候选范围，标注预算严格一致）
                rng = np.random.default_rng(int(getattr(args, 'mc_seed_base', 0)) + rd)

                if len(lu_unl) >= B_eff:
                    select = rng.choice(lu_unl, size=B_eff, replace=False)
                else:
                    # LU 剩余不足时，把余下的全取走
                    select = lu_unl


            else:
                # 基于模型概率的不确定性基线需要先拿 LU 未标注的预测概率
                unl_list = [ds_train[i] for i in lu_unl]  # List[(x,y,idx)]
                probs = al_strategy.predict_prob(unl_list)  # (N, C)
                if hasattr(probs, 'cpu'):
                    probs = probs.cpu().numpy()

                # 置信度、margin、熵
                top2 = np.partition(probs, -2, axis=1)[:, -2:]
                margin_gap = top2[:, -1] - top2[:, -2]  # top1-top2
                ent = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)

                if args.abl_cas == 'entropy':
                    order = np.argsort(-ent)  # 最大熵优先
                    select = lu_unl[order[:B_eff]]

                elif args.abl_cas == 'margin':
                    order = np.argsort(margin_gap)  # 最小间隔优先
                    select = lu_unl[order[:B_eff]]

                elif args.abl_cas == 'uniform_pred':
                    # 按“预测类别”尽量均衡地随机抽
                    preds = probs.argmax(axis=1)
                    classes = np.unique(preds)
                    per = max(1, int(np.ceil(B_eff / len(classes))))
                    chosen = []
                    for c in classes:
                        pool_c = lu_unl[preds == c]
                        if len(pool_c) == 0:
                            continue
                        take = min(per, len(pool_c), B_eff - len(chosen))
                        if take > 0:
                            chosen.extend(np.random.choice(pool_c, size=take, replace=False).tolist())
                        if len(chosen) >= B_eff:
                            break
                    # 不足则从剩余里随机补齐
                    if len(chosen) < B_eff:
                        rest = np.setdiff1d(lu_unl, np.array(chosen, dtype=int), assume_unique=False)
                        if len(rest) > 0:
                            add = np.random.choice(rest, size=(B_eff - len(chosen)), replace=False).tolist()
                            chosen.extend(add)
                    select = np.array(chosen, dtype=int)

                else:
                    raise ValueError(f"Unknown abl_cas={args.abl_cas}")
        # ===== CAS Ablation End =====

        # 同步 CAS 内部的已标注掩码（这行保留不变）
        #al_strategy.update(select)

        # 同步到训练侧
        #labeled_mask[select] = True
        unlabeled_idx = all_idx[~labeled_mask]
        print(f"[Round {rd}] Query B={len(select)}. Labeled now: {int(labeled_mask.sum())} / {n_pool}")

        # ====== 使用 Strategy 的训练（替换自写 DataLoader+训练循环+评估） ======

        # 3) Strategy 训练：当前方法固定使用 Mixup + PGRA，需要未标注数据
        al_strategy.train_one_round_with_unlabel(rd)

        center_counts = al_strategy.update_class_centers_ema()
        print(f"[Center EMA] Round {rd}: counts={center_counts.tolist()}")

        # 4) Strategy 评测（其接口期望 list 形式的数据集）
        test_list = [ds_test[i] for i in range(len(ds_test))]
        val_acc = al_strategy.predict(test_list, rd)
        print(f"[Round {rd}] Acc={val_acc:.2f}%")
        acc_history.append(val_acc)

        acc_ck, _ = cal_acc(
            test_loader, netF, netB, netC,
            args=args,  # 传入输出路径
            round_idx=round,  # 当前轮次
            debug=True  # 开启写混淆矩阵文件
        )

        print(f"[CHECK] cal_acc(test_loader) = {acc_ck:.2f}%, strategy.predict = {val_acc:.2f}%")
        # 两个指标应非常接近；若相差过大，优先检查 strategy.predict 的实现（是否用了错集/含增广）

        # 5) 通过 Strategy.save 保存（内部会调 NetWrap.save 写 F/B/C）
        if val_acc > best:
            best = val_acc
            rounds_no_improve = 0
            al_strategy.save(rd, args)
            print(f"[BEST] New best {best:.2f}% at round {rd}")
        else:
            rounds_no_improve += 1

        # 早停：连续 N 轮没有提升就提前终止
        early_stop_patience = getattr(args, "early_stop_patience", 0)
        if early_stop_patience > 0 and rounds_no_improve >= early_stop_patience and rd >= 4:
            print(f"[Early Stop] No improvement for {early_stop_patience} rounds. Stopped at round {rd}.")
            break

    print(f"[DONE] Best target acc = {best:.2f}% | saved under {args.output_dir}")
    meta = {
        "name": args.name,
        "seed": int(getattr(args, "label_universe_seed", -1)),
        "shots": int(getattr(args, "shots", -1)) if getattr(args, "shots", None) is not None else None,
        "lu_size": int(len(label_universe)) if 'label_universe' in locals() else None,
        "n_pool": int(n_pool),
        "num_round": int(args.num_round),
    }
    try:
        meta["b_per_round"] = int(round(args.ratio_per_round * n_pool))
    except Exception:
        meta["b_per_round"] = None

    meta["time_sec"] = round(time.time() - start_time, 2)
    write_run_summary(args.output_dir, meta, acc_history, best)
    return best, acc_history


def main() -> None:
    """解析命令行参数并启动目标域自适应。"""
    p = argparse.ArgumentParser(
        description='Source-Free Few-Shot Domain Adaptation for RFFI with CAS + PGRA'
    )
    p.add_argument('--gpu_id', type=str, default='0', help='GPU 编号；-1 为 CPU')

    # === RF 数据相关 ===
    p.add_argument('--rf_root', type=str, required=True, help='ORACLE-S 根目录（包含 S1/S2/…）')
    p.add_argument('--s_folder', type=str, default='S1', help='源域子目录名称（如 S1）')
    p.add_argument('--t_folder', type=str, default='S2', help='目标域子目录名称（如 S2）')
    p.add_argument('--rf_ft', type=str, default='62ft', help='特征版本（如 32ft/62ft）')

    # === 模型结构 ===
    p.add_argument('--in_channels', type=int, default=2, help='输入通道数（IQ=2）')
    p.add_argument('--channels', type=int, default=16, help='MACNN 基础通道数')
    p.add_argument('--class_num', type=int, default=16, help='类别数')
    p.add_argument('--bottleneck', type=int, default=512, help='颈部维度')
    p.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'], help='分类头类型（wn/linear）')
    p.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'], help='颈部类型（bn/ori）')

    # === 优化与调度 ===
    p.add_argument('--batch_size', type=int, default=64, help='批大小')
    p.add_argument('--max_epoch', type=int, default=2, help='目标域每轮训练 epoch 数')
    p.add_argument('--lr_backbone', type=float, default=1e-5, help='Backbone 学习率')
    p.add_argument('--lr_head', type=float, default=5e-4, help='分类头学习率')
    p.add_argument('--gamma', type=float, default=0.1, help='PGRA 类中心 EMA 更新系数')
    p.add_argument('--smooth', type=float, default=0.05, help='标签平滑系数')
    p.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值')

    # === 适应轮次与未标注数据使用 ===
    p.add_argument('--num_round', type=int, default=8, help='适应轮数')
    p.add_argument('--ratio_per_round', type=float, default=0.04, help='每轮候选样本比例/预算')
    p.add_argument('--freeze_f_rounds', type=int, default=2, help='前 K 轮冻结 backbone')
    p.add_argument('--ul_ratio', type=float, default=0.2, help='每轮用于 PGRA 的未标注样本比例')

    # === 初始化与输出 ===
    p.add_argument('--init_pool', type=str, default='source', choices=['source'], help='初始化来源')
    p.add_argument('--source_dir', type=str, required=True, help='源域最佳权重目录（含 best_* 或 source_*）')
    p.add_argument('--output', type=str, default='exps_adapt', help='输出根目录')
    p.add_argument('--output_dir', type=str, default=None, help='可选：输出目录')
    p.add_argument('--name', type=str, default='macnn_rf62', help='实验名')

    # === 初始 few-shot 标记集与 MC 实验 ===
    p.add_argument('--label_universe_frac', type=float, default=1.0,
                   help='兼容旧参数；当前方法中主要使用 --shots 构造初始 few-shot 标记集')
    p.add_argument('--label_universe_idx_file', type=str, default=None,
                   help='兼容旧参数：可选的索引文件')
    p.add_argument('--label_universe_seed', type=int, default=42,
                   help='初始 few-shot 标记集采样随机种子')
    p.add_argument('--shots', type=int, default=None,
                   help='每类初始标记目标样本数，例如 5 或 10')
    p.add_argument('--mc_runs', type=int, default=1, help='Monte Carlo 重复次数')
    p.add_argument('--mc_seed_base', type=int, default=30, help='Monte Carlo 基础种子')

    p.add_argument('--use_val', action='store_true', help='若目标域目录包含 x_val/y_val，则使用验证集')
    p.add_argument('--make_val_from_train', action='store_true',
                   help='当 val 为空时，从 train 中分层切分验证集')
    p.add_argument('--val_ratio', type=float, default=0.10, help='验证集比例')
    p.add_argument('--val_seed', type=int, default=42, help='验证集切分随机种子')

    # === WiSig 参数 ===
    p.add_argument('--dataset', type=str, default='wisig', help='数据集类型：wisig / rf')
    p.add_argument('--dataset_path', type=str, default=r'D:\pydata\Datasets', help='WiSig 主目录')
    p.add_argument('--wisig_sub', type=str, default='wisig', help='WiSig 子目录')
    p.add_argument('--tgt_dates', type=str, default='2,3,4', help='目标域日期列表')
    p.add_argument('--wisig_val_ratio', type=float, default=0.10, help='目标域验证集比例')
    p.add_argument('--wisig_test_ratio', type=float, default=0.20, help='目标域测试集比例')
    p.add_argument('--query_budget', type=int, default=None,
                   help='每轮 CAS 选择的候选样本数；若为空，则默认等于 shots')

    # === 当前论文方法的损失权重 ===
    # 旧 LFTL 损失关闭：InfoMax / VPA 不再作为主方法使用
    p.add_argument('--beta_im', type=float, default=0.0, help='旧 InfoMax 权重，当前方法默认关闭')
    p.add_argument('--beta_vpa', type=float, default=0.0, help='旧 VPA 权重，当前方法默认关闭')
    p.add_argument('--beta_pgra', type=float, default=1.0, help='PGRA 损失权重')
    p.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup Beta 分布参数')
    p.add_argument('--mixup_prob', type=float, default=0.5,
                   help='每个 labeled batch 执行 Mixup 的概率；其余 batch 使用原始监督CE')
    p.add_argument('--lambda_mixup', type=float, default=0.0,
                   help='兼容旧 Mixup 额外训练接口；当前 Mixup 已并入主损失，默认关闭')
    p.add_argument('--tau', type=float, default=0.5, help='PGRA center-affinity temperature')
    p.add_argument('--sfada_ubl', type=int, default=2, help='兼容旧参数')
    p.add_argument('--mem_momentum', type=float, default=0.9, help='兼容旧参数')

    # === 选样策略（含消融）===
    p.add_argument('--abl_cas', type=str, default='cas',
                   choices=['cas', 'random', 'entropy', 'margin', 'uniform_pred'],
                   help='选样消融')
    p.add_argument('--cd_ratio', type=float, default=1.0,
                   help='α：CAS 跨轮对比权重')
    p.add_argument('--uct_lambda', type=float, default=0.5,
                   help='λ：CAS 类级转移性项权重')
    p.add_argument('--uct_kappa', type=int, default=-1,
                   help='κ：UCT 统计 top-κ 样本数；-1 表示自动设置')
    p.add_argument('--warmup_rounds', type=int, default=0,
                   help='CAS 预热轮数；当前方法建议设为 0')
    p.add_argument('--ucm_off', type=int, default=0, choices=[0, 1],
                   help='0: normal CAS; 1: UCT-only')
    p.add_argument('--shot_schedule', type=str, default=None,
                   help='每轮候选数量，逗号分隔')
    # labeled_mask[select] = True

    # === 兼容旧实验开关 ===
    p.add_argument('--use_uct_loss', action='store_true',
                   help='兼容旧参数，当前主方法默认不使用')
    p.add_argument('--lent_off', action='store_true',
                   help='兼容旧参数，当前主方法默认无 Lent')





    args = p.parse_args()

    # 当前主方法不使用 InfoMax/Lent，这里仅保留兼容字段
    args.beta_im_gate = 0.0

    if getattr(args, 'output_dir', None) in (None, ''):
        args.output_dir = args.output

    tag = f"{args.s_folder}-{args.t_folder}-{args.rf_ft}"
    args.output_dir = osp.join(
        args.output,
        'rf',
        f"b{args.ratio_per_round}x{args.num_round}",
        tag,
        args.name
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print('[LOG dir]:', args.output_dir)

    # === Monte Carlo 外层循环 ===
    base_name = args.name
    tag = f"{args.s_folder}-{args.t_folder}-{args.rf_ft}"
    all_rows = []
    all_best = []

    for i in range(args.mc_runs):
        run_seed = args.mc_seed_base + i
        set_global_seed(run_seed)
        args.label_universe_seed = run_seed
        args.name = f"{base_name}_mc{i:02d}"

        args.output_dir = osp.join(
            args.output,
            'rf',
            f"b{args.ratio_per_round}x{args.num_round}",
            tag,
            args.name
        )
        os.makedirs(args.output_dir, exist_ok=True)
        print('[LOG dir]:', args.output_dir)

        best_acc, acc_hist = adapt_target(args)

        all_best.append(best_acc)
        row = {
            "run": i,
            "seed": run_seed,
            "name": args.name,
            "best_acc": round(float(best_acc), 2),
            "acc_last": round(float(acc_hist[-1]), 2),
            "acc_mean_last3": round(float(np.mean(acc_hist[-3:])), 2)
            if len(acc_hist) >= 3 else round(float(acc_hist[-1]), 2)
        }
        all_rows.append(row)

    # === MC 汇总 ===
    summary_root = osp.join(
        args.output,
        'rf',
        f"b{args.ratio_per_round}x{args.num_round}",
        tag,
        f"{base_name}_MC_SUMMARY"
    )
    os.makedirs(summary_root, exist_ok=True)

    mean_acc = float(np.mean(all_best)) if len(all_best) > 0 else 0.0
    std_acc = float(np.std(all_best, ddof=1)) if len(all_best) > 1 else 0.0

    with open(osp.join(summary_root, "mc_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "name", "best_acc", "acc_last", "acc_mean_last3"])
        for r in all_rows:
            w.writerow([r["run"], r["seed"], r["name"], r["best_acc"], r["acc_last"], r["acc_mean_last3"]])
        w.writerow([])
        w.writerow(["mean(best_acc)", f"{mean_acc:.2f}"])
        w.writerow(["std(best_acc)", f"{std_acc:.2f}"])

    with open(osp.join(summary_root, "mc_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "mc_runs": len(all_best),
            "mean_best_acc": round(mean_acc, 2),
            "std_best_acc": round(std_acc, 2),
            "rows": all_rows
        }, f, ensure_ascii=False, indent=2)

    with open(osp.join(summary_root, "mc_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"MC runs={len(all_best)}\n")
        f.write(f"mean(best_acc)={mean_acc:.2f}%\n")
        f.write(f"std(best_acc)={std_acc:.2f}%\n\n")
        for r in all_rows:
            f.write(
                f"run={r['run']:02d} seed={r['seed']} "
                f"best={r['best_acc']:.2f}% "
                f"last={r['acc_last']:.2f}% "
                f"last3-mean={r['acc_mean_last3']:.2f}% "
                f"name={r['name']}\n"
            )

    print(f"[MC] mean={mean_acc:.2f}% | std={std_acc:.2f}%")
    print(f"[MC] summary saved to: {summary_root}")
if __name__ == '__main__':
    main()
