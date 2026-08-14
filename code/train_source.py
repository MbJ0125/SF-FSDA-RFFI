# -*- coding: utf-8 -*-
"""
文件：train_source.py（中文注释版，独立）
用途：在 ORACLE-S 的源域（例如 S1）上进行监督训练，得到强初始化（netF/netB/netC）。

主要流程：
1）加载 ORACLE-S 的 x_train_*.npy / y_train_*.npy、x_test_*.npy / y_test_*.npy
2）构建 MACNN 作为特征提取器（netF） + FeatureNeck（netB） + ClassifierHead（netC）
3）监督交叉熵（支持标签平滑）训练，分层学习率（F×0.1，B/C×1.0）
4）周期性评估，若更优则记录 best；最终保存 source_F/B/C.pt

注释风格：
- 为函数/类提供 docstring（PyCharm 可在调用处/悬浮查看）
- 为关键逻辑提供行内中文注释（便于快速理解实现与论文的对应）
"""

import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import network, loss
from loss import cross_entropy_loss  # 你文件里的 CE 实现（也可用 F.cross_entropy）
from dataset import IQNPYDataset, iq_train_transform, iq_test_transform
from model.CNNmodel import MACNN
import os.path as osp
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class NumpyIQDataset(Dataset):
    """直接用内存里的 numpy 数组构造数据集，便于自定义划分后送 DataLoader。"""
    def __init__(self, X: np.ndarray, Y: np.ndarray, transform=None):
        assert len(X) == len(Y)
        self.X = X
        self.Y = np.asarray(Y, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.Y[idx])

        # 可选的预处理/增强：保持你原来的 iq_train_transform / iq_test_transform
        if self.transform is not None:
            x = self.transform(x)

        # 若 transform 返回 numpy，则转成张量
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # 保证类型
        if isinstance(y, np.generic):
            y = int(y)
        return x, y

def _try_wisig_file(root, sub, date):
    # 兼容两种命名：rx_1-1_dateX.pkl（dataloader里常用）与 rx_1_1_dateX.pkl（你截图）
    p1 = osp.join(root, sub, f"rx_1-1_date{date}.pkl")
    p2 = osp.join(root, sub, f"rx_1_1_date{date}.pkl")
    return p1 if osp.exists(p1) else p2

def data_load(args):
    """
    WiSig / WiSig-EQ 分支（优先）：
      - 用你给的 wisig_dataloader 直接读取 .pkl（data1）
      - 先按 9:1 切 S1 的 train/test；再按 train_frac/test_frac 细分
    兼容分支（原 .npy）：
      - 完全保持你原来的做法
    """
    assert 0.0 < args.train_frac <= 1.0, "train_frac 必须在 (0,1] 内"
    assert 0.0 < args.test_frac  <= 1.0, "test_frac 必须在 (0,1] 内"

    # ---------- WiSig / WiSig-EQ 分支 ----------
    if getattr(args, "dataset", "") in ["wisig", "wisig-eq"]:
        from wisig_dataloader import load_single_dataset, Dataset_
        root = getattr(args, "dataset_path", r"D:\pydata\Datasets")
        sub  = getattr(args, "wisig_sub", "wisig")
        date = str(getattr(args, "src_date", "1"))
        src_pkl = os.path.join(root, sub, f"rx_1-1_date{date}.pkl")
        assert os.path.exists(src_pkl), f"[WiSig] 源域文件不存在：{src_pkl}"

        # 读取 WiSig/WiSig-EQ：返回 X:[N,2,L], Y:[N]
        X_all, Y_all = load_single_dataset(src_pkl, num_device=args.class_num)

        # 先 9:1 切成 train/test
        X_tr_all, X_te_all, Y_tr_all, Y_te_all = train_test_split(
            X_all, Y_all, test_size=0.1, random_state=args.split_seed, stratify=Y_all
        )

        # 从 train 抽 train_frac
        if args.train_frac < 1.0:
            X_train, _, Y_train, _ = train_test_split(
                X_tr_all, Y_tr_all,
                test_size=1.0 - args.train_frac,
                random_state=args.split_seed,
                stratify=Y_tr_all
            )
        else:
            X_train, Y_train = X_tr_all, Y_tr_all

        # 从 test 抽 test_frac
        if args.test_frac < 1.0:
            X_val, X_test, Y_val, Y_test = train_test_split(
                X_te_all, Y_te_all,
                test_size=args.test_frac,
                random_state=args.split_seed,
                stratify=Y_te_all
            )
        else:
            X_test, Y_test = X_te_all, Y_te_all
            X_val,  Y_val  = X_te_all[:0], Y_te_all[:0]

        # Dataset
        ds_tr = Dataset_('train', X_train, Y_train.astype(np.int64))
        ds_va = Dataset_('val',   X_val,   Y_val.astype(np.int64))
        ds_te = Dataset_('test',  X_test,  Y_test.astype(np.int64))

        dls = {
            "source_tr": DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=False),
            "source_te": DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False),
            "test":      DataLoader(ds_te, batch_size=args.batch_size*2, shuffle=False, num_workers=0, drop_last=False),
        }

        print(f"[WiSig-Source] train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}  file={src_pkl}")
        return dls


    # ---------- 兼容原 .npy 分支（保持你现在的逻辑不变） ----------
    root = args.rf_root
    ft = args.rf_ft

    # 1) 载入 S1 的 train 与 test
    X_tr_all = np.load(osp.join(root, f"x_train_{ft}.npy"))
    Y_tr_all = np.load(osp.join(root, f"y_train_{ft}.npy")).astype(np.int64)
    X_te_all = np.load(osp.join(root, f"x_test_{ft}.npy"))
    Y_te_all = np.load(osp.join(root, f"y_test_{ft}.npy")).astype(np.int64)

    # 2) 从 S1 train 中抽取 train_frac 作为训练（其余丢弃，不再使用）
    if args.train_frac < 1.0:
        X_train, _, Y_train, _ = train_test_split(
            X_tr_all, Y_tr_all,
            test_size=1.0 - args.train_frac,
            random_state=args.split_seed,
            stratify=Y_tr_all
        )
    else:
        X_train, Y_train = X_tr_all, Y_tr_all

    # 3) 从 S1 test 中抽取 test_frac 作为测试，剩余全作为验证
    if args.test_frac < 1.0:
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_te_all, Y_te_all,
            test_size=args.test_frac,
            random_state=args.split_seed,
            stratify=Y_te_all
        )
    else:
        X_test, Y_test = X_te_all, Y_te_all
        X_val,  Y_val  = X_te_all[:0], Y_te_all[:0]

    # 4) 包成 Dataset/DataLoader（沿用你的 transform）
    ds_tr = NumpyIQDataset(X_train, Y_train, transform=iq_train_transform())
    ds_va = NumpyIQDataset(X_val,   Y_val,   transform=iq_test_transform())
    ds_te = NumpyIQDataset(X_test,  Y_test,  transform=iq_test_transform())

    dls = {
        "source_tr": DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=False),
        "source_te": DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False),
        "test":      DataLoader(ds_te, batch_size=args.batch_size*2, shuffle=False, num_workers=0, drop_last=False),
    }
    print("--------DataLoader (RF, S1-train 抽样 + S1-test 再划分)-------")
    print(f"train: {len(ds_tr)}, val(from S1-test rest): {len(ds_va)}, test(from S1-test frac): {len(ds_te)}.")
    return dls



class MacnnBackbone(nn.Module):
    """MACNN 适配为特征提取器（netF）。"""
    def __init__(self, in_channels: int = 2, channels: int = 16, num_classes: int = 16):
        super().__init__()
        self.core = MACNN(in_channels=in_channels, channels=channels, num_classes=num_classes)
        self.in_features = getattr(self.core, 'emb_dim', channels * 12)
        print(f"[LFTL] Using MACNN backbone (in_channels={in_channels}, channels={channels}, emb_dim={self.in_features})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.core, 'forward_features'):
            return self.core.forward_features(x)
        out = self.core(x)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            feat, _ = out
            return feat
        raise RuntimeError('MACNN.forward 未返回 (emb, logits)，请在 MACNN 实现 forward_features()。')


def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    """为每个 param_group 记录初始 lr（用于自定义 lr_scheduler）。"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer: optim.Optimizer, iter_num: int, max_iter: int, gamma: float = 10, power: float = 0.75) -> optim.Optimizer:
    """多项式衰减学习率（poly），与原版 LFTL/SHOT 类似。"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def split_xy(batch):
    """通用 batch 解包：兼容 dict 与 (x,y,...)。"""
    if isinstance(batch, dict):
        x = batch.get('x') or batch.get('data') or batch.get('inputs') or next(iter(batch.values()))
        for ky in ('y', 'label', 'labels', 'target', 'targets'):
            if ky in batch:
                y = batch[ky]
                break
        else:
            raise KeyError('字典 batch 中未找到标签键')
        return x, y
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError('batch 长度 < 2')
        return batch[0], batch[1]
    raise TypeError(type(batch))


def cal_acc(loader: DataLoader, netF: nn.Module, netB: nn.Module, netC: nn.Module, flag: bool = False):
    """在验证/测试集上计算准确率（以及可选的各类准确率）。"""
    dev = next(netF.parameters()).device
    start_test = True
    with torch.no_grad():
        for batch in loader:
            inputs, labels = split_xy(batch)
            inputs = inputs.to(dev)
            labels = labels.to(dev).long()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.detach().cpu(); all_label = labels.detach().cpu(); start_test = False
            else:
                all_output = torch.cat((all_output, outputs.detach().cpu()), 0)
                all_label  = torch.cat((all_label,  labels.detach().cpu()), 0)
    all_output = torch.softmax(all_output.float(), dim=1)
    predict = torch.argmax(all_output, dim=1)
    accuracy = (predict == all_label).float().mean().item() * 100.0
    # 平均熵（可选用于观测信心）
    mean_ent = (-(all_output.clamp_min(1e-12) * all_output.clamp_min(1e-12).log()).sum(dim=1)).mean().item()
    if flag:
        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(all_label.numpy(), predict.numpy())
        cls_acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = cls_acc.mean()
        acc_str = ' '.join([str(np.round(i, 2)) for i in cls_acc])
        return aacc, acc_str
    else:
        return accuracy, mean_ent


def train_source(args):
    """源域（S1）监督训练主流程：CE（支持标签平滑）+ 分层学习率 + 定期评估保存 best。

    返回：训练完成后的 (netF, netB, netC)
    """
    dls = data_load(args)

    # 根据 --net 构建特征提取器
    if args.net.lower() == 'macnn':
        netF = MacnnBackbone(in_channels=args.in_channels, channels=args.channels, num_classes=args.class_num).cuda()
    elif args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    else:
        raise ValueError(f"未知的 --net: {args.net}")

    netB = network.FeatureNeck(feature_dim=netF.in_features, bottleneck_dim=args.bottleneck, type=args.classifier).cuda()
    netC = network.ClassifierHead(class_num=args.class_num, bottleneck_dim=args.bottleneck, type=args.layer).cuda()

    # 分层学习率：F×0.1，B/C×1.0
    param_group = []
    for _, v in netF.named_parameters(): param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for _, v in netB.named_parameters(): param_group += [{'params': v, 'lr': args.lr}]
    for _, v in netC.named_parameters(): param_group += [{'params': v, 'lr': args.lr}]
    optimizer = op_copy(optim.SGD(param_group))

    acc_init = 0.0
    max_iter = args.max_epoch * len(dls["source_tr"])  # 总迭代数
    interval_iter = max(max_iter // 10, 1)              # 每 10% 评估一次
    iter_num = 0

    netF.train(); netB.train(); netC.train()
    print("--------Training (Source)-------")
    start_time = time.time()
    iter_source = iter(dls["source_tr"])

    while iter_num < max_iter:
        try:
            batch = next(iter_source)
        except StopIteration:
            iter_source = iter(dls["source_tr"])
            batch = next(iter_source)
        inputs_source, labels_source = split_xy(batch)

        if inputs_source.size(0) <= 1:
            continue  # 避免极小 batch 影响 BN

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        dev = next(netF.parameters()).device
        inputs_source = inputs_source.to(dev, non_blocking=True)
        labels_source = labels_source.to(dev, non_blocking=True).long()
        outputs_source = netC(netB(netF(inputs_source)))

        # 监督交叉熵（带标签平滑）
        loss_ce = cross_entropy_loss(outputs_source, labels_source, epsilon=args.smooth)

        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        # 日志
        if iter_num % 100 == 0:
            now = time.time()
            print(f"iter_num: {iter_num}/{max_iter}, loss: {loss_ce.item():.4f}, dt/100it: {(now-start_time):.2f}s")
            start_time = now

        # 周期评估并记录 best
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval(); netB.eval(); netC.eval()
            acc_s_te, _ = cal_acc(dls['source_te'], netF, netB, netC, False)
            print(f"[Eval] Iter {iter_num}/{max_iter} | Acc = {acc_s_te:.2f}%")
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
            netF.train(); netB.train(); netC.train()

    # 保存最优权重（供适配阶段使用）
    os.makedirs(args.output_dir_src, exist_ok=True)
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
    print("save model in", args.output_dir_src)

    return netF, netB, netC


def test_target(args):
    """（可选）在源训练结束后，直接在目标域测试集（或同域测试集）做一次 sanity-check。"""
    dls = data_load(args)
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.gpu_id) != "-1" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.net.lower() == 'macnn':
        netF = MacnnBackbone(in_channels=args.in_channels, channels=args.channels, num_classes=args.class_num).to(device)
    elif args.net[:3] == 'res':
        netF = network.ResBase(res_name=args.net).to(device)
    elif args.net[:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).to(device)
    else:
        raise ValueError(f"未知的 --net: {args.net}")

    netB = network.FeatureNeck(feature_dim=netF.in_features, bottleneck_dim=args.bottleneck, type=args.classifier).to(device)
    netC = network.ClassifierHead(class_num=args.class_num, bottleneck_dim=args.bottleneck, type=args.layer).to(device)

    netF.load_state_dict(torch.load(osp.join(args.output_dir_src, "source_F.pt"), map_location=device))
    netB.load_state_dict(torch.load(osp.join(args.output_dir_src, "source_B.pt"), map_location=device))
    netC.load_state_dict(torch.load(osp.join(args.output_dir_src, "source_C.pt"), map_location=device))
    netF.eval(); netB.eval(); netC.eval()

    acc, _ = cal_acc(dls['test'], netF, netB, netC, False)
    print(f"[Test] Acc = {acc:.2f}%")


def main():
    """解析命令行并启动源域训练（MACNN + ORACLE-S）。"""
    parser = argparse.ArgumentParser(description='MACNN on ORACLE-S（中文注释版）')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--dset', type=str, default='rf', choices=['rf'])
    parser.add_argument('--rf_root', type=str, required=True, help='ORACLE-S 子集根目录（含 x_train_*.npy 等）')
    parser.add_argument('--rf_ft', type=str, default='62ft', help='特征版本（如 32ft/62ft）')

    parser.add_argument('--net', type=str, default='macnn', help='macnn / resnet50 / vgg16 等')
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--class_num', type=int, default=16)

    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.05)

    parser.add_argument('--output', type=str, default='exps_oracle')
    parser.add_argument('--train_frac', type=float, default=0.1, help='从 S1 train 中抽做训练的比例，例如 0.1 表示10%')
    parser.add_argument('--test_frac', type=float, default=0.2, help='从 S1 test 中抽做测试的比例，其余作为验证')
    parser.add_argument('--split_seed', type=int, default=30, help='划分的随机种子')
    # WiSig 数据集专用参数
    parser.add_argument('--dataset', type=str, default='rf', help='数据集类型：rf 或 wisig')
    parser.add_argument('--dataset_path', type=str, default=r'D:\pydata\Datasets', help='WiSig 数据集主目录')
    parser.add_argument('--wisig_sub', type=str, default='wisig', help='WiSig 子目录，如 wisig 或 wisig-eq')
    parser.add_argument('--src_date', type=str, default='1', help='WiSig 源域日期编号，例如 1')


    args = parser.parse_args()

    # 输出目录
    args.output_dir_src = osp.join(args.output, f'uda_rf_{args.rf_ft}')
    os.makedirs(args.output_dir_src, exist_ok=True)
    print("[LOG dir]:", args.output_dir_src)

    # 训练 & 可选测试
    train_source(args)
    test_target(args)


if __name__ == "__main__":
    main()
