import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


from .utils import lr_scheduler, cal_acc, op_copy, cross_entropy_loss, info_maximize_loss, print_log
from .utils import euclidean_metric, cosine_metric, l1norm, l2norm, Entropy


class MixedDataset(Dataset):
    def __init__(self, input_a, input_b, target_a, target_b, args):
        self.input_a = input_a
        self.input_b = input_b
        self.target_a = target_a
        self.target_b = target_b
        self.args = args
        self.class_num = args.class_num

        assert input_a.size(0) == input_b.size(0) and target_a.size(0) == target_b.size(0)

    def __getitem__(self, index):
        #prepare mixed data
        l = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
        l = max(l, 1 - l)
        x1 = self.input_a[index]
        x2 = self.input_b[index]

        y1 = self.target_a[index].long()
        y2 = self.target_b[index].long()
        y1 = torch.zeros(self.class_num).scatter_(0, y1.cpu(), 1)
        y2 = torch.zeros(self.class_num).scatter_(0, y2.cpu(), 1)
        
        x = l * x1 + (1 - l) * x2
        y = l * y1 + (1 - l) * y2
        return x, y

    def __len__(self):
        return self.input_a.size(0)

class Strategy:
    def __init__(self, train_dataset, idxs_lb, net, handler, train_transform, test_transform, args, source_dataset):
        self.train_dataset = train_dataset
        self.idxs_lb = idxs_lb
        self.ori_net_dict = net.state_dict()
        self.net = net
        self.handler = handler #List -> Dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.args = args
        self.source_dataset = source_dataset

        self.n_pool = len(train_dataset)
        self.batch_size = args.batch_size

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.class_num = args.class_num
        self.sfada_ubl = -1
        if hasattr(args, "sfada_ubl"):
            self.sfada_ubl = args.sfada_ubl

        # 固定 unlabeled 池（300）
        self.fixed_unlabeled = None
        self.fixed_unlabeled_size = 300

        self.lbl_memory_embs = torch.zeros([self.n_pool, self.net.get_embedding_dim()])
        self.lbl_memory_probs = torch.zeros([self.n_pool, self.class_num])
        
        self.n_labeled = round(self.n_pool * args.ratio_per_round * args.num_round)

        self.ema_F = getattr(args, 'ema_F', None)
        self.ema_B = getattr(args, 'ema_B', None)
        self.ema_C = getattr(args, 'ema_C', None)

        # mixup_seed_only / mixup_seed_only_rounds: 只对初始标注做mixup
        self._seed_indices = None
        if getattr(args, 'mixup_seed_only', False) or int(getattr(args, 'mixup_seed_only_rounds', 0)) > 0:
            self._seed_indices = getattr(args, '_initial_labeled_idx', None)
        # 现有：
        # ==============  Memory Bank (momentum teacher) ==============
        # 保存 logits，而不是 log p
        self.mem_logits = torch.zeros((self.n_pool, self.class_num), dtype=torch.float32)

        # 标记是否完成初始化（首轮不做 momentum）
        self.mem_init = False

        # === New SF-FSDA components ===
        # Store confirmed pseudo labels: {global_index: pseudo_label}
        self.pseudo_labels = {}

        # Persistent class-center anchors, shape: [class_num, embedding_dim]
        self.class_centers = None


    def print_log(self, log_str):
        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        print(log_str)

    def save(self, round, args):
        self.net.save(round, args)

    def query(self, n):
        pass

    def update(self, q_idxs):
        self.idxs_lb[q_idxs] = True

    def add_pseudo_labels(self, indices, pseudo_labels):
        """
        Store confirmed pseudo labels.
        indices: global indices in the target training pool.
        pseudo_labels: pseudo labels predicted by model-prototype consistency.
        """
        for idx, y in zip(indices, pseudo_labels):
            self.pseudo_labels[int(idx)] = int(y)

    def build_labeled_dataset(self, indices):
        """
        Build labeled dataset for training.
        For initial few-shot samples, use their ground-truth labels.
        For confirmed pseudo-labeled samples, use stored pseudo labels.
        """
        data = []
        for i in indices:
            item = self.train_dataset[int(i)]
            x = item[0]
            y = item[1]

            if int(i) in self.pseudo_labels:
                y = self.pseudo_labels[int(i)]

            data.append((x, int(y), int(i)))
        return data

    @torch.no_grad()
    def init_class_centers(self):
        """
        Initialize or recompute class-center anchors from current labeled target samples.
        Labeled samples include initial few-shot samples and confirmed pseudo-labeled samples.
        """
        idxs_labeled = np.where(self.idxs_lb == 1)[0].tolist()
        labeled_data = self.build_labeled_dataset(idxs_labeled)

        centers = torch.zeros(self.class_num, self.net.get_embedding_dim())
        counts = torch.zeros(self.class_num)

        loader = DataLoader(
            self.handler(labeled_data, transform=self.test_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.worker,
            drop_last=False
        )

        self.init_net(training=False)
        self.net.eval()

        for x, y, _ in loader:
            x = x.to(self.device)
            y = y.long()

            _, emb = self.net(x)
            emb = emb.detach().cpu()

            for c in range(self.class_num):
                mask = (y == c)
                if mask.any():
                    centers[c] += emb[mask.cpu()].sum(dim=0)
                    counts[c] += mask.sum().cpu()

        valid = counts > 0
        centers[valid] = centers[valid] / counts[valid].unsqueeze(1)

        self.class_centers = centers
        return counts

    @torch.no_grad()
    def update_class_centers_ema(self):
        """
        EMA update for persistent class-center anchors.
        """
        old_centers = None if self.class_centers is None else self.class_centers.clone()

        counts = self.init_class_centers()

        if old_centers is not None:
            gamma = float(getattr(self.args, "gamma", 0.1))
            valid = counts > 0
            self.class_centers[valid] = (
                    gamma * self.class_centers[valid]
                    + (1.0 - gamma) * old_centers[valid]
            )

        return counts

    @torch.no_grad()
    def confirm_by_prototype(self, candidate_idx):
        """
        Confirm pseudo labels by checking consistency between:
        1) model prediction
        2) nearest class-center prototype prediction
        """
        if self.class_centers is None:
            self.init_class_centers()

        candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
        if len(candidate_idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        candidate_data = [self.train_dataset[int(i)] for i in candidate_idx]

        loader = DataLoader(
            self.handler(candidate_data, transform=self.test_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.worker,
            drop_last=False
        )

        self.init_net(training=False)
        self.net.eval()

        centers = F.normalize(self.class_centers.to(self.device), dim=1)

        confirmed_idx = []
        pseudo_labels = []
        offset = 0

        for x, _, _ in loader:
            x = x.to(self.device)

            logits, emb = self.net(x)
            probs = F.softmax(logits, dim=1)
            conf, y_model = probs.max(dim=1)
            conf_thresh = float(getattr(self.args, "pseudo_conf_threshold", 0.0))

            emb = F.normalize(emb, dim=1)
            sim = emb @ centers.t()
            y_proto = sim.argmax(dim=1)

            agree = (y_model == y_proto).detach().cpu().numpy()
            conf_ok = (conf > conf_thresh).detach().cpu().numpy()
            batch_indices = candidate_idx[offset: offset + x.size(0)]
            offset += x.size(0)

            for j, ok in enumerate(agree):
                if ok and conf_ok[j]:
                    confirmed_idx.append(int(batch_indices[j]))
                    pseudo_labels.append(int(y_model[j].detach().cpu()))

        return np.array(confirmed_idx, dtype=np.int64), np.array(pseudo_labels, dtype=np.int64)

    def center_pgra_loss(self, embs_u):
        """
        PGRA loss based on entropy minimization of center-affinity distribution.
        """
        if self.class_centers is None:
            return torch.tensor(0.0, device=self.device)

        centers = F.normalize(self.class_centers.to(self.device), dim=1)
        embs_u = F.normalize(embs_u, dim=1)

        tau = float(getattr(self.args, "tau", 0.5))
        logits_center = embs_u @ centers.t() / tau
        q = F.softmax(logits_center, dim=1)

        loss_pgra = -(q * q.clamp_min(1e-12).log()).sum(dim=1).mean()
        return loss_pgra

    def mixup_criterion(self, inputs, labels):
        alpha = self.args.mixup_alpha

        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)

        labels = labels.to(inputs.device).long()

        index = torch.randperm(inputs.size(0), device=inputs.device)

        mixed_x = lam * inputs + (1 - lam) * inputs[index]

        y_a = F.one_hot(labels, num_classes=self.class_num).float()
        y_b = F.one_hot(labels[index], num_classes=self.class_num).float()

        mixed_y = lam * y_a + (1 - lam) * y_b

        outputs, _ = self.net(mixed_x)

        log_prob = F.log_softmax(outputs, dim=1)

        loss_mix = -(mixed_y.to(inputs.device) * log_prob).sum(dim=1).mean()

        return loss_mix


    def init_net(self, training=True):
        self.net = self.net.to(self.device)

    def get_ubl_dataset(self):
        """
        不再使用固定300未标注集！
        动态构造：使用所有“不可标注池”(非320-shot)的数据做DA
        """

        # DA unlabeled = 未标注 且 不在label_universe里
        idxs_da_unlabeled = np.where(
            (self.idxs_lb == 0) & (~np.isin(np.arange(len(self.train_dataset)), self.label_universe))
        )[0]

        ubl_dataset = [self.train_dataset[i] for i in idxs_da_unlabeled]

        print(f"[DA-UNLABELED] Using {len(ubl_dataset)} samples for DA (dynamic update)")

        return ubl_dataset

    def train_with_mixup(self, data, optimizer, interval=100):

        input_a = torch.cat(data[0], dim=0)
        target_a = torch.cat(data[2], dim=0)

        rand_idxs = list(range(input_a.size(0)))
        random.shuffle(rand_idxs)
        
        # mix labeled samples
        input_b = input_a[rand_idxs]
        target_b = target_a[rand_idxs]

        mixed_dataset = MixedDataset(input_a, input_b, target_a, target_b, self.args)
        mixed_dataloader = DataLoader(mixed_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)

        for i, (input, target) in enumerate(mixed_dataloader):
            if input.size(0) == 1:
                continue
            mixed_inputs = input.to(self.device)
            mixed_targets = target.to(self.device)

            mixed_outputs, _ = self.net(mixed_inputs)
            mixed_outputs = nn.LogSoftmax(dim=1)(mixed_outputs)
            loss_mixed = (- mixed_targets * mixed_outputs).sum(dim=1).mean() * self.args.lambda_mixup

            optimizer.zero_grad()
            loss_mixed.backward()
            optimizer.step()

           # if i % interval == 0:
               # log_str = "[MIXUP], Loss: %.4f" % (loss_mixed.item())
               # print_log(log_str, self.args)

    def update_ema_teacher(netF, netB, netC, ema_F, ema_B, ema_C, momentum=0.999):
        with torch.no_grad():
            for param, ema_param in zip(netF.parameters(), ema_F.parameters()):
                ema_param.data = momentum * ema_param.data + (1 - momentum) * param.data
            for param, ema_param in zip(netB.parameters(), ema_B.parameters()):
                ema_param.data = momentum * ema_param.data + (1 - momentum) * param.data
            for param, ema_param in zip(netC.parameters(), ema_C.parameters()):
                ema_param.data = momentum * ema_param.data + (1 - momentum) * param.data

    def train_one_round_with_unlabel(self, round):
        n_epoch = self.args.max_epoch
        self.init_net(training=True)
        param_group = self.net.named_parameters(self.args)

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb].tolist()

        # mixup_seed_only：将标注集分为初始种子和伪标签两部分
        lbl_dataset = self.build_labeled_dataset(idxs_train)
        if self._seed_indices is not None:
            seed_set = [item for item in lbl_dataset if item[2] in self._seed_indices]
            pseudo_set = [item for item in lbl_dataset if item[2] not in self._seed_indices]
            train_lbl_dataloader = DataLoader(self.handler(seed_set + pseudo_set, transform=self.train_transform),
                                batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
            self._seed_count = len(seed_set)
        else:
            train_lbl_dataloader = DataLoader(self.handler(lbl_dataset, transform=self.train_transform),
                                batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
        
        if self.source_dataset is not None:
            train_src_dataloader = DataLoader(self.handler(self.source_dataset, transform=self.train_transform),
                                batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)



        # ================= 正确：每轮动态未标注池 Tu =================
        idxs_unl_global = np.where(self.idxs_lb == 0)[0]
        assert len(idxs_unl_global) > 0

        ul_ratio = float(getattr(self.args, "ul_ratio", 100)) / 100.0
        da_num = int(len(idxs_unl_global) * ul_ratio)
        idxs_ubl_da = np.random.choice(idxs_unl_global, da_num, replace=False)

        print(f"[DA] Global TU = {len(idxs_unl_global)}")
        print(f"[DA] Used {len(idxs_ubl_da)} for DA training (ratio={ul_ratio:.2f})")

        ubl_dataset = [self.train_dataset[i] for i in idxs_ubl_da.tolist()]

        ubl_num = len(ubl_dataset)

        print(f"[DA-UNLABELED] Using {ubl_num} samples for DA (ratio={ul_ratio:.2f})")

        print(f"[DA] DA-only unlabeled = {ubl_num} samples (excluding label-universe)")

        # 根据 1000（或更少）数量重新计算 batch 数
        ubl_batch_num = int(ubl_num / self.batch_size) + int(ubl_num % self.batch_size != 0)

        # ===== 计算 max_iter =====
        max_iter = n_epoch * ubl_batch_num

        log_str = "------Train at Round %d with %d Annotated Data and %d Unlabeled Data-----" % (round, len(lbl_dataset), ubl_num)
        print_log(log_str, self.args)
        iter_num = 0
        interval = max_iter // 10
        for epoch in range(n_epoch):
            #ubl_dataset = self.get_ubl_dataset()
            train_ubl_dataloader = DataLoader(self.handler(ubl_dataset, transform=self.train_transform),
                            batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
            #assert len(train_ubl_dataloader) == ubl_batch_num, "%d!=%d" % (len(train_ubl_dataloader), ubl_batch_num)
            #assert len(train_ubl_dataloader) >= len(train_lbl_dataloader)

            # update features in the memory
            momentum = self.args.mem_momentum
            _, lbl_embs = self.get_output(lbl_dataset)
            updated_lbl_embs = (1 - momentum) * self.lbl_memory_embs[idxs_train] + momentum * lbl_embs
            lbl_gnds = torch.tensor([item[1] for item in lbl_dataset])
            updated_lbl_probs = self.lbl_memory_probs[idxs_train]
            # 确保索引是 int64，并与 updated_lbl_probs 在同一设备
            idx = lbl_gnds.to(torch.long).unsqueeze(1)
            idx = idx.to(updated_lbl_probs.device)
            updated_lbl_probs.scatter_(1, idx, 1)

            self.lbl_memory_embs[idxs_train] = updated_lbl_embs
            self.lbl_memory_probs[idxs_train] = updated_lbl_probs

            mixed_dataset = [[], [], [], []]
            
            for _ in range(len(train_ubl_dataloader)):
                self.net.train()
                try:
                    inputs_ubl, label_ubl, _ = next(iter_ubl)
                except:
                    iter_ubl = iter(train_ubl_dataloader)
                    inputs_ubl, label_ubl, _ = next(iter_ubl)
                
                try:
                    inputs_lbl, label_lbl, idx = next(iter_lbl)
                except:
                    iter_lbl = iter(train_lbl_dataloader)
                    inputs_lbl, label_lbl, idx = next(iter_lbl)

                if inputs_ubl.size(0) == 1 or inputs_lbl.size(0) == 1:
                    continue
                iter_num += 1

                lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
                inputs_ubl = inputs_ubl.to(self.device)
                inputs_lbl = inputs_lbl.to(self.device)

                # unlabeled branch for PGRA
                outputs_ubl, embs_ubl = self.net(inputs_ubl)

                # Mixup supervision on labeled samples.
                # 支持延迟mixup: mixup_start_round 轮之前只做标准CE
                mixup_start_round = int(getattr(self.args, "mixup_start_round", 1))

                if round >= mixup_start_round:
                    # mixup_seed_only_rounds: 在前N轮中，mixup只对初始种子样本做
                    seed_only_rounds = int(getattr(self.args, "mixup_seed_only_rounds", 0))
                    if seed_only_rounds > 0 and round <= seed_only_rounds and self._seed_indices is not None:
                        batch_is_seed = len(idx) > 0 and all(int(i) in self._seed_indices for i in idx)
                        do_mixup = batch_is_seed and np.random.rand() < float(getattr(self.args, "mixup_prob", 0.5))
                    else:
                        do_mixup = np.random.rand() < float(getattr(self.args, "mixup_prob", 0.5))

                    if do_mixup:
                        loss_mix = self.mixup_criterion(inputs_lbl, label_lbl)
                        mix_mode = "mix"
                    else:
                        outputs_lbl, _ = self.net(inputs_lbl)
                        loss_mix = cross_entropy_loss(
                            outputs_lbl, label_lbl,
                            reduction=True, use_gpu=self.use_cuda)
                        mix_mode = "ce"
                else:
                    outputs_lbl, _ = self.net(inputs_lbl)
                    loss_mix = cross_entropy_loss(
                        outputs_lbl, label_lbl,
                        reduction=True, use_gpu=self.use_cuda)
                    mix_mode = "ce(no_mix)"

                # PGRA loss
                loss_pgra = self.center_pgra_loss(embs_ubl)

                beta_pgra = float(getattr(self.args, "beta_pgra", 1.0))
                loss = loss_mix + beta_pgra * loss_pgra


                mixed_dataset[0].append(inputs_lbl.float().cpu())
                mixed_dataset[1].append(inputs_ubl.float().cpu())
                mixed_dataset[2].append(label_lbl.data.float().cpu())
                mixed_dataset[3].append(label_ubl.data.float().cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            #if self.args.lambda_mixup > 0 and len(lbl_dataset) > 1:
                #self.train_with_mixup(mixed_dataset, optimizer, interval) #mixup works. 87.03->87.27
            print_log(
                f"[Round {round}, Epoch {epoch}] "
                f"Lmix({mix_mode})={loss_mix.item():.4f}  "
                f"Lpgra={(beta_pgra * loss_pgra).item():.4f}  "
                f"Total={loss.item():.4f}",
                self.args
            )
    def train_one_round(self, round):
        if self.args.beta_im > 0:
            return self.train_one_round_with_unlabel(round)
        else:
            assert self.args.beta_im == 0 and self.args.beta_vpa == 0 and self.args.lambda_mixup == 0, "im, att and mixup are only for unlabeled data"

            
        n_epoch = self.args.max_epoch
        self.init_net(training=True)
        param_group = self.net.named_parameters(self.args)

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb].tolist()
        round_dataset = self.build_labeled_dataset(idxs_train)
        train_dataloader = DataLoader(self.handler(round_dataset, transform=self.train_transform),
                            batch_size=self.batch_size, shuffle=True, num_workers=self.args.worker, drop_last=False)
        log_str = "------Train at Round %d with %d Annotated Data-----" % (round, len(round_dataset))
        print_log(log_str, self.args)

        max_iter = n_epoch * len(train_dataloader)
        iter_num = 0
        interval = max_iter // 10
        for epoch in range(n_epoch):
            self.net.train()
            for _, (inputs, label, _) in enumerate(train_dataloader):
                if inputs.size(0) == 1:
                    continue
                iter_num += 1
                lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
                inputs = inputs.to(self.device)
                outputs, _ = self.net(inputs)
                loss = cross_entropy_loss(outputs, label, reduction=True, use_gpu=self.use_cuda)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter_num % interval == 0:
                    log_str = "[Round: %d, Epoch: %d, Iter: %d/%d], Loss: %.4f" % (round, epoch, iter_num, max_iter, loss.item())
                    print_log(log_str, self.args)

    def predict(self, test_dataset, round=-1):
        # test_dataset: list
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        log_str = "------Evaluate at Round %d on %d Data-----" % (round, len(test_dataset))
        print_log(log_str, self.args)
        self.init_net(training=False)
        self.net.eval()
        preds = torch.zeros(len(test_dataset))
        gnds = torch.zeros(len(test_dataset))
        with torch.no_grad():
            for x, y, idxs in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.data.cpu().float()
                gnds[idxs] = y.data.cpu().float()
        return cal_acc(preds, gnds)

    def predict_prob(self, test_dataset, return_logits=False):
        """
        批处理推理版本 —— 支持返回 probability 或 (probability + logits)
        return_logits = True 时，返回 (probs, logits) 两个矩阵
        """

        # === 1) 构建批量 dataloader ===
        test_loader = DataLoader(
            self.handler(test_dataset, transform=self.test_transform),
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=self.args.worker,
            drop_last=False
        )

        self.init_net(training=False)
        self.net.eval()

        all_probs = []
        all_logits = []

        # === 2) 批量前向传播 ===
        with torch.no_grad():
            for x, y, idxs in test_loader:
                x = x.to(self.device)

                # 你模型的 net(x) 返回 (logits, features)
                logits, _ = self.net(x)

                probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu())
                if return_logits:
                    all_logits.append(logits.cpu())

        # === 3) 拼接 ===
        all_probs = torch.cat(all_probs, dim=0)
        if return_logits:
            all_logits = torch.cat(all_logits, dim=0)
            return all_probs, all_logits

        return all_probs

    def predict_prob_dropout_split(self, test_dataset, n_drop):
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        self.init_net(training=False)
        self.net.train()
        probs = torch.zeros([n_drop, len(test_dataset), self.class_num])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in test_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, _ = self.net(x)
                    probs[i][idxs] += F.softmax(out, dim=1).data.cpu()
        
        return probs

    def get_embedding(self, test_dataset):
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        self.init_net(training=False)
        self.net.eval()
        embedding = torch.zeros([len(test_dataset), self.net.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                _, e1 = self.net(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding
    
    def get_output(self, test_dataset):
        test_dataloader = DataLoader(self.handler(test_dataset, transform=self.test_transform),
                            batch_size=self.batch_size*3, shuffle=False, num_workers=self.args.worker, drop_last=False)
        self.init_net(training=False)
        self.net.eval()
        probs = torch.zeros([len(test_dataset), self.class_num])
        embedding = torch.zeros([len(test_dataset), self.net.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                out, emb = self.net(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.data.cpu()
                embedding[idxs] = emb.data.cpu()
        
        return probs, embedding

