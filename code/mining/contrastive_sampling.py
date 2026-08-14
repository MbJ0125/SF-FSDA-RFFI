import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy


class CASampling(Strategy):
    def __init__(self, train_dataset, idxs_lb, net, handler, train_transform, test_transform, args, source_dataset):
        super(CASampling, self).__init__(
            train_dataset, idxs_lb, net, handler, train_transform, test_transform, args, source_dataset
        )
        self.output_dir = args.output_dir
        self.seed = args.seed
        self.shot_indices = getattr(args, "shot_indices", None)

        # 池级“上一轮 log 概率”缓存：形状 (n_pool, C)
        # 只初始化一次（避免每轮重建导致对比失效）
        if not hasattr(self, "prev_logp_pool"):
            self.prev_logp_pool = torch.full((self.n_pool, self.class_num), float('nan'))

    def query(self, n: int):

        # ========== Step 1: 限制 CAS 候选池到 label_universe (20-shot×16类) ==========
        label_universe_mask = np.zeros(len(self.idxs_lb), dtype=bool)
        label_universe_mask[self.label_universe] = True

        idxs_unl = np.where(self.idxs_lb == 0)[0]

        print(f"[CAS] Candidates remaining = {len(idxs_unl)}")
        assert len(idxs_unl) >= n, \
            "可标注池候选数不足，请减少轮数或增加 shots!"

        if len(idxs_unl) == 0 or n <= 0:
            return np.array([], dtype=int)

        if n > len(idxs_unl):
            n = len(idxs_unl)

        # ========== Step 2: 求 logits + prob ==========
        self.net.eval()
        unl_list = [self.train_dataset[i] for i in idxs_unl.tolist()]

        # predict_prob 返回 probability，但 Momentum 需要 logits
        probs_r, logits_r = self.predict_prob(unl_list, return_logits=True)
        logp_r = (probs_r + 1e-12).log()

        # ========== Step 3: Momentum Teacher ==========
        # 如果 memory 还没初始化 → 初始化
        if not self.mem_init:
            self.mem_logits[idxs_unl] = logits_r.detach().cpu()
            self.mem_init = True

        # 从 memory 中取 teacher logits
        logits_teacher = self.mem_logits[idxs_unl]
        prev_logp = torch.log_softmax(logits_teacher, dim=1)

        # 对比式后验
        alpha = float(getattr(self.args, "cd_ratio", 0.5))
        p_tilde = logp_r + alpha * (logp_r - prev_logp)
        probs_tilde = torch.softmax(p_tilde, dim=1)

        # ========== Step 4: UCM ==========
        top2_vals, top2_idx = torch.topk(p_tilde, k=2, dim=1)
        ucm = top2_vals[:, 0] - top2_vals[:, 1]
        ya = top2_idx[:, 0]

        # ========== Step 5: UCT ==========
        lam = float(getattr(self.args, 'uct_lambda', 0.0))
        ucm_off = bool(getattr(self.args, 'ucm_off', False))

        N = ucm.numel()
        kappa = int(getattr(self.args, 'uct_kappa', -1))
        if kappa < 0:
            kappa = max(50, N // 10)
        kappa = min(kappa, N)

        if ucm_off:
            entropy = -(probs_tilde * probs_tilde.clamp_min(1e-12).log()).sum(dim=1)
            topk_idx = entropy.topk(kappa, largest=True).indices
            is_topk = torch.zeros(N, dtype=torch.bool)
            is_topk[topk_idx] = True
        else:
            #order = torch.argsort(ucm, descending=False)
            #rank = torch.empty_like(order)
            #rank[order] = torch.arange(N)
            #is_topk = rank < kappa
            order = torch.argsort(ucm, descending=True)  # margin 越大，排得越前
            is_topk = torch.zeros(N, dtype=torch.bool)
            is_topk[order[:kappa]] = True

        C = probs_r.shape[1]
        uct_counts = torch.zeros(C)
        for c in range(C):
            uct_counts[c] = ((ya == c) & is_topk).sum()
        denom = float(uct_counts.max()) if uct_counts.max() > 0 else 1.0
        uct = uct_counts / denom
        uct_per = uct[ya]

        # ========== Step 6: final score ==========
        if ucm_off and lam <= 0:
            score = -entropy
        elif ucm_off:
            score = -uct_per * lam
        elif lam <= 0:
            score = ucm
        else:
            score = ucm + lam * uct_per

        # ========== Step 7: Pick n ==========
        pick_local = torch.topk(score, k=n, largest=False).indices
        chosen = idxs_unl[pick_local.cpu().numpy()]

        # ========== Step 8: Momentum 更新 ==========
        m = float(getattr(self.args, "mem_momentum", 0.9))
        self.mem_logits[idxs_unl] = (
                m * self.mem_logits[idxs_unl]
                + (1 - m) * logits_r.detach().cpu()
        )

        print(f"[CAS] Selected idx = {chosen.tolist()}")

        return chosen

    def label_statistic(self, labels, metrics, num_label, topk=500, descending=True):
        """可选：早期版本的类均衡加权（未在主流程中调用时可保留以备后用）"""
        idx_topk = metrics.sort(descending=descending)[1][:topk]
        statistic = torch.zeros(num_label)
        for idx in idx_topk:
            l = int(labels[idx])
            statistic[l] += 1.0
        label_weight = statistic / (statistic.max() if statistic.max() > 0 else 1.0)
        instance_weight = label_weight[labels]
        return instance_weight
