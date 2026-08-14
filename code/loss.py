# loss.py
import torch
import torch.nn.functional as F

# -----------------------------
# Visual Persistence-guided Adaptation Loss
# -----------------------------
def visual_persistence_loss(f_u, f_l, gamma=0.9):
    """
    f_u: [N_u, D] 未标注样本特征
    f_l: [N_l, D] 已标注样本特征（anchor）
    gamma: 动量更新系数
    """
    # 动量更新 (Eq.7)
    with torch.no_grad():
        f_l_ema = gamma * f_l + (1 - gamma) * f_l.detach()

    # 余弦相似度计算 (Eq.6)
    sim = F.cosine_similarity(f_u.unsqueeze(1), f_l_ema.unsqueeze(0), dim=-1)  # [N_u, N_l]
    sim = F.softmax(sim, dim=1)  # 归一化为概率分布
    # 熵最小化 - 越集中越好
    vpa_loss = -(sim * sim.log()).sum(dim=1).mean()
    return vpa_loss

# -----------------------------
# Cross-Entropy Loss (Eq.5)
# -----------------------------
def cross_entropy_loss(logits, labels, epsilon: float = 0.0):
    """CE with optional label smoothing (epsilon)."""
    epsilon = float(epsilon)
    return F.cross_entropy(logits, labels, label_smoothing=epsilon)

# -----------------------------
# Entropy Minimization Loss (Eq.8)
# -----------------------------
def entropy_min_loss(logits):
    p = F.softmax(logits, dim=1)
    ent = -(p * p.log()).sum(dim=1)
    return ent.mean()

# -----------------------------
# Total Loss (Eq.9)
# -----------------------------
def lftl_total_loss(logits_l, labels_l, logits_u, f_u, f_l, beta1=0.5, beta2=0.1):
    """
    logits_l: labeled logits
    labels_l: labeled targets
    logits_u: unlabeled logits
    f_u: unlabeled features
    f_l: labeled features
    """
    Lce = cross_entropy_loss(logits_l, labels_l)
    Lvpa = visual_persistence_loss(f_u, f_l)
    Lent = entropy_min_loss(logits_u)
    return Lce + beta1 * Lvpa + beta2 * Lent
