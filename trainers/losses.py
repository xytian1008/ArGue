import torch.nn.functional as F
import torch
from torch import nn

def transpose(x):
    return x.t() if x.dim() == 2 else x.permute(0, 2, 1)

def contrastive_loss(visual_features, class_prototypes, labels=None, t=0.07):
    logits = t.exp() * visual_features @ transpose(class_prototypes)

    if labels is not None:
        return F.cross_entropy(logits, labels), logits
    else:
        return None, logits

def cross_entropy_loss(visual_features, class_prototypes, n_cls, n_attr_per, labels=None, t=0.07):
    logits = t.exp() * visual_features @ transpose(class_prototypes)
    logits = logits.reshape(-1, n_cls, n_attr_per).mean(dim = -1)

    if labels is not None:
        return F.cross_entropy(logits, labels), logits
    else:
        return None, logits

def bias_loss(visual_features, class_prototypes, n_cls, n_bias_per, labels, t = 0.07, scale = 45):
    logits = t.exp() * visual_features @ transpose(class_prototypes)
    logits = logits.reshape(-1, n_cls, n_bias_per).mean(dim = -1)
    lsm = nn.LogSoftmax(dim = 1)
    log_probs = lsm(logits)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return -entropy, logits