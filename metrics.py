import torch
import torch.nn as nn
from torch.nn import functional as F


def precision_recall_f1(pred, labels, idx2target_vocab):
    
    pred_idxs = torch.argmax(pred, dim = 1)
    label_idxs = labels
    pred_names = [idx2target_vocab[i.item()] for i in pred_idxs]
    original_names = [idx2target_vocab[i.item()] for i in label_idxs]

    tp, fp, fn = 0, 0, 0

    for pr_name, orig_name in zip(pred_names, original_names):

        pr_subtokens = pr_name.split('|')
        orig_subtokens = orig_name.split('|')

        for subtoken in pr_subtokens:
            if subtoken in orig_subtokens:
                tp += 1
            else:
                fp += 1
        for subtoken in orig_subtokens:
            if not subtoken in pr_subtokens:
                fn += 1

    epsilon = 1e-7

    '''precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)'''

    return tp, fp, fn
