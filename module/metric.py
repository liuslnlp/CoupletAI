from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sf = SmoothingFunction()


def calc_bleu(cand: List[int or str], ref: List[int or str]):
    return sentence_bleu([ref], cand, smoothing_function=sf.method1)


def calc_rouge_l(cand: List[int or str], ref: List[int or str], beta: float = 1.2):
    len_cand = len(cand)
    len_ref = len(ref)
    lengths = [[0 for j in range(len_ref + 1)] for i in range(len_cand + 1)]
    for i in range(len_cand):
        for j in range(len_ref):
            if cand[i] == ref[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            elif lengths[i + 1][j] > lengths[i][j + 1]:
                lengths[i + 1][j + 1] = lengths[i + 1][j]
            else:
                lengths[i + 1][j + 1] = lengths[i][j + 1]
    lcs = lengths[-1][-1]
    eps = 1e-10
    r = lcs * 1.0 / (eps + len_ref)
    p = lcs * 1.0 / (eps + len_cand)
    f = ((1 + beta**2) * r * p) / (eps + r + beta ** 2 * p)
    return f
