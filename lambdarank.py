# Michael A. Alcorn (malcorn@redhat.com)
# A (slightly modified) implementation of LamdaRank as described in [1].
#   [1] https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
#   [2] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import rankdata

INPUT_DIM = 50
N_DOCS = 20
N_PAIRS = 200
N_REL = 5
N_IRR = N_DOCS - N_REL


def idcg(n_rel):
    # Assuming binary relevance.
    nums = np.ones(n_rel)
    denoms = np.log2(np.arange(n_rel) + 1 + 1)
    return (nums / denoms).sum()


# Model.
model = torch.nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Document scores.
doc_features = np.random.randn(N_DOCS, INPUT_DIM)
relevant = np.zeros(N_DOCS)
relevant[:N_REL] = 1

docs = torch.from_numpy(np.array(doc_features, dtype = "float32"))
docs = docs.to(device)
doc_scores = model(docs)
doc_scores_np = doc_scores.cpu().detach().numpy().flatten()

# Document ranks.
doc_ranks = rankdata(-doc_scores_np)

# Compute lambdas.
N = 1 / idcg(N_REL)
lambs = np.zeros(N_DOCS, dtype = "float32")
for i in range(N_REL):
    rel_score = doc_scores_np[i]
    diffs = rel_score - doc_scores_np[N_REL:]
    exped = np.exp(diffs)
    # See equation (6) in [2].
    lamb = -1 / (1 + exped) * N * np.abs(1 / np.log2(1 + doc_ranks[i]) - 1 / np.log2(1 + doc_ranks[N_REL:]))
    # See section 6.1 in [1], but lambdas have opposite signs from [2].
    lambs[i] -= lamb.sum()
    lambs[N_REL:] += lamb

# Accumulate lambda scaled gradients.
d_Cs = -lambs
grads = {}
for i in range(N_DOCS):
    model.zero_grad()
    doc_scores[i].backward(retain_graph = True)
    for param in model.parameters():
        # See section 4.1 in [2].
        if param not in grads:
            grads[param] = d_Cs[i] * param.grad
        else:
            grads[param] += d_Cs[i] * param.grad

# Update model weights.
lr = 0.00001
with torch.no_grad():
    for param in model.parameters():
        param -= lr * grads[param].to(device)
