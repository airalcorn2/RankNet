# Michael A. Alcorn (malcorn@redhat.com)
# A (slightly modified) implementation of LamdaRank as described in [1].
#   [1] https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
#   [2] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

import numpy as np
import torch
import torch.nn as nn


def idcg(n_rel):
    # Assuming binary relevance.
    nums = np.ones(n_rel)
    denoms = np.log2(np.arange(n_rel) + 1 + 1)
    return (nums / denoms).sum()


# Data.
input_dim = 50
n_docs = 20
n_rel = 5
n_irr = n_docs - n_rel

doc_features = np.random.randn(n_docs, input_dim)

# Model.
model = torch.nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Document scores.
docs = torch.from_numpy(np.array(doc_features, dtype="float32"))
docs = docs.to(device)
doc_scores = model(docs)

# Document ranks.
(sorted_scores, sorted_idxs) = doc_scores.sort(dim=0, descending=True)
doc_ranks = torch.zeros(n_docs).to(device)
doc_ranks[sorted_idxs] = 1 + torch.arange(n_docs).view((n_docs, 1)).to(device).float()
doc_ranks = doc_ranks.view((n_docs, 1))

# Compute lambdas.
# See equation (6) in [2] and equation (9) in [1].
score_diffs = doc_scores[:n_rel] - doc_scores[n_rel:].view(n_irr)
exped = score_diffs.exp()
N = 1 / idcg(n_rel)
dcg_diffs = 1 / (1 + doc_ranks[:n_rel]).log2() - (1 / (1 + doc_ranks[n_rel:]).log2()).view(n_irr)
lamb_updates = 1 / (1 + exped) * N * dcg_diffs.abs()
lambs = torch.zeros((n_docs, 1)).to(device)
lambs[:n_rel] += lamb_updates.sum(dim=1, keepdim=True)
lambs[n_rel:] -= lamb_updates.sum(dim=0, keepdim=True).t()

# Accumulate lambda scaled gradients.
model.zero_grad()
doc_scores.backward(lambs)

# Update model weights.
lr = 0.00001
with torch.no_grad():
    for param in model.parameters():
        param += lr * param.grad
