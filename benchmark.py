"""
benchmark.py

A small, self-contained likelihood-based multiple-choice benchmark harness.

What it does, for every ensemble member:
  1. For each question, score every answer choice by its total log-probability
     under the model (teacher-forced), and predict the highest-scoring choice.
  2. Report accuracy.
And across the members it computes the uncertainty decomposition we derived:
     total = epistemic + aleatoric
  where each member's predictive distribution over the choices is one posterior
  sample (see the project's "Theory of uncertainty" notes).

IMPORTANT: run locally this uses the TINY model with random weights, so every
number is meaningless - this only proves the pipeline runs end to end. Real
numbers come from running the same code on the full Mixtral-8x7B on a GPU.

The dataset here is a tiny hard-coded toy set so the harness runs offline with
no extra downloads. To use a real benchmark later, replace `toy_dataset()` with
a loader that returns the same format: a list of
    {"question": str, "choices": [str, ...], "answer": int}.

Run it from the Code folder with:

    python benchmark.py
"""

import torch

from sanity_check import load_config, set_seed, load_model
from routing import (
    set_member,
    restore,
    topk_member,
    rank_shift_member,
    random_member,
)


def toy_dataset():
    """A handful of trivial MCQ items, just to exercise the pipeline offline."""
    return [
        {"question": "The capital of France is",
         "choices": ["Paris", "Berlin", "Madrid"], "answer": 0},
        {"question": "Two plus two equals",
         "choices": ["three", "four", "five"], "answer": 1},
        {"question": "The opposite of hot is",
         "choices": ["cold", "warm", "fast"], "answer": 0},
        {"question": "A dog is a kind of",
         "choices": ["plant", "animal", "mineral"], "answer": 1},
        {"question": "The sun rises in the",
         "choices": ["west", "north", "east"], "answer": 2},
    ]


def loglikelihood(model, tokenizer, context, continuation, device):
    """
    Total log-probability of `continuation` following `context`, teacher-forced.

    Standard MCQ scoring: tokenize context and context+continuation, run one
    forward pass, and sum the log-probs the model assigns to the actual
    continuation tokens.
    """
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids
    full_ids = tokenizer(context + continuation, return_tensors="pt").input_ids
    ctx_len = ctx_ids.shape[1]
    cont_len = full_ids.shape[1] - ctx_len
    if cont_len <= 0:
        return float("-inf")

    full_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(full_ids).logits  # (1, T, vocab)
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    # The token at position i is predicted from the logits at position i-1, so
    # the continuation tokens (positions ctx_len .. T-1) are predicted by the
    # logits at positions ctx_len-1 .. T-2.
    targets = full_ids[0, ctx_len:]                       # (cont_len,)
    pred_logprobs = logprobs[0, ctx_len - 1 : -1, :]       # (cont_len, vocab)
    token_lp = pred_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum().item()


def predict(model, tokenizer, item, device):
    """
    Score every choice and return (predicted_index, distribution_over_choices).

    The distribution is a softmax over the per-choice log-likelihoods - this is
    the member's predictive distribution we feed into the uncertainty math.
    """
    lps = [
        loglikelihood(model, tokenizer, item["question"], " " + choice, device)
        for choice in item["choices"]
    ]
    lps = torch.tensor(lps)
    pred = int(lps.argmax())
    dist = torch.softmax(lps, dim=0)
    return pred, dist


def evaluate_member(model, tokenizer, dataset, device):
    """Run one member over the dataset. Returns (accuracy, list-of-distributions)."""
    correct = 0
    dists = []
    for item in dataset:
        pred, dist = predict(model, tokenizer, item, device)
        correct += int(pred == item["answer"])
        dists.append(dist)
    return correct / len(dataset), dists


def entropy(p):
    """Shannon entropy in nats, safe against zeros."""
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum(-1)


def decompose_uncertainty(member_dists):
    """
    Given {member_name: [dist per example]}, compute the mean total / aleatoric /
    epistemic uncertainty over the dataset.

      total      = H(mean predictive)
      aleatoric  = mean over members of H(member predictive)
      epistemic  = total - aleatoric  = mean KL(member || mean)
    """
    names = list(member_dists)
    n_examples = len(member_dists[names[0]])
    totals, aleatorics = [], []
    for i in range(n_examples):
        ps = torch.stack([member_dists[name][i] for name in names])  # (M, C)
        pbar = ps.mean(dim=0)                                        # (C,)
        totals.append(entropy(pbar))
        aleatorics.append(entropy(ps).mean())
    total = torch.stack(totals).mean().item()
    aleatoric = torch.stack(aleatorics).mean().item()
    return total, aleatoric, total - aleatoric


def build_members(model):
    """The ensemble member set (see Overview method table). None = unmodified."""
    return {
        "top2_baseline": None,                       # ranks 1-2 (Mixtral default)
        "top3":          topk_member(3),             # ranks 1-3
        "rank_2_3":      rank_shift_member(1, 2),    # mild shift
        "rank_3_4":      rank_shift_member(2, 2),    # 2nd-best route (disjoint)
        "random_k2":     random_member(model, 2, 0),  # control (Q2)
    }


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    tokenizer, model, device = load_model(cfg)
    dataset = toy_dataset()
    members = build_members(model)

    print(f"=== Benchmark (model={cfg['model_id']}, {len(dataset)} items) ===")
    print("Tiny model has random weights -> numbers are meaningless; this only")
    print("checks that the harness runs end to end.\n")

    accuracies = {}
    member_dists = {}
    for name, member in members.items():
        if member is not None:
            set_member(model, member)
        acc, dists = evaluate_member(model, tokenizer, dataset, device)
        if member is not None:
            restore(model)
        accuracies[name] = acc
        member_dists[name] = dists
        print(f"  {name:14s} accuracy = {acc:.3f}")

    # Uncertainty decomposition over the principled members (the random control
    # is not a posterior sample, so it is excluded from the ensemble).
    ensemble = {n: member_dists[n] for n in ("top3", "rank_2_3", "rank_3_4")}
    total, aleatoric, epistemic = decompose_uncertainty(ensemble)

    print("\n=== Uncertainty decomposition (nats, mean over items) ===")
    print(f"  ensemble members: {list(ensemble)}")
    print(f"  total      = {total:.6f}")
    print(f"  aleatoric  = {aleatoric:.6f}")
    print(f"  epistemic  = {epistemic:.6e}   (= mean KL of each member to the mean)")
    if epistemic < 1e-9:
        print("  note: epistemic ~ 0 - expected here (random-weight model has no")
        print("        real disagreement structure); the math is what we are testing.")

    print("\nRESULT: harness ran end to end.")


if __name__ == "__main__":
    main()
