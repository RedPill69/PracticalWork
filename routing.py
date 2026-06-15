"""
routing.py

This is the heart of Stage 1: turning ONE Mixtral model into an "ensemble" by
changing which experts the router picks - without copying weights or retraining.

Background (how Mixtral routes):
  Each decoder layer has a Mixture-of-Experts block. For every token, a small
  linear layer (the "gate"/router) scores all experts, keeps the top 2, runs a
  softmax over those 2 scores, and the layer output is the weighted sum of the
  2 chosen expert FFNs.

What we do here:
  A "member" of our ensemble = the same model run with a fixed RULE ("policy")
  for which experts each layer uses. We swap the router for a `MemberRouter`
  that applies the policy and recomputes the softmax weights over ONLY the
  chosen experts so they still sum to 1.

  That last point is the "Option A" renormalization from the project notes:
  if we keep a single expert its weight becomes 1.0 (NOT the old ~0.5 top-2
  weight), so the FFN contribution is not silently halved across all layers.

The policies (one per member type in the plan):
  - "topk"       : per token, take the gate's k highest-ranked experts.
                   k=2 reproduces the baseline; k=3 is the "Top-3" member.
  - "rank_shift" : per token, take a window of the ranking that SKIPS the top.
                   e.g. start=1, k=2 -> ranks 2-3; start=2, k=1 -> rank 3 alone.
                   This is the "rank-shifted" diversity member.
  - "fixed"      : the SAME explicit expert set for every token in the layer.
                   Used for the random-expert control (see `random_member`) and
                   for single-expert debugging.

Top-2 (the baseline) needs no override at all - just run the unmodified model.

In this transformers version the routing decision lives in the router submodule
`MixtralTopKRouter`, NOT in `MixtralSparseMoeBlock.forward` - so the router is
exactly where we intervene.
"""

import random

import torch
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import MixtralTopKRouter

# The MoE block class name we look for when walking the model.
MOE_BLOCK_NAME = "MixtralSparseMoeBlock"


class MemberRouter(MixtralTopKRouter):
    """
    A drop-in replacement for Mixtral's router that applies one of our routing
    POLICIES instead of taking the natural top-2.

    Built from the layer's existing gate so it reuses the SAME trained weight
    (we never create new random router weights).
    """

    def __init__(self, gate, policy):
        # Deliberately skip MixtralTopKRouter.__init__: it would allocate a
        # fresh random weight. We only set up plain nn.Module bookkeeping and
        # then reuse the trained pieces from the existing gate.
        torch.nn.Module.__init__(self)

        self.top_k = gate.top_k
        self.num_experts = gate.num_experts
        self.hidden_dim = gate.hidden_dim
        self.weight = gate.weight  # share the SAME trained parameter, no copy

        self.kind = policy["kind"]

        if self.kind in ("topk", "rank_shift"):
            self.k = policy["k"]
            self.start = policy.get("start", 0)  # rank offset; topk implies 0
        elif self.kind == "fixed":
            # The explicit experts this layer must use, e.g. [0, 2]. A buffer so
            # it moves with .to(device) and is saved with the module.
            self.register_buffer(
                "forced_experts", torch.tensor(policy["experts"], dtype=torch.long)
            )
        else:
            raise ValueError(f"unknown policy kind: {self.kind!r}")

    def forward(self, hidden_states):
        # Score every expert exactly like the original router does.
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        seq_len = router_probs.shape[0]

        if self.kind == "topk":
            # Per token, the k experts with the highest probability.
            chosen, router_indices = router_probs.topk(self.k, dim=-1)  # (seq_len, k)

        elif self.kind == "rank_shift":
            # Per token, sort experts by probability (highest first) and take a
            # window of length k starting at rank `start` (0-indexed). start=1
            # skips each token's single best expert.
            order = torch.argsort(router_probs, dim=-1, descending=True)  # (seq_len, E)
            router_indices = order[:, self.start : self.start + self.k]   # (seq_len, k)
            chosen = router_probs.gather(-1, router_indices)              # (seq_len, k)

        else:  # "fixed": same experts for every token in this layer
            idx = self.forced_experts.to(router_probs.device)
            router_indices = idx.unsqueeze(0).expand(seq_len, -1)  # (seq_len, k')
            chosen = router_probs[:, idx]                          # (seq_len, k')

        # Keep only the chosen experts' probabilities and renormalize so they
        # sum to 1 (Option A). This is the single line everything hinges on.
        router_scores = chosen / chosen.sum(dim=-1, keepdim=True)

        # Same (logits, scores, indices) contract the MoE block expects.
        return router_logits, router_scores, router_indices


def get_moe_blocks(model):
    """Return the MoE blocks in order, one per decoder layer."""
    return [m for m in model.modules() if type(m).__name__ == MOE_BLOCK_NAME]


def _expand_member(member, n_layers):
    """
    Normalize a `member` argument into a list of per-layer policy dicts.

    Accepts either:
      - a policy dict applied to EVERY layer, e.g. {"kind": "topk", "k": 3}; or
      - a list of explicit expert sets, one per layer (the "fixed" policy),
        e.g. [[1], [3]] for a 2-layer model.
    """
    if isinstance(member, dict):
        return [member] * n_layers

    # Otherwise it is a per-layer list of expert indices ("fixed" policy).
    if len(member) != n_layers:
        raise ValueError(
            f"member has {len(member)} layer specs but model has {n_layers} MoE layers"
        )
    return [{"kind": "fixed", "experts": experts} for experts in member]


def set_member(model, member):
    """
    Turn `model` into one ensemble member.

    `member` is either a policy dict (applied to all layers) or a per-layer list
    of expert indices. Examples:
      - set_member(model, topk_member(3))            # Top-3
      - set_member(model, rank_shift_member(1, 2))   # ranks 2-3
      - set_member(model, random_member(model, 2))   # random-expert control
      - set_member(model, [[1], [3]])                # explicit, for debugging

    We keep each original gate on the block (`block._original_gate`) so the
    model can be restored later with `restore(model)`.
    """
    blocks = get_moe_blocks(model)
    policies = _expand_member(member, len(blocks))

    for block, policy in zip(blocks, policies):
        if not hasattr(block, "_original_gate"):
            block._original_gate = block.gate  # remember the real router once
        block.gate = MemberRouter(block._original_gate, policy)


def restore(model):
    """Undo `set_member`: put every original router back."""
    for block in get_moe_blocks(model):
        if hasattr(block, "_original_gate"):
            block.gate = block._original_gate
            del block._original_gate


# --- Member builders -------------------------------------------------------
# Small helpers so experiment code reads like the plan (Top-3, rank-shift, ...).


def topk_member(k):
    """Per token, the gate's k highest-ranked experts. k=3 is the Top-3 member."""
    return {"kind": "topk", "k": k}


def rank_shift_member(start, k):
    """
    Per token, k experts from the ranking starting at rank `start` (0-indexed),
    skipping the higher ranks. start=1, k=2 -> ranks 2-3.
    """
    return {"kind": "rank_shift", "k": k, "start": start}


def random_member(model, k, seed=0):
    """
    Build a random-expert control: k experts chosen uniformly at random per
    layer, the SAME set for every token (gate used only for the weighting).

    Returns a per-layer list (the "fixed" policy), so it flows through
    `set_member` like any explicit member. Seeded for reproducibility.
    """
    blocks = get_moe_blocks(model)
    # Read expert count from whatever gate is currently on the block.
    num_experts = blocks[0].gate.num_experts
    rng = random.Random(seed)
    return [sorted(rng.sample(range(num_experts), k)) for _ in blocks]


def record_routing(model):
    """
    Attach a forward hook to every router that records which experts it picked.

    Returns (handles, log):
      - log[i] gets filled with the (seq_len, k) tensor of expert indices the
        i-th layer's router selected on the next forward pass.
      - call h.remove() for each handle in handles when done.
    """
    log = {}
    handles = []
    for i, block in enumerate(get_moe_blocks(model)):
        def hook(_module, _inputs, output, i=i):
            # router forward returns (router_logits, router_scores, router_indices)
            log[i] = output[2].detach().cpu()
        handles.append(block.gate.register_forward_hook(hook))
    return handles, log
