"""
routing.py

This is the heart of Stage 1: turning ONE Mixtral model into an "ensemble" by
changing which experts the router picks - without copying weights or retraining.

Background (how Mixtral routes):
  Each decoder layer has a Mixture-of-Experts block. For every token, a small
  linear layer (the "gate") scores all experts, the block keeps the top 2, runs
  a softmax over those 2 scores, and the layer output is the weighted sum of the
  2 chosen expert FFNs.

What we do here:
  A "member" of our ensemble = the same model run with a fixed RULE ("policy")
  for which experts each layer uses. We wrap the layer's gate in a `MemberGate`
  that MASKS the gate logits, and we set the block's `top_k`, so the block's own
  selection (softmax -> topk -> renormalize) ends up choosing exactly the experts
  our policy wants. We do NOT touch the expert FFNs, so 4-bit quantized experts
  keep working untouched.

  Because the block renormalizes the softmax over only the chosen experts, the
  weights still sum to 1 - the "Option A" renormalization from the project notes:
  if a single expert is kept its weight becomes 1.0 (NOT the old ~0.5 top-2
  weight), so the FFN contribution is not silently halved across all layers.
  Masking the unchosen logits to -inf and letting the block renormalize is
  algebraically identical to softmax-over-all-then-renormalize-the-chosen.

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

In this transformers version (4.x) the routing decision lives INSIDE
`MixtralSparseMoeBlock.forward` (gate -> softmax -> topk -> renormalize), and the
gate itself is a plain `nn.Linear`. So we intervene on the gate (mask its logits)
and on the block's `top_k`, not on a separate router module.
"""

import random

import torch
from torch import nn

# The MoE block class name we look for when walking the model.
MOE_BLOCK_NAME = "MixtralSparseMoeBlock"

NEG_INF = float("-inf")


class MemberGate(nn.Module):
    """
    A drop-in replacement for a layer's gate (`nn.Linear`) that returns the SAME
    logits with some entries masked to -inf, so the MoE block's own top-k lands
    on the experts our policy wants.

    Reuses the layer's existing gate, so it shares the SAME trained weight
    (we never create new random gate weights). Pair it with the right `top_k` on
    the block (see `set_member`).
    """

    def __init__(self, gate, policy, num_experts):
        super().__init__()
        self.gate = gate  # the original nn.Linear, shared (no copy)
        self.num_experts = num_experts
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
        # Same scoring as the original gate. The block already reshaped
        # hidden_states to (num_tokens, hidden_dim) before calling us.
        logits = self.gate(hidden_states)  # (num_tokens, num_experts)

        if self.kind == "topk":
            # No masking: the block's topk with top_k=k picks the k best.
            return logits

        if self.kind == "rank_shift":
            # Mask each token's top `start` experts to -inf, so the block's topk
            # (with top_k=k) lands on ranks [start, start+k).
            if self.start > 0:
                order = torch.argsort(logits, dim=-1, descending=True)
                drop = order[:, : self.start]              # (num_tokens, start)
                logits = logits.clone()
                logits.scatter_(-1, drop, NEG_INF)
            return logits

        # "fixed": keep only the forced experts, mask everything else to -inf.
        idx = self.forced_experts.to(logits.device)
        masked = torch.full_like(logits, NEG_INF)
        masked[:, idx] = logits[:, idx]
        return masked


def get_moe_blocks(model):
    """Return the MoE blocks in order, one per decoder layer."""
    return [m for m in model.modules() if type(m).__name__ == MOE_BLOCK_NAME]


def _policy_top_k(policy):
    """How many experts per token this policy keeps (the block's `top_k`)."""
    if policy["kind"] in ("topk", "rank_shift"):
        return policy["k"]
    return len(policy["experts"])  # "fixed"


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

    We keep each original gate and top_k on the block (`block._original_gate`,
    `block._original_top_k`) so the model can be restored with `restore(model)`.
    """
    blocks = get_moe_blocks(model)
    policies = _expand_member(member, len(blocks))

    for block, policy in zip(blocks, policies):
        if not hasattr(block, "_original_gate"):
            block._original_gate = block.gate          # remember the real gate
            block._original_top_k = block.top_k        # ...and its top_k, once
        block.gate = MemberGate(block._original_gate, policy, block.num_experts)
        block.top_k = _policy_top_k(policy)


def restore(model):
    """Undo `set_member`: put every original gate and top_k back."""
    for block in get_moe_blocks(model):
        if hasattr(block, "_original_gate"):
            block.gate = block._original_gate
            block.top_k = block._original_top_k
            del block._original_gate
            del block._original_top_k


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
    num_experts = blocks[0].num_experts
    rng = random.Random(seed)
    return [sorted(rng.sample(range(num_experts), k)) for _ in blocks]


def record_routing(model):
    """
    Attach a forward hook to every MoE block that records which experts it picked.

    Returns (handles, log):
      - log[i] gets filled with the (num_tokens, k) tensor of expert indices the
        i-th layer selected on the next forward pass. We recompute the selection
        from the block's returned router_logits exactly as the block does
        (softmax -> topk with the block's current top_k), so it reflects whatever
        member is active (masked logits included).
      - call h.remove() for each handle in handles when done.
    """
    log = {}
    handles = []
    for i, block in enumerate(get_moe_blocks(model)):
        def hook(module, _inputs, output, i=i):
            # MixtralSparseMoeBlock.forward returns (final_hidden_states, router_logits)
            router_logits = output[1]
            probs = torch.softmax(router_logits, dim=-1)
            _, selected = probs.topk(module.top_k, dim=-1)  # (num_tokens, top_k)
            log[i] = selected.detach().cpu()
        handles.append(block.register_forward_hook(hook))
    return handles, log
