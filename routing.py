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
  A "member" of our ensemble = the same model, but with a fixed RULE for which
  experts each layer must use (e.g. layer 0 -> experts [0, 2]). We swap the
  router for a `MemberRouter` that forces those experts and recomputes the
  softmax weights over ONLY the forced experts so they still sum to 1.

  That last point is the "Option A" renormalization from the project notes:
  if we force a single expert its weight becomes 1.0 (NOT the old ~0.5 top-2
  weight), so the FFN contribution is not silently halved across all layers.

In this transformers version the routing decision lives in the router submodule
`MixtralTopKRouter`, NOT in `MixtralSparseMoeBlock.forward` - so the router is
exactly where we intervene.
"""

import torch
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import MixtralTopKRouter

# The MoE block class name we look for when walking the model.
MOE_BLOCK_NAME = "MixtralSparseMoeBlock"


class MemberRouter(MixtralTopKRouter):
    """
    A drop-in replacement for Mixtral's router that FORCES a fixed set of
    experts for this layer instead of taking the natural top-k.

    Built from the layer's existing gate so it reuses the SAME trained weight
    (we never create new random router weights).
    """

    def __init__(self, gate, forced_experts):
        # Deliberately skip MixtralTopKRouter.__init__: it would allocate a
        # fresh random weight. We only set up plain nn.Module bookkeeping and
        # then reuse the trained pieces from the existing gate.
        torch.nn.Module.__init__(self)

        self.top_k = gate.top_k
        self.num_experts = gate.num_experts
        self.hidden_dim = gate.hidden_dim
        self.weight = gate.weight  # share the SAME trained parameter, no copy

        # The experts this layer is forced to use, e.g. [0, 2]. Registered as a
        # buffer so it moves with .to(device) and is saved with the module.
        self.register_buffer(
            "forced_experts", torch.tensor(forced_experts, dtype=torch.long)
        )

    def forward(self, hidden_states):
        # Score every expert exactly like the original router does.
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.softmax(router_logits.float(), dim=-1)

        seq_len = router_probs.shape[0]
        idx = self.forced_experts.to(router_probs.device)

        # Every token in this layer uses the SAME forced experts.
        router_indices = idx.unsqueeze(0).expand(seq_len, -1)  # (seq_len, k')

        # Keep only the forced experts' probabilities and renormalize so they
        # sum to 1 (Option A). This is the single line everything hinges on.
        chosen = router_probs[:, idx]  # (seq_len, k')
        router_scores = chosen / chosen.sum(dim=-1, keepdim=True)

        # Same (logits, scores, indices) contract the MoE block expects.
        return router_logits, router_scores, router_indices


def get_moe_blocks(model):
    """Return the MoE blocks in order, one per decoder layer."""
    return [m for m in model.modules() if type(m).__name__ == MOE_BLOCK_NAME]


def set_member(model, member):
    """
    Turn `model` into one ensemble member.

    `member` is one list of expert indices PER decoder layer, e.g. for the tiny
    2-layer model: [[0, 1], [2, 3]]. Pass a single-element list like [[3]] per
    layer to force exactly one expert.

    We keep each original gate on the block (`block._original_gate`) so the
    model can be restored later with `restore(model)`.
    """
    blocks = get_moe_blocks(model)
    if len(member) != len(blocks):
        raise ValueError(
            f"member has {len(member)} layer specs but model has {len(blocks)} MoE layers"
        )

    for block, forced_experts in zip(blocks, member):
        if not hasattr(block, "_original_gate"):
            block._original_gate = block.gate  # remember the real router once
        block.gate = MemberRouter(block._original_gate, forced_experts)


def restore(model):
    """Undo `set_member`: put every original router back."""
    for block in get_moe_blocks(model):
        if hasattr(block, "_original_gate"):
            block.gate = block._original_gate
            del block._original_gate


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
