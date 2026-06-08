"""
check_routing.py

Verifies the routing override on the tiny Mixtral model. It checks three things:

  1. BASELINE: with no changes, record which experts the router naturally picks.
  2. FORCED:   force a specific expert per layer and confirm the router now
               reports exactly those experts (the override took effect).
  3. EFFECT:   the model's output logits change between two different members,
               proving the forced expert actually "fires" (it is not a no-op).

Run it from the Code folder with:

    python check_routing.py
"""

import torch

from sanity_check import load_config, set_seed, load_model
from routing import set_member, restore, record_routing


def forward_logits(model, inputs):
    """Run one forward pass and return the raw output logits."""
    with torch.no_grad():
        return model(**inputs).logits


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    tokenizer, model, device = load_model(cfg)
    inputs = tokenizer(cfg["prompt"], return_tensors="pt").to(device)

    # --- 1. Baseline: what does the unmodified router pick? ---------------
    handles, log = record_routing(model)
    base_logits = forward_logits(model, inputs)
    for h in handles:
        h.remove()
    print("=== Baseline routing (natural top-2 per layer) ===")
    for layer, idx in log.items():
        # idx is (seq_len, top_k): show the experts picked for each token.
        print(f"  layer {layer}: experts per token =\n{idx.tolist()}")

    # --- 2. Force a specific expert per layer ------------------------------
    # Tiny model = 2 layers, 4 experts. Force a single expert in each layer.
    member = [[1], [3]]
    set_member(model, member)

    handles, log = record_routing(model)
    forced_logits = forward_logits(model, inputs)
    for h in handles:
        h.remove()

    print(f"\n=== Forced routing, member = {member} ===")
    all_ok = True
    for layer, idx in log.items():
        picked = sorted(set(int(e) for row in idx.tolist() for e in row))
        expected = member[layer]
        ok = picked == expected
        all_ok = all_ok and ok
        print(f"  layer {layer}: experts used = {picked}  expected {expected}  {'OK' if ok else 'WRONG'}")

    # --- 3. A different member must give different outputs -----------------
    restore(model)
    set_member(model, [[0], [2]])
    other_logits = forward_logits(model, inputs)
    restore(model)

    changed = not torch.allclose(forced_logits, other_logits)
    differs_from_base = not torch.allclose(forced_logits, base_logits)

    print("\n=== Effect check ===")
    print(f"  forced member differs from baseline:        {differs_from_base}")
    print(f"  two different members give different output: {changed}")

    print("\nRESULT:", "ALL CHECKS PASSED" if (all_ok and changed and differs_from_base) else "SOMETHING IS OFF")


if __name__ == "__main__":
    main()
