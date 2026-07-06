"""
check_routing.py

Verifies the routing policies on the tiny Mixtral model. The tiny model has
RANDOM weights, so its predictions are meaningless - we only check that each
policy selects the experts it promises and that the override actually changes
the output (it is not a silent no-op). Real accuracy comes later, on the full
Mixtral, in the benchmark harness.

Checks:
  1. BASELINE   : record the natural top-2 the gate picks per token.
  2. TOPK(3)    : every token uses exactly 3 experts, and the baseline top-2 are
                  a subset of them (top-3 must contain top-2).
  3. RANK_SHIFT : ranks 2-3 per token -> exactly 2 experts, and each token's
                  single best (baseline rank-1) expert is NOT among them.
  4. RANKS      : [0, 2] per token -> exactly 2 experts, containing the
                  baseline rank-1 but NOT the baseline rank-2 (the pair member);
                  [0] -> exactly the baseline rank-1 alone.
  5. DROP       : the dropped expert is never selected; tokens whose baseline
                  top-2 did not contain it are routed exactly as the baseline.
  6. NOISE      : sigma=0 reproduces the baseline selection exactly; the same
                  seed gives the same selection twice (fresh apply); a
                  different seed changes the selection; always 2 experts.
  7. RANDOM     : the same randomly drawn set for every token, reproducible.
  8. FIXED      : an explicit per-layer set is forced exactly.
  9. EFFECT     : two different members give different output logits.

Run it from the Code folder with:

    python check_routing.py
"""

import torch

from sanity_check import load_config, config_arg, set_seed, load_model
from routing import (
    set_member,
    restore,
    record_routing,
    topk_member,
    rank_shift_member,
    rank_select_member,
    drop_expert_member,
    gate_noise_member,
    random_member,
)


def forward_logits(model, inputs):
    """Run one forward pass and return the raw output logits."""
    with torch.no_grad():
        return model(**inputs).logits


def routing_log(model, inputs):
    """Run one forward pass and return {layer: (seq_len, k) expert-index tensor}."""
    handles, log = record_routing(model)
    forward_logits(model, inputs)
    for h in handles:
        h.remove()
    return log


def main():
    cfg = load_config(config_arg())
    set_seed(cfg["seed"])
    tokenizer, model, device = load_model(cfg)
    inputs = tokenizer(cfg["prompt"], return_tensors="pt").to(model.device)

    results = []  # (name, ok) per check

    # --- 1. Baseline: what does the unmodified router pick? ---------------
    base_log = routing_log(model, inputs)
    base_logits = forward_logits(model, inputs)
    print("=== Baseline routing (natural top-2 per layer) ===")
    for layer, idx in base_log.items():
        print(f"  layer {layer}: experts per token =\n{idx.tolist()}")
    # Per-token baseline top-1 (rank-1) and the top-2 set, used by later checks.
    base_top1 = {layer: idx[:, 0].tolist() for layer, idx in base_log.items()}
    base_top2 = {
        layer: [set(int(e) for e in row) for row in idx.tolist()]
        for layer, idx in base_log.items()
    }

    # --- 2. TOPK(3): exactly 3 experts/token, and they contain the top-2 ---
    set_member(model, topk_member(3))
    log = routing_log(model, inputs)
    restore(model)
    ok = True
    for layer, idx in log.items():
        for tok, row in enumerate(idx.tolist()):
            chosen = set(int(e) for e in row)
            ok = ok and len(chosen) == 3 and base_top2[layer][tok] <= chosen
    results.append(("topk(3): 3 experts/token, contains baseline top-2", ok))

    # --- 3. RANK_SHIFT(start=1, k=2): ranks 2-3, skips each token's best ---
    set_member(model, rank_shift_member(start=1, k=2))
    log = routing_log(model, inputs)
    restore(model)
    ok = True
    for layer, idx in log.items():
        for tok, row in enumerate(idx.tolist()):
            chosen = set(int(e) for e in row)
            ok = ok and len(chosen) == 2 and base_top1[layer][tok] not in chosen
    results.append(("rank_shift(1,2): 2 experts/token, excludes baseline rank-1", ok))

    # --- 4. RANKS: [0, 2] keeps rank-1 + rank-3; [0] keeps rank-1 alone ----
    # Per-token baseline rank-2, to verify the pair skips it.
    base_rank2 = {layer: idx[:, 1].tolist() for layer, idx in base_log.items()}

    set_member(model, rank_select_member([0, 2]))
    log = routing_log(model, inputs)
    restore(model)
    ok = True
    for layer, idx in log.items():
        for tok, row in enumerate(idx.tolist()):
            chosen = set(int(e) for e in row)
            ok = (ok and len(chosen) == 2
                  and base_top1[layer][tok] in chosen
                  and base_rank2[layer][tok] not in chosen)
    results.append(("ranks([0,2]): keeps baseline rank-1, skips rank-2", ok))

    set_member(model, rank_select_member([0]))
    log = routing_log(model, inputs)
    restore(model)
    ok = True
    for layer, idx in log.items():
        for tok, row in enumerate(idx.tolist()):
            ok = ok and set(int(e) for e in row) == {base_top1[layer][tok]}
    results.append(("ranks([0]): exactly the baseline rank-1 expert", ok))

    # --- 5. DROP: dropped expert gone, unaffected tokens routed as baseline
    drop = 1
    set_member(model, drop_expert_member(drop))
    log = routing_log(model, inputs)
    restore(model)
    ok = True
    for layer, idx in log.items():
        for tok, row in enumerate(idx.tolist()):
            chosen = set(int(e) for e in row)
            ok = ok and len(chosen) == 2 and drop not in chosen
            if drop not in base_top2[layer][tok]:
                # This token never wanted the dropped expert: identical routing.
                ok = ok and chosen == base_top2[layer][tok]
    results.append((f"drop({drop}): expert {drop} gone, others routed as baseline", ok))

    # --- 6. NOISE: sigma=0 == baseline; same seed reproducible; seeds differ
    set_member(model, gate_noise_member(sigma=0.0, seed=0))
    log = routing_log(model, inputs)
    restore(model)
    ok = True
    for layer, idx in log.items():
        for tok, row in enumerate(idx.tolist()):
            ok = ok and set(int(e) for e in row) == base_top2[layer][tok]
    results.append(("noise(sigma=0): identical to the baseline selection", ok))

    def noise_selection(sigma, seed):
        set_member(model, gate_noise_member(sigma, seed))
        log = routing_log(model, inputs)
        restore(model)
        return {layer: idx.tolist() for layer, idx in log.items()}

    sel_a = noise_selection(5.0, seed=0)
    sel_b = noise_selection(5.0, seed=0)   # fresh apply, same seed
    sel_c = noise_selection(5.0, seed=1)
    two_experts = all(
        len(set(row)) == 2 for sel in (sel_a, sel_c)
        for rows in sel.values() for row in rows
    )
    results.append(("noise(5.0): same seed reproduces, new seed differs, 2 experts",
                    sel_a == sel_b and sel_a != sel_c and two_experts))

    # --- 7. RANDOM: same drawn set for every token, reproducible ----------
    member_a = random_member(model, k=2, seed=0)
    member_b = random_member(model, k=2, seed=0)
    reproducible = member_a == member_b
    set_member(model, member_a)
    log = routing_log(model, inputs)
    restore(model)
    ok = reproducible
    for layer, idx in log.items():
        rows = [sorted(set(int(e) for e in row)) for row in idx.tolist()]
        same_every_token = all(r == sorted(member_a[layer]) for r in rows)
        ok = ok and same_every_token
    results.append(("random(k=2): fixed drawn set per layer, reproducible", ok))

    # --- 8. FIXED: an explicit per-layer set is forced exactly ------------
    # Tiny model = 2 layers, 4 experts.
    fixed = [[1], [3]]
    set_member(model, fixed)
    log = routing_log(model, inputs)
    restore(model)
    ok = True
    for layer, idx in log.items():
        picked = sorted(set(int(e) for row in idx.tolist() for e in row))
        ok = ok and picked == fixed[layer]
    results.append(("fixed [[1],[3]]: forces exactly those experts", ok))

    # --- 9. Effect: two different members give different outputs ----------
    set_member(model, topk_member(3))
    logits_a = forward_logits(model, inputs)
    restore(model)
    set_member(model, rank_shift_member(start=1, k=2))
    logits_b = forward_logits(model, inputs)
    restore(model)
    effect = (
        not torch.allclose(logits_a, logits_b)
        and not torch.allclose(logits_a, base_logits)
    )
    results.append(("members change the output logits (not a no-op)", effect))

    # --- Report -----------------------------------------------------------
    print("\n=== Policy checks ===")
    all_ok = True
    for name, ok in results:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print("\nRESULT:", "ALL CHECKS PASSED" if all_ok else "SOMETHING IS OFF")


if __name__ == "__main__":
    main()
