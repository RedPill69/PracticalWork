"""
chat.py

A tiny interactive REPL to talk to the model with one ensemble MEMBER active, so
you can feel the difference between the routing variants by hand. Type a prompt,
get a completion; switch which experts are active with /member.

Run it from the Code folder with:

    python chat.py                   # tiny model (local.yaml) - output is gibberish
    python chat.py --config server.yaml   # real Mixtral in 4-bit on a GPU

Commands inside the chat:
    /member top2                 unmodified Mixtral (baseline)
    /member top3                 the 3 highest-ranked experts per token
    /member rank23               ranks 2-3 (mild shift)
    /member rank34               ranks 3-4 (the "second-best route")
    /member random               random experts per layer (control)
    /member fixed 0 2            force explicit experts (same set every layer)
    /quit                        leave
"""

import sys

import torch

from sanity_check import load_config, config_arg, set_seed, load_model

# Model output can contain any unicode; make printing robust on Windows consoles
# (whose default cp1252 encoding would otherwise crash on e.g. CJK characters).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass
from routing import (
    set_member,
    restore,
    topk_member,
    rank_shift_member,
    random_member,
)


def build_member(model, name, rest):
    """Translate a /member command into a routing member (or None for baseline)."""
    if name == "top2":
        return None                       # unmodified model
    if name == "top3":
        return topk_member(3)
    if name == "rank23":
        return rank_shift_member(1, 2)
    if name == "rank34":
        return rank_shift_member(2, 2)
    if name == "random":
        return random_member(model, 2, 0)
    if name == "fixed":
        # rest = list of expert indices, e.g. "0 2"; same set for every layer.
        experts = [int(x) for x in rest]
        n_layers = len(routing_layers(model))
        return [experts for _ in range(n_layers)]
    raise ValueError(f"unknown member {name!r}")


def routing_layers(model):
    from routing import get_moe_blocks
    return get_moe_blocks(model)


def generate(model, tokenizer, prompt, gen_cfg):
    """Generate a completion for one prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 40),
            temperature=gen_cfg.get("temperature", 0.8),
            do_sample=gen_cfg.get("do_sample", True),
            pad_token_id=tokenizer.eos_token_id,
        )
    # Only decode the newly generated tokens (drop the prompt).
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    cfg = load_config(config_arg())
    set_seed(cfg["seed"])
    tokenizer, model, _ = load_model(cfg)
    gen_cfg = cfg.get("chat", {})

    active = "top2"  # which member is currently applied
    print("Chat ready. Active member: top2 (baseline).")
    print("Switch with e.g. '/member top3'. Leave with '/quit'.\n")

    while True:
        try:
            line = input(f"[{active}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line == "/quit":
            break

        if line.startswith("/member"):
            parts = line.split()
            name, rest = parts[1] if len(parts) > 1 else "", parts[2:]
            try:
                member = build_member(model, name, rest)
            except ValueError as e:
                print(f"  {e}")
                continue
            restore(model)                 # always start from the clean model
            if member is not None:
                set_member(model, member)
            active = name
            print(f"  active member -> {active}")
            continue

        # Otherwise treat the line as a prompt.
        print(generate(model, tokenizer, line, gen_cfg))

    restore(model)
    print("bye.")


if __name__ == "__main__":
    main()
