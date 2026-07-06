"""
run_eval.py

Stage-1 benchmark, part 1 of 2: run the accuracy evaluation once per ensemble
member and save the raw results to disk. Part 2 (analyze.py) turns those results
into the accuracy table and the uncertainty decomposition.

We use the standard lm-evaluation-harness for the accuracy itself (correct task
protocols, comparable numbers), and just plug each member in by swapping the
router before the run. The same model object is reused for every member - we only
change its routing - so no weights are reloaded.

Run it from the Code folder with:

    python run_eval.py                    # tiny model, tiny limit (local.yaml)
    python run_eval.py --config server.yaml   # the real Mixtral run on a GPU

Output: one JSON file per member in cfg["eval"]["output_dir"], plus a
members.json manifest. log_samples is on, so each file contains the per-document
per-choice log-likelihoods we need for the uncertainty math.
"""

import os
import json
import argparse

import lm_eval
from lm_eval.models.huggingface import HFLM

from sanity_check import load_config, set_seed, load_model
from routing import (
    set_member,
    restore,
    rank_select_member,
    random_member,
    get_moe_blocks,
)


def build_members(model):
    """
    The ensemble member set. Each value is the argument to set_member (or None for
    the unmodified baseline). `role` drives the analysis: only "principled" members
    form the ensemble for the uncertainty decomposition; "control" is the random
    baseline; "baseline" is the reference for the accuracy loss; "anchor" appears
    in the accuracy table but stays out of the ensemble math.

    This set probes how much each token's top-ranked expert carries: the pair
    members keep rank 1 (so they stay competent) and vary only the partner rank
    (the diversity knob). The accuracy gradient over the partner rank, plus the
    rank-1-only anchor, shows how performance decays as the partner walks down
    the ranking. The first run (2026-06-16) showed that members WITHOUT rank 1
    (rank_2_3, rank_3_4) collapse to chance, so those are not repeated here.

    Member names use 1-indexed ranks (pair_1_3 = ranks 1 and 3); the code is
    0-indexed. Pairs whose partner rank does not exist on the current model
    (the tiny test model has only 4 experts) are skipped.
    """
    num_experts = get_moe_blocks(model)[0].num_experts

    members = {
        "top2_baseline": {"member": None,                    "role": "baseline"},
        "rank1_only":    {"member": rank_select_member([0]), "role": "anchor"},
    }
    for partner in (3, 4, 5, 8):          # 1-indexed rank of the second expert
        if partner <= num_experts:
            members[f"pair_1_{partner}"] = {
                "member": rank_select_member([0, partner - 1]),
                "role": "principled",
            }
    members["random_k2"] = {"member": random_member(model, 2, 0), "role": "control"}
    return members


def evaluate_member(model, tokenizer, ecfg):
    """Run lm-eval for the currently-applied member and return its result dict."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=ecfg["batch_size"])
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=list(ecfg["tasks"]),
        num_fewshot=ecfg["num_fewshot"],
        limit=ecfg["limit"],
        log_samples=True,     # we need the per-doc per-choice log-likelihoods
        bootstrap_iters=0,    # we compute our own confidence intervals in analyze.py
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="local.yaml", help="path to a YAML config")
    p.add_argument("--limit", type=int, default=None,
                   help="override eval.limit; use a tiny value for a GPU smoke run, "
                        "e.g. --config server.yaml --limit 2")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    ecfg = cfg["eval"]
    if args.limit is not None:
        ecfg["limit"] = args.limit
        print(f"(smoke run: eval.limit overridden to {args.limit})")
    out_dir = ecfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, model, _ = load_model(cfg)
    members = build_members(model)

    # Manifest so analyze.py knows the members, their roles, and the run settings.
    manifest = {
        "model_id": cfg["model"]["model_id"],
        "tasks": list(ecfg["tasks"]),
        "num_fewshot": ecfg["num_fewshot"],
        "limit": ecfg["limit"],
        "members": {name: spec["role"] for name, spec in members.items()},
    }
    with open(os.path.join(out_dir, "members.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for name, spec in members.items():
        print(f"\n=== Evaluating member: {name} ({spec['role']}) ===")
        if spec["member"] is not None:
            set_member(model, spec["member"])
        try:
            results = evaluate_member(model, tokenizer, ecfg)
        finally:
            restore(model)   # always return to the clean model, even on error

        path = os.path.join(out_dir, f"eval_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results["results"], "samples": results["samples"]},
                f, default=str,   # samples hold numbers/strings; default=str is a safety net
            )
        # Quick look at the headline metrics for this member.
        for task, metrics in results["results"].items():
            accs = {k: round(v, 4) for k, v in metrics.items()
                    if isinstance(v, float) and ("acc" in k)}
            print(f"  {task}: {accs}")
        print(f"  saved -> {path}")

    print(f"\nDone. Now run:  python analyze.py --config {config_path}")


if __name__ == "__main__":
    main()
