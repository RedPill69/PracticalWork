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
    drop_expert_member,
    gate_noise_member,
    get_moe_blocks,
)


def build_members(model):
    """
    The ensemble member set. Each value is the argument to set_member (or None for
    the unmodified baseline). `role` drives the analysis: "baseline" is the
    reference for the accuracy loss; "principled/<family>" members form the
    ensembles for the uncertainty decomposition, grouped per family so that two
    extraction mechanisms are never mixed into one ensemble; "control" would be
    the random baseline and "anchor" a member shown in the accuracy table but
    kept out of the ensemble math.

    This set compares two extraction mechanisms that perturb each token's
    TOP-ranked expert selectively. The earlier scans showed that the top-ranked
    expert carries the model (removing it everywhere collapses accuracy to
    chance) while swapping the low-weight partner changes little - so the
    remaining design space is dosed perturbation of the top choice:

      - drop_<e> ("jackknife"): expert e is removed from every layer's pool and
        the router picks the best remaining experts. A token is affected only
        in layers where e was among its top choices, so each member deviates
        from the baseline exactly where "its" expert mattered.
      - noise_<sigma>_<seed> ("MC-router"): seeded Gaussian noise, sigma times
        the token's own logit spread, on the gate scores. Only near-tied
        routing decisions flip, so members diverge where the router is least
        decided. Two sigmas bracket the scale; two seeds per sigma give a
        minimal ensemble each.

    Deterministic members measured in earlier runs (the rank-anchored pairs,
    the random control) are not repeated: on the identical eval subset they
    reproduce exactly; their results live in the dated results folders.
    """
    num_experts = get_moe_blocks(model)[0].num_experts

    members = {"top2_baseline": {"member": None, "role": "baseline"}}
    for e in range(num_experts):
        members[f"drop_{e}"] = {
            "member": drop_expert_member(e),
            "role": "principled/drop",
        }
    for sigma in (0.5, 1.0):
        for seed in (0, 1):
            members[f"noise_{sigma}_{seed}"] = {
                "member": gate_noise_member(sigma, seed),
                "role": f"principled/noise{sigma}",
            }
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
