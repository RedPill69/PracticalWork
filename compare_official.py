"""
compare_official.py

How do our baseline numbers compare to the officially reported Mixtral-8x7B
results? Pure post-processing of the saved eval files from run_eval.py - no
model, no GPU.

The catch: "accuracy" is not one number. lm-eval logs two metrics per doc:
  - acc      : argmax of the RAW per-choice log-likelihoods (what analyze.py
               uses everywhere, chosen for consistency with the uncertainty
               math, which needs one distribution over choices);
  - acc_norm : argmax of the log-likelihoods normalized by the BYTE LENGTH of
               each choice - corrects for long answers accumulating more
               negative log-likelihood. Official numbers for ARC/HellaSwag
               are acc_norm; MMLU has same-length choices (A/B/C/D), so only
               acc exists there.

Reference numbers this script compares against (hardcoded below):
  - Mixtral paper (Jiang et al. 2024, arXiv:2401.04088, Table 2):
      MMLU 70.6 (5-shot), ARC-c 59.7 (25-shot), HellaSwag 84.4 (10-shot)
  - HF Open LLM Leaderboard v1 (lm-eval-harness, fp16, run 2024-01-04):
      MMLU acc .716 (5-shot), ARC acc .637 / acc_norm .664 (25-shot),
      HellaSwag acc .670 / acc_norm .865 (10-shot)

Known, expected reasons our numbers can differ (keep in mind when reading):
  - fewshot count: we run 5-shot everywhere; official ARC is 25-shot and
    HellaSwag 10-shot (more shots help);
  - subset: our limit N takes the FIRST N docs of each (sub)task, not a
    random sample - ordering bias is possible on top of the small-n noise;
  - precision: we run 4-bit nf4, official numbers are fp16/bf16 - the residual
    gap after accounting for metric+shots bounds what higher precision could
    recover;
  - harness details: the paper uses Mistral's internal pipeline, not lm-eval.

Run from the Code folder (point --dir at a results folder):

    python compare_official.py --dir results/2026-06-16_limit100
"""

import argparse

import numpy as np

from analyze import load_runs, all_leaf_tasks, leaves_for

# task -> {metric -> {source -> reported value}}; None = metric not defined.
OFFICIAL = {
    "mmlu": {
        "acc":      {"paper (5-shot)": 0.706, "leaderboard (5-shot)": 0.716},
        "acc_norm": None,
    },
    "arc_challenge": {
        "acc":      {"leaderboard (25-shot)": 0.637},
        "acc_norm": {"paper (25-shot)": 0.597, "leaderboard (25-shot)": 0.664},
    },
    "hellaswag": {
        "acc":      {"leaderboard (10-shot)": 0.670},
        "acc_norm": {"paper (10-shot)": 0.844, "leaderboard (10-shot)": 0.865},
    },
}


def logged_metric(run, leaves, metric):
    """
    Mean of a per-doc metric ("acc" or "acc_norm") that lm-eval logged into
    the samples, pooled over the leaf tasks. None if absent (acc_norm on MMLU).

    Unlike analyze.py we do not recompute correctness here - the point is to
    read out exactly what lm-eval itself scored, per metric.
    """
    vals = []
    for t in leaves:
        for s in run["samples"].get(t, []):
            if metric in s:
                vals.append(float(s[metric]))  # may be str (default=str on save)
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results", help="results folder from run_eval.py")
    args = p.parse_args()

    manifest, runs = load_runs(args.dir)
    baseline = next(m for m, r in manifest["members"].items() if r == "baseline")
    run = runs[baseline]
    leaves = all_leaf_tasks(runs)

    print(f"Baseline member: {baseline}   "
          f"(ours: {manifest['num_fewshot']}-shot, limit {manifest['limit']}, 4-bit)")

    for task in manifest["tasks"]:
        task_leaves = leaves_for(task, leaves)
        if not task_leaves:
            continue
        print(f"\n=== {task} ===")
        for metric in ("acc", "acc_norm"):
            ours = logged_metric(run, task_leaves, metric)
            if ours is None:
                continue
            refs = (OFFICIAL.get(task) or {}).get(metric)
            print(f"  {metric:9s} ours: {ours:.3f}")
            for src, val in (refs or {}).items():
                print(f"            {src:22s} {val:.3f}   (delta {ours - val:+.3f})")

    print("\nNote: deltas mix four effects - metric definition, fewshot count, "
          "first-N subset, 4-bit quantization. See the header comment.")


if __name__ == "__main__":
    main()
