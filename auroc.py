"""
auroc.py

Does the ensemble's uncertainty actually PREDICT errors? Pure post-processing
of the saved eval files from run_eval.py - no model, no GPU.

The decomposition (analyze.py) reports one mean uncertainty per run; this
script asks the per-question version: if questions are ranked by an
uncertainty signal, do the wrongly answered ones end up on top? Two standard
views:

  - AUROC: the probability that a randomly chosen wrong question gets a higher
    uncertainty than a randomly chosen right one (0.5 = useless, 1.0 = perfect).
  - Selective prediction: answer only the fraction of questions the signal is
    most certain about, abstain on the rest; report accuracy on the kept part.

Signals, all computed per question over the "principled" ensemble members:

  - epistemic    : total - aleatoric = mean KL of members to the ensemble mean
                   (the disagreement term - the signal this project is about)
  - total        : entropy of the ensemble-mean distribution
  - aleatoric    : mean entropy of the member distributions
  - base_entropy : entropy of the BASELINE's own distribution. This is the
                   single-model signal every practitioner already has for free,
                   so it is the reference the ensemble signals must beat.

Errors are judged for two predictors: the baseline member (can the signal flag
the unmodified model's mistakes?) and the ensemble-mean prediction (the
ensemble as its own predictor).

Run from the Code folder (point --dir at a results folder):

    python auroc.py --dir results/2026-07-06_limit100
"""

import os
import json
import argparse

import numpy as np

from analyze import (
    load_runs,
    member_families,
    choice_loglikelihoods,
    entropy,
    all_leaf_tasks,
    leaves_for,
)


def gather(runs, members, leaves):
    """
    Collect, per question, each member's answer distribution and the gold index.

    Returns (dists, gold):
      dists: member -> {doc_key -> softmax distribution over the choices}
      gold:  doc_key -> index of the correct choice
    Doc keys are "task:doc_id" so questions from different leaf tasks never
    collide (same convention as analyze.py / overlap.py).
    """
    dists = {m: {} for m in members}
    gold = {}
    for m in members:
        for t in leaves:
            for s in runs[m]["samples"].get(t, []):
                key = f"{t}:{s['doc_id']}"
                lls = choice_loglikelihoods(s)
                if not np.isfinite(lls).all():
                    # Same rationale as analyze.per_doc: skip the doc for this
                    # member, and the intersection rule keeps every signal
                    # finite instead of poisoning a family's stats with NaN.
                    continue
                p = np.exp(lls - lls.max())
                dists[m][key] = p / p.sum()
                gold[key] = int(s["target"])
    return dists, gold


def auroc(scores, labels):
    """
    AUROC of `scores` for detecting labels==1, via the rank (Mann-Whitney U)
    formulation with average ranks for ties. No sklearn dependency.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    # Average rank per unique score value: a value occupying sorted positions
    # [cum-count+1, cum] gets their mean rank cum - (count-1)/2.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    ranks = (cum - (counts - 1) / 2.0)[inverse]
    u = ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def selective_accuracy(scores, correct, coverages):
    """
    Accuracy on the `coverage` fraction of questions with the LOWEST scores
    (most certain kept, most uncertain abstained), for each coverage level.
    """
    order = np.argsort(scores)            # ascending: most certain first
    correct = np.asarray(correct, dtype=float)[order]
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(correct))))
        out[c] = float(correct[:k].mean())
    return out


def analyze_group(dists, gold, principled, baseline):
    """All error-prediction stats for one task group."""
    # Only questions every needed member answered (analyze.py intersection rule).
    needed = principled + [baseline]
    common = sorted(set.intersection(*(set(dists[m]) for m in needed)))

    signals = {"epistemic": [], "total": [], "aleatoric": [], "base_entropy": []}
    base_correct, ens_correct = [], []
    for key in common:
        ps = np.stack([dists[m][key] for m in principled])   # (M, C)
        pbar = ps.mean(axis=0)
        total = entropy(pbar)
        aleatoric = float(np.mean([entropy(p) for p in ps]))
        signals["total"].append(total)
        signals["aleatoric"].append(aleatoric)
        signals["epistemic"].append(total - aleatoric)
        signals["base_entropy"].append(entropy(dists[baseline][key]))
        base_correct.append(int(int(dists[baseline][key].argmax()) == gold[key]))
        ens_correct.append(int(int(pbar.argmax()) == gold[key]))

    base_errors = 1 - np.array(base_correct)
    ens_errors = 1 - np.array(ens_correct)

    coverages = [1.0, 0.9, 0.8, 0.7, 0.5]
    return {
        "n_docs": len(common),
        "base_acc": float(np.mean(base_correct)),
        "ensemble_acc": float(np.mean(ens_correct)),
        "auroc": {
            name: {
                "baseline_errors": auroc(vals, base_errors),
                "ensemble_errors": auroc(vals, ens_errors),
            }
            for name, vals in signals.items()
        },
        # Selective prediction judged on the baseline's correctness: keep the
        # most-certain fraction of questions, how accurate is the baseline there?
        "selective_base_acc": {
            name: selective_accuracy(vals, base_correct, coverages)
            for name, vals in signals.items()
        },
    }


def print_group(task, stats):
    print(f"\n=== {task} (n={stats['n_docs']}, "
          f"base acc {stats['base_acc']:.3f}, "
          f"ensemble acc {stats['ensemble_acc']:.3f}) ===")

    print(f"  {'signal':14s} {'AUROC base-err':>15s} {'AUROC ens-err':>14s}")
    for name, a in stats["auroc"].items():
        print(f"  {name:14s} {a['baseline_errors']:15.3f} {a['ensemble_errors']:14.3f}")

    covs = list(next(iter(stats["selective_base_acc"].values())))
    header = "".join(f"{int(c * 100):>7d}%" for c in covs)
    print(f"  baseline accuracy at coverage:{header}")
    for name, accs in stats["selective_base_acc"].items():
        row = "".join(f"{accs[c]:8.3f}" for c in covs)
        print(f"  {name:14s}{' ' * 16}{row}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results", help="results folder from run_eval.py")
    args = p.parse_args()

    manifest, runs = load_runs(args.dir)
    roles = manifest["members"]
    baseline = next(m for m, r in roles.items() if r == "baseline")
    # Signals are computed per extraction family (see analyze.member_families),
    # so ensembles of different mechanisms are never mixed.
    families = {fam: [m for m in ms if m in runs]
                for fam, ms in member_families(roles).items()}
    families = {fam: ms for fam, ms in families.items() if len(ms) >= 2}

    leaves = all_leaf_tasks(runs)
    groups = [(g, leaves_for(g, leaves)) for g in manifest["tasks"]
              if not g.startswith("probe_")]   # probes: no valid correctness
    groups = [(g, lv) for g, lv in groups if lv]
    benchmark_leaves = [t for t in leaves if not t.startswith("probe_")]
    if len(groups) > 1 and benchmark_leaves:
        groups.append(("overall", benchmark_leaves))

    out = {}
    for task, task_leaves in groups:
        out[task] = {}
        for fam, fam_members in families.items():
            dists, gold = gather(runs, fam_members + [baseline], task_leaves)
            out[task][fam] = analyze_group(dists, gold, fam_members, baseline)
            print_group(f"{task} [{fam}]", out[task][fam])

    path = os.path.join(args.dir, "auroc.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
