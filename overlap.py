"""
overlap.py

Do the ensemble members fail on the SAME questions or on DIFFERENT ones?
Pure post-processing of the saved eval files from run_eval.py - no model,
no GPU.

Why this matters: the epistemic term only carries signal if members disagree
in a structured way. Two extremes, both bad:
  - members err on exactly the same docs  -> no diversity, epistemic collapses
    (the shared-backbone worry, Kirsch 2025);
  - a member errs everywhere (chance level) -> its "disagreement" is noise,
    not knowledge (the rank_3_4 / random_k2 problem from the 2026-06-16 run).
The useful middle: members are individually strong AND their (few) errors
fall on different docs - then a doc where they disagree is informative.

What it reports, per task and pooled overall (same task grouping as analyze.py):
  - per-doc correctness histogram: on how many docs are 0,1,...,M members
    correct? (errors concentrated on the same docs -> mass at 0 and M)
  - pairwise overlap: agreement rate, error-set Jaccard, and the Jaccard
    expected if the two members' errors were INDEPENDENT - the reference
    point that tells us whether shared errors are more than coincidence
  - complementarity vs the Top-2 baseline: of the docs the baseline gets
    wrong, how many does each member rescue (get right)? And how many
    baseline-correct docs does it break?

Correctness = argmax of the raw log-likelihoods vs gold, identical to the
accuracy in analyze.py (the acc metric, not acc_norm - see compare_official.py
for that distinction).

Run from the Code folder (point --dir at a results folder):

    python overlap.py --dir results/2026-06-16_limit100
"""

import os
import json
import argparse
import itertools

import numpy as np

from analyze import load_runs, per_doc, all_leaf_tasks, leaves_for


def member_correct_by_doc(run, leaves):
    """doc-key -> 0/1 correctness for one member, pooled over the leaf tasks.

    Keys are "task:doc_id" (like member_dists in analyze.py), so docs from
    different leaf tasks never collide.
    """
    out = {}
    for t in leaves:
        if t in run["samples"]:
            _, c = per_doc(run["samples"][t])
            out.update({f"{t}:{k}": v for k, v in c.items()})
    return out


def jaccard(a, b):
    """|a & b| / |a | b| for two sets; 0 if both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def independent_jaccard(e1, e2):
    """
    Error-set Jaccard expected if two members erred INDEPENDENTLY with
    error rates e1, e2 (per doc: P(both wrong) / P(at least one wrong)).
    Measured Jaccard far above this = members share errors systematically.
    """
    denom = e1 + e2 - e1 * e2
    return (e1 * e2) / denom if denom > 0 else 0.0


def analyze_group(correct, members, baseline):
    """All overlap stats for one task group. `correct`: member -> {doc: 0/1}."""
    # Only docs every member answered (same intersection rule as analyze.py).
    common = sorted(set.intersection(*(set(correct[m]) for m in members)))
    n = len(common)
    flags = {m: np.array([correct[m][d] for d in common]) for m in members}
    errors = {m: {d for d in common if not correct[m][d]} for m in members}

    # How many members are correct per doc? (0..M histogram)
    n_correct = sum(flags[m] for m in members)
    hist = {int(k): int((n_correct == k).sum()) for k in range(len(members) + 1)}

    # Pairwise: agreement (same right/wrong verdict), error Jaccard vs the
    # independence reference.
    pairs = {}
    for a, b in itertools.combinations(members, 2):
        pairs[f"{a} vs {b}"] = {
            "agreement": float((flags[a] == flags[b]).mean()),
            "error_jaccard": jaccard(errors[a], errors[b]),
            "error_jaccard_if_independent": independent_jaccard(
                len(errors[a]) / n, len(errors[b]) / n),
        }

    # Complementarity vs the baseline: rescued = baseline wrong but member
    # right; broken = baseline right but member wrong.
    base_wrong = errors[baseline]
    base_right = set(common) - base_wrong
    comp = {}
    for m in members:
        if m == baseline:
            continue
        comp[m] = {
            "baseline_errors": len(base_wrong),
            "rescued": len({d for d in base_wrong if correct[m][d]}),
            "broken": len({d for d in base_right if not correct[m][d]}),
        }

    return {"n_docs": n, "docs_correct_histogram": hist,
            "pairwise": pairs, "vs_baseline": comp}


def print_group(task, stats, members):
    print(f"\n=== {task} (n={stats['n_docs']}) ===")

    hist = stats["docs_correct_histogram"]
    print("  members correct per doc:")
    for k in sorted(hist):
        print(f"    {k}/{len(members)} correct: {hist[k]:5d}")

    print(f"  {'pair':32s} {'agree':>6s} {'errJac':>7s} {'indep':>7s}")
    for pair, p in stats["pairwise"].items():
        print(f"  {pair:32s} {p['agreement']:6.3f} {p['error_jaccard']:7.3f} "
              f"{p['error_jaccard_if_independent']:7.3f}")

    print(f"  {'member':14s} {'rescued':>8s} {'broken':>7s}   (of "
          f"{next(iter(stats['vs_baseline'].values()))['baseline_errors']} baseline errors)")
    for m, c in stats["vs_baseline"].items():
        print(f"  {m:14s} {c['rescued']:8d} {c['broken']:7d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results", help="results folder from run_eval.py")
    args = p.parse_args()

    manifest, runs = load_runs(args.dir)
    baseline = next(m for m, r in manifest["members"].items() if r == "baseline")
    members = [m for m in manifest["members"] if m in runs]

    # Same task grouping as analyze.py: manifest tasks (+ pooled overall).
    leaves = all_leaf_tasks(runs)
    groups = [(g, leaves_for(g, leaves)) for g in manifest["tasks"]]
    groups = [(g, lv) for g, lv in groups if lv]
    if len(groups) > 1:
        groups.append(("overall", leaves))

    out = {}
    for task, task_leaves in groups:
        correct = {m: member_correct_by_doc(runs[m], task_leaves) for m in members}
        out[task] = analyze_group(correct, members, baseline)
        print_group(task, out[task], members)

    path = os.path.join(args.dir, "overlap.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
