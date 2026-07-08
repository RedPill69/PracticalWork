"""
probe.py

Can the uncertainty signals tell "the model CANNOT know" from "the model
merely finds it hard"? Pure post-processing of the saved eval files from
run_eval.py - no model, no GPU.

The probe pair (tasks/probe_hellaswag_*.yaml) shares the same contexts; in the
unanswerable twin the true ending is replaced by another document's ending, so
no offered choice is correct. This is the discrimination task that plain
entropy cannot do even in principle (an ambiguous-but-answerable question and
an unanswerable one can produce the same spread), while the epistemic term
claims exactly this ability: members should disagree more where knowledge is
absent by construction.

Reported per member family (same signals as auroc.py):
  - AUROC of each signal for separating unanswerable (label 1) from
    answerable (label 0) questions, pooled over both probe tasks;
  - the mean of each signal on the two sides, to show the direction and size
    of the shift.

Run from the Code folder (point --dir at a results folder that includes the
probe tasks):

    python probe.py --dir results/2026-07-XX_limit100
"""

import os
import json
import argparse

import numpy as np

from analyze import load_runs, member_families, entropy, all_leaf_tasks, leaves_for
from auroc import gather, auroc


def signals_by_doc(runs, members, baseline, leaves):
    """
    Per doc-key, all four per-question signals computed over the family
    `members` (plus the baseline's own entropy). Same math as auroc.py.
    """
    dists, _ = gather(runs, members + [baseline], leaves)
    common = sorted(set.intersection(*(set(dists[m]) for m in members + [baseline])))
    out = {}
    for key in common:
        ps = np.stack([dists[m][key] for m in members])
        pbar = ps.mean(axis=0)
        total = entropy(pbar)
        aleatoric = float(np.mean([entropy(p) for p in ps]))
        out[key] = {
            "epistemic": total - aleatoric,
            "total": total,
            "aleatoric": aleatoric,
            "base_entropy": entropy(dists[baseline][key]),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results", help="results folder from run_eval.py")
    p.add_argument("--answerable", default="probe_hellaswag_answerable")
    p.add_argument("--unanswerable", default="probe_hellaswag_unanswerable")
    args = p.parse_args()

    manifest, runs = load_runs(args.dir)
    roles = manifest["members"]
    baseline = next(m for m, r in roles.items() if r == "baseline")
    families = {fam: [m for m in ms if m in runs]
                for fam, ms in member_families(roles).items()}
    families = {fam: ms for fam, ms in families.items() if len(ms) >= 2}

    leaves = all_leaf_tasks(runs)
    leaves_ans = leaves_for(args.answerable, leaves)
    leaves_un = leaves_for(args.unanswerable, leaves)
    if not leaves_ans or not leaves_un:
        raise SystemExit(f"probe tasks not found in {args.dir} "
                         f"(looked for {args.answerable} / {args.unanswerable})")

    out = {}
    for fam, members in families.items():
        sig_ans = signals_by_doc(runs, members, baseline, leaves_ans)
        sig_un = signals_by_doc(runs, members, baseline, leaves_un)

        print(f"\n=== {fam} (n={len(sig_ans)} answerable / {len(sig_un)} unanswerable) ===")
        print(f"  {'signal':14s} {'AUROC':>7s} {'mean ans.':>10s} {'mean unans.':>12s}")
        out[fam] = {}
        for name in ("epistemic", "total", "aleatoric", "base_entropy"):
            scores = ([d[name] for d in sig_ans.values()]
                      + [d[name] for d in sig_un.values()])
            labels = [0] * len(sig_ans) + [1] * len(sig_un)
            a = auroc(scores, labels)
            mean_ans = float(np.mean([d[name] for d in sig_ans.values()]))
            mean_un = float(np.mean([d[name] for d in sig_un.values()]))
            out[fam][name] = {"auroc": a, "mean_answerable": mean_ans,
                              "mean_unanswerable": mean_un}
            print(f"  {name:14s} {a:7.3f} {mean_ans:10.4f} {mean_un:12.4f}")

    path = os.path.join(args.dir, "probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
