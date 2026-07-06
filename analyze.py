"""
analyze.py

Stage-1 benchmark, part 2 of 2: turn the saved per-member eval results
(from run_eval.py) into the actual answers. No model, no GPU - pure
post-processing of the JSON files in cfg["eval"]["output_dir"].

It reports, per task and pooled overall:
  - accuracy per member (from the logged per-doc results),
  - accuracy LOSS vs the Top-2 baseline,
  - the router-vs-random comparison (do principled members beat random?),
  - the uncertainty decomposition total = aleatoric + epistemic over the
    principled members (epistemic = mean KL of each member to the ensemble mean),
  - bootstrap 95% confidence intervals on accuracy (resampled from the per-doc
    correctness, so error bars come from a single run).

Run it from the Code folder with:

    python analyze.py                       # reads results/ (local.yaml)
    python analyze.py --config server.yaml
"""

import os
import json
import glob

import numpy as np

from sanity_check import load_config, config_arg


# --- loading ---------------------------------------------------------------

def load_runs(out_dir):
    """Load the manifest and every member's eval file from out_dir."""
    with open(os.path.join(out_dir, "members.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    runs = {}  # member -> {"results": ..., "samples": ...}
    for path in glob.glob(os.path.join(out_dir, "eval_*.json")):
        name = os.path.basename(path)[len("eval_"):-len(".json")]
        with open(path, encoding="utf-8") as f:
            runs[name] = json.load(f)
    return manifest, runs


def choice_loglikelihoods(sample):
    """Per-choice log-likelihoods from one lm-eval sample record."""
    lls = []
    for r in sample["filtered_resps"]:
        lls.append(float(r[0]) if isinstance(r, (list, tuple)) else float(r))
    return np.array(lls)


def per_doc(samples):
    """
    From a list of lm-eval samples, return:
      dists: doc_id -> softmax distribution over the choices,
      correct: doc_id -> 0/1 (argmax of the raw log-likelihoods == gold).
    """
    dists, correct = {}, {}
    for s in samples:
        lls = choice_loglikelihoods(s)
        gold = int(s["target"])
        p = np.exp(lls - lls.max())
        dists[s["doc_id"]] = p / p.sum()
        correct[s["doc_id"]] = int(int(lls.argmax()) == gold)
    return dists, correct


# --- metrics ---------------------------------------------------------------

def entropy(p):
    """Shannon entropy in nats, safe against zeros."""
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log(p)).sum())


def decompose(doc_dists):
    """
    doc_dists: list over docs of {member -> distribution}.
    Returns mean (total, aleatoric, epistemic) in nats over the docs.
      total      = H(mean predictive)
      aleatoric  = mean over members of H(member predictive)
      epistemic  = total - aleatoric  = mean KL(member || mean)
    """
    totals, aleatorics = [], []
    for per_member in doc_dists:
        ps = np.stack(list(per_member.values()))   # (M, C)
        pbar = ps.mean(axis=0)
        totals.append(entropy(pbar))
        aleatorics.append(np.mean([entropy(p) for p in ps]))
    total, aleatoric = float(np.mean(totals)), float(np.mean(aleatorics))
    return total, aleatoric, total - aleatoric


def bootstrap_ci(correct_flags, iters=2000, seed=0):
    """95% CI on the mean of 0/1 correctness via resampling with replacement."""
    rng = np.random.default_rng(seed)
    arr = np.array(correct_flags, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    means = arr[rng.integers(0, len(arr), size=(iters, len(arr)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- main ------------------------------------------------------------------
#
# lm-eval reports a "group" task like MMLU as 57 separate leaf tasks
# (mmlu_abstract_algebra, ...). We aggregate the leaves back into the task the
# manifest asked for, so MMLU shows as one line, not 57. A leaf belongs to a
# requested task `g` if it equals `g` or starts with `g + "_"`.


def all_leaf_tasks(runs):
    """Every leaf task name that appears in any member's saved samples."""
    leaves = set()
    for run in runs.values():
        leaves.update(run["samples"].keys())
    return sorted(leaves)


def leaves_for(group, leaves):
    """The leaf tasks belonging to a requested task/group."""
    return [t for t in leaves if t == group or t.startswith(group + "_")]


def member_correct(run, leaves):
    """0/1 correctness per doc for one member, pooled over the given leaf tasks."""
    flags = []
    for t in leaves:
        if t in run["samples"]:
            _, c = per_doc(run["samples"][t])
            flags.extend(c.values())
    return flags


def member_dists(run, leaves):
    """doc -> distribution for one member, pooled over the given leaf tasks."""
    out = {}
    for t in leaves:
        if t in run["samples"]:
            d, _ = per_doc(run["samples"][t])
            out.update({f"{t}:{k}": v for k, v in d.items()})
    return out


def main():
    cfg = load_config(config_arg())
    out_dir = cfg["eval"]["output_dir"]
    manifest, runs = load_runs(out_dir)
    roles = manifest["members"]

    baseline = next((m for m, r in roles.items() if r == "baseline"), None)
    principled = [m for m, r in roles.items() if r == "principled"]
    controls = [m for m, r in roles.items() if r == "control"]

    # Show members in manifest order (baseline first), not filesystem order.
    # Eval files whose member is NOT in the manifest are leftovers from an
    # earlier run with a different member set - report and ignore them, so two
    # runs never silently mix in one table.
    members = [m for m in roles if m in runs]
    stray = sorted(m for m in runs if m not in roles)
    if stray:
        print(f"(ignoring stale eval files not in members.json: {', '.join(stray)})")

    # Iterate the tasks the manifest asked for (MMLU's subtasks aggregated into
    # one "mmlu"), plus a pooled "overall" when there is more than one task.
    leaves = all_leaf_tasks(runs)
    groups = [(g, leaves_for(g, leaves)) for g in manifest["tasks"]]
    groups = [(g, lv) for g, lv in groups if lv]   # keep only tasks that ran
    if len(groups) > 1:
        groups.append(("overall", leaves))

    summary = {}
    for task, task_leaves in groups:
        print(f"\n=== {task} ===")

        # Accuracy + bootstrap CI per member, and loss vs baseline.
        acc, ci = {}, {}
        for name in members:
            flags = member_correct(runs[name], task_leaves)
            acc[name] = float(np.mean(flags)) if flags else float("nan")
            ci[name] = bootstrap_ci(flags)

        base_acc = acc.get(baseline, float("nan"))
        print(f"  {'member':14s} {'acc':>7s}  {'95% CI':>16s}   loss vs base")
        for name in members:
            lo, hi = ci[name]
            loss = base_acc - acc[name]
            tag = {baseline: "(baseline)"}.get(name, "")
            print(f"  {name:14s} {acc[name]:7.3f}  [{lo:6.3f}, {hi:6.3f}]   "
                  f"{loss:+.3f} {tag}")

        # Router vs random: best principled member vs the control.
        if principled and controls:
            best_principled = max(principled, key=lambda m: acc[m])
            for ctrl in controls:
                gap = acc[best_principled] - acc[ctrl]
                print(f"  router vs random: {best_principled} - {ctrl} = {gap:+.3f}")

        # Uncertainty decomposition over the principled members.
        unc = None
        if len(principled) >= 2:
            dists_by_member = {m: member_dists(runs[m], task_leaves) for m in principled}
            common = set.intersection(*(set(d) for d in dists_by_member.values()))
            doc_dists = [{m: dists_by_member[m][doc] for m in principled}
                         for doc in sorted(common)]
            if doc_dists:
                total, aleatoric, epistemic = decompose(doc_dists)
                unc = {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}
                print(f"  uncertainty (nats): total={total:.4f}  "
                      f"aleatoric={aleatoric:.4f}  epistemic={epistemic:.4f}")

        summary[task] = {"acc": acc, "ci": ci, "uncertainty": unc}

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "summary": summary}, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
