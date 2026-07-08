# Practical Work - Uncertainty (Stage 1)

Code for the Masterarbeit *"Epistemic Uncertainty for LLMs via MoE-Splitting"*.
The full plan and project notes live in the Obsidian vault under `Practical Work/`.

It turns one Mixture-of-Experts model (Mixtral) into an ensemble by changing which
experts the router picks - no retraining, no copied weights - and benchmarks the
resulting members to measure how much accuracy each loses and how much they disagree.

## Setup

```bash
# 1. Create the conda environment (only needed once)
conda env create -f environment.yml

# 2. Activate it (every time you start working)
conda activate practical_work
```

If you add a new dependency later, add it to `environment.yml` and run
`conda env update -f environment.yml --prune` to keep the env in sync.

## Configs: local vs server

Two configs, picked with `--config` (default `local.yaml`):

- `local.yaml` - tiny Mixtral on CPU, float32. For fast development and smoke tests.
  Numbers are meaningless (random weights); only the *code* is being checked.
- `server.yaml` - the real Mixtral-8x7B in **4-bit** on a CUDA GPU. The actual run.

Swap the model or run size by editing the config - no code changes.

## Sanity check

```bash
python sanity_check.py                 # local.yaml (tiny model)
```

Prints the device, output shape, predicted next token, and MoE block count.

## Routing (ensemble members)

`routing.py` defines routing policies that turn the model into ensemble **members**:

- `drop_expert_member(e)` - "jackknife": expert `e` is removed from every layer's pool,
  the router picks the best remaining experts. Affects a token only in layers where `e`
  was among its top choices, so the perturbation is dosed and member-specific.
- `gate_noise_member(sigma, seed)` - "MC-router": seeded Gaussian noise (sigma times the
  token's own logit spread) on the gate scores. Only near-tied routing decisions flip.
- `frozen_noise_member(sigma, seed)` - like gate-noise, but the noise vector is drawn once
  per layer and reused for every token and pass, so the member is a fixed, deterministic
  function (one routing perturbation = one posterior sample).
- `route_sample_member(temperature, seed)` - router sampling: the experts are drawn from
  the gate's own softmax at the given temperature (Gumbel-top-k) instead of argmax, then
  weighted by their original gate scores. T -> 0 recovers the baseline; T = 1 samples
  the router's actual distribution.
- `rank_select_member([0, 2])` - per token, exactly the experts at the given ranks of
  its own gate ranking (0-indexed); `[0]` keeps each token's best expert alone.
- `topk_member(3)` / `rank_shift_member(1, 2)` - rank windows (top-3, ranks 2-3).
- `random_member(model, 2, 0)` - random experts per layer (control: does the router matter?)
- a plain list like `[[1], [3]]` - force explicit experts per layer (debugging)

Mixtral is top-2, so the baseline (ranks 1-2) is the unmodified model. The member set the
benchmark actually runs lives in `run_eval.py:build_members`: currently the baseline plus
four extraction families, `drop_0..7`, `noise_<sigma>_<seed>` (sigma 0.5/1.0, seeds
0/1), `frozen_1.0_<seed>` (seeds 0/1), and `sample_1.0_<seed>` (seeds 0/1). Earlier scans
showed each token's top-ranked
expert carries the model (rank
windows without it collapse to chance; pairing it with any partner gives near-clones),
so these mechanisms perturb the top choice *selectively* instead. Member roles use
`principled/<family>` so the analysis never mixes two mechanisms into one ensemble;
earlier member sets (rank windows, rank-anchored pairs) are in the git history and their
results in the dated results folders. Apply a member with `set_member(model, member)`,
undo with `restore(model)`; the gate softmax is renormalized over only the chosen
experts. Verify all policies on the tiny model:

```bash
python check_routing.py
```

## Chat (manual inspection)

Talk to the model with one member active, and switch members live to feel the difference:

```bash
python chat.py                         # tiny model -> gibberish, but switching works
python chat.py --config server.yaml    # real Mixtral
```

Inside: `/member top2|top3|rank23|rank34|random|fixed <experts>`, `/quit`.

## Benchmark (two steps)

The benchmark is a **hybrid**: the standard lm-evaluation-harness measures accuracy
(correct task protocols, comparable numbers), and our own `analyze.py` computes the
uncertainty decomposition from the logged per-example log-likelihoods.

The task list in both configs also includes the **answerable/unanswerable probe pair**
(`tasks/probe_hellaswag_*.yaml`, always run 0-shot): the same HellaSwag contexts, but in
the unanswerable twin the true ending is swapped for another document's ending, so no
offered choice is correct. `probe.py` then measures whether the uncertainty signals can
separate "cannot know" from "merely hard" - the discrimination that plain entropy cannot
do even in principle. The probe pair stays out of the pooled `overall` numbers.

```bash
# 1. Run the accuracy eval once per member, saving raw results to results/
python run_eval.py                     # local.yaml: tiny model, tiny limit
python run_eval.py --config server.yaml

# 2. Turn those results into the answers (accuracy + loss + uncertainty + CIs)
python analyze.py
python analyze.py --config server.yaml
```

`run_eval.py` applies each member's routing, runs lm-eval with `log_samples`, and writes
one file per member to `results/`. `analyze.py` reports, per task and pooled: accuracy
per member, **loss vs the Top-2 baseline**, the **router-vs-random** comparison, the
**total/aleatoric/epistemic** decomposition over the principled members, and bootstrap
95% confidence intervals. It needs no GPU and no model - so the Stage-2 questions can be
answered later from the same saved files.

Each member has a `role` in the manifest: `baseline` (reference for the loss),
`principled/<family>` (forms the ensemble for the uncertainty math, grouped per
extraction family), `anchor` (shown in the accuracy table, kept out of the ensemble),
`control` (the random baseline). Files in
`results/` from members not in the current manifest are reported and ignored - archive
each finished run into a dated subfolder (e.g. `results/2026-06-16_limit100/`) so runs
never mix.

## Post-hoc analyses (no GPU)

These scripts read the same saved eval files; all take `--dir <results folder>`:

```bash
python overlap.py          --dir results/2026-06-16_limit100
python compare_official.py --dir results/2026-06-16_limit100
python auroc.py            --dir results/2026-07-06_limit100
python probe.py            --dir results   # needs a run that included the probe tasks
```

`overlap.py` asks whether members fail on the SAME or on DIFFERENT questions: the
members-correct-per-doc histogram, pairwise agreement and error-set Jaccard vs the
value expected under independent errors, and how many baseline errors each member
rescues (vs breaks). `compare_official.py` puts our baseline next to the officially
reported Mixtral-8x7B numbers (paper + HF leaderboard, sources in its header) and
explains the acc vs acc_norm metric difference. `auroc.py` tests whether the
uncertainty signals actually PREDICT errors (AUROC + selective prediction, per
extraction family), judged against the baseline's own entropy - the signal that needs
no ensemble and that any ensemble signal must beat to be useful.

## Running the real model on a GPU (Runpod)

One 48 GB card (e.g. RTX 6000 Ada) is enough; the model is not gated. On a fresh pod
(recent PyTorch CUDA template, repo cloned to `/workspace`):

```bash
cd /workspace/Code
bash runpod_setup.sh     # pinned packages + safetensors-only model download (~88GB)

# Smoke run first: confirms the 4-bit load + routing on the real model.
python sanity_check.py --config server.yaml
python run_eval.py --config server.yaml --limit 2
python analyze.py  --config server.yaml

# Then the real run: edit eval.limit in server.yaml (or drop --limit) and re-run.
python run_eval.py --config server.yaml
python analyze.py  --config server.yaml

# Afterwards: archive the run into a dated folder and copy it off the pod.
mv results results_tmp && mkdir results && mv results_tmp results/$(date +%F)_limitN
```

`runpod_setup.sh` documents the environment that worked (torch 2.5.1, transformers
4.57.6, safetensors-only download with `--exclude "consolidated*"`, `HF_HOME` on the
persistent `/workspace` volume). The 4-bit path (`quantization: 4bit`) uses bitsandbytes
nf4 pinned to GPU 0 and needs ~24 GB. **It can only be tested on a CUDA GPU** - locally
we verify everything else (routing, the lm-eval integration, the analysis) on the tiny
model. Note `eval.limit` is **per (sub)task**, so for MMLU it is per subject x 57 - set
it deliberately. `analyze.py` aggregates MMLU's subtasks into one `mmlu` line.
Use Runpod's Stop (not Terminate) to pause the pod: Stop keeps the `/workspace` volume
with the downloaded model, Terminate deletes it.

## Notebook

`sanity.ipynb` walks through the sanity check step by step, reusing the functions in
`sanity_check.py`:

```bash
jupyter notebook sanity.ipynb
```

## Notes

- Locally we use the **tiny** model so everything runs fast; its numbers are meaningless.
  The real Mixtral-8x7B runs on a GPU via `server.yaml`.
- Out of scope for Stage 1 (this code): free-form generation + semantic entropy. For
  multiple choice the answer options already are the meaning clusters, so no clustering
  is needed; semantic entropy is the Stage-2 free-form experiment.
