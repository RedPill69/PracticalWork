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

`routing.py` defines the ensemble **members** as routing policies applied to the gate:

- `topk_member(3)` - per token, the gate's 3 highest-ranked experts ("Top-3")
- `rank_shift_member(1, 2)` - ranks 2-3 (mild shift, shares one expert with baseline)
- `rank_shift_member(2, 2)` - ranks 3-4 (the "second-best route", disjoint from baseline)
- `random_member(model, 2, 0)` - random experts per layer (control: does the router matter?)
- a plain list like `[[1], [3]]` - force explicit experts per layer (debugging)

Mixtral is top-2, so the baseline (ranks 1-2) is the unmodified model. The members form
a graded series: baseline -> Top-3 -> ranks 2-3 -> ranks 3-4 -> random. Apply one with
`set_member(model, member)`, undo with `restore(model)`; the gate softmax is renormalized
over only the chosen experts. Verify all policies on the tiny model:

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

## Running the real model on a GPU

```bash
huggingface-cli download mistralai/Mixtral-8x7B-v0.1      # download once (~90GB)
python run_eval.py --config server.yaml                   # 4-bit load + per-member sweep
python analyze.py  --config server.yaml                   # final Stage-1 answers
```

The 4-bit path (`quantization: 4bit`) uses bitsandbytes nf4 + `device_map="auto"` and
fits on one A100 80GB. **It can only be tested on a CUDA GPU** - locally we verify
everything else (routing, the lm-eval integration, the analysis) on the tiny model.
Start with a small `eval.limit` to time one pass, then scale up.

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
