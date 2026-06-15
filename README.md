# Practical Work - Uncertainty (Stage 1)

Code for the Masterarbeit *"Epistemic Uncertainty for LLMs via MoE-Splitting"*.
The full plan and project notes live in the Obsidian vault under `Practical Work/`.

So far this repo contains a **sanity check** (load a tiny Mixtral, run one forward pass)
and the **routing override** (`routing.py`) that turns one MoE model into ensemble
members by changing which experts the gate picks - no retraining, no copied weights.

## Setup

```bash
# 1. Create the conda environment (only needed once)
conda env create -f environment.yml

# 2. Activate it (every time you start working)
conda activate practical_work
```

If you add a new dependency later, add it to `environment.yml` and run
`conda env update -f environment.yml --prune` to keep the env in sync.

## Run the sanity check

```bash
python sanity_check.py
```

This prints the device used, the shape of the model output, the predicted next token,
and how many Mixture-of-Experts blocks the model has.

## Routing (ensemble members)

`routing.py` defines the ensemble **members** as routing policies applied to the gate:

- `topk_member(3)` - per token, the gate's 3 highest-ranked experts (the "Top-3" member)
- `rank_shift_member(start=1, k=2)` - per token, ranks 2-3 (mild shift, shares one expert with the baseline)
- `rank_shift_member(start=2, k=2)` - per token, ranks 3-4 (the "second-best route", disjoint from the top-2 baseline)
- `random_member(model, k=2, seed=0)` - k random experts per layer (control: does the trained router matter?)
- a plain list like `[[1], [3]]` - force explicit experts per layer (debugging)

Mixtral is top-2, so the baseline (ranks 1-2) is just the unmodified model. The members
above form a graded difficulty series: baseline -> Top-3 -> ranks 2-3 -> ranks 3-4 -> random.

Apply one with `set_member(model, member)` and undo it with `restore(model)`. In every
case the gate softmax is renormalized over only the chosen experts (so the weights sum
to 1). Verify all policies on the tiny model with:

```bash
python check_routing.py
```

The tiny model has random weights, so this checks **correctness of the routing only** -
real accuracy comes later, on the full Mixtral, via the benchmark harness.

## Benchmark harness

`benchmark.py` is a small likelihood-based multiple-choice scorer: for each member
it scores every answer choice by its total log-probability, predicts the highest, and
reports accuracy. Across the members it computes the uncertainty decomposition
(`total = epistemic + aleatoric`, with epistemic = mean KL of each member to the
ensemble mean). Run it with:

```bash
python benchmark.py
```

It uses a tiny hard-coded toy MCQ set so it runs offline. **On the tiny model every
number is meaningless** (random weights) - this only proves the pipeline runs end to
end before spending money on a GPU. To use a real benchmark later, replace
`toy_dataset()` with a loader returning the same format
(`{"question", "choices", "answer"}`); nothing else changes.

## Notebook

`sanity.ipynb` does the same thing step by step (it reuses the functions from
`sanity_check.py`, so there is no duplicated logic). Open it for small interactive tests:

```bash
jupyter notebook sanity.ipynb
```

## Notes

- For now we only use the **tiny** model (`hf-internal-testing/Mixtral-tiny`) so everything
  runs fast on a laptop. The real Mixtral-8x7B needs a big GPU and comes in a later step.
- To swap the model, change the `model_id` line in `config.yaml` - no code changes needed.
