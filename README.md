# Practical Work - Uncertainty (Stage 1)

Code for the Masterarbeit *"Epistemic Uncertainty for LLMs via MoE-Splitting"*.
The full plan and project notes live in the Obsidian vault under `Practical Work/`.

Right now this repo only contains a small **sanity check**: load a tiny test version of
Mixtral and run one forward pass, to confirm the setup works on any machine.

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
