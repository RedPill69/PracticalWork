"""
sanity_check.py

A first, minimal check that our setup works. It does four things:
  1. read settings from config.yaml
  2. fix the random seed (so runs are reproducible)
  3. download / load the small Mixtral test model
  4. run ONE forward pass on a prompt and print a few facts about the model

Run it from the Code folder with:

    python sanity_check.py

The notebook (sanity.ipynb) imports these same functions, so the logic lives
in exactly one place.
"""

import random

import yaml
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path="config.yaml"):
    """Read the YAML settings file into a plain Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """
    Make a run reproducible by seeding every random number generator we use.
    Without this, results could differ slightly between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Use the GPU ('cuda') if one is available, otherwise fall back to the CPU.
    The tiny model runs fine on a laptop CPU, so this works everywhere.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg):
    """
    Download (first time) or load from cache (afterwards) the tokenizer and model.

    - The tokenizer turns text into the numbers the model understands.
    - The model is loaded in float32 (standard precision) and moved onto the device.

    Returns: tokenizer, model, device (a string like "cpu" or "cuda").
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()  # evaluation mode: we only run the model, we do not train it

    return tokenizer, model, device


def count_moe_blocks(model):
    """
    Count how many 'MixtralSparseMoeBlock' modules the model contains.

    This block holds the router + the experts - it is the exact place we will
    modify in a later session to build our ensemble. Counting it now confirms
    the model really has the Mixture-of-Experts structure we expect.
    """
    count = 0
    for module in model.modules():
        if type(module).__name__ == "MixtralSparseMoeBlock":
            count += 1
    return count


def run_sanity(cfg):
    """Tie everything together: seed, load, one forward pass, then print a report."""
    set_seed(cfg["seed"])

    tokenizer, model, device = load_model(cfg)

    # Turn the prompt text into model input numbers, placed on the model's device.
    inputs = tokenizer(cfg["prompt"], return_tensors="pt").to(device)

    # Run the model WITHOUT tracking gradients (faster, less memory - we are not training).
    with torch.no_grad():
        outputs = model(**inputs)

    # 'logits' are the model's raw scores for every possible next token.
    # Shape = (batch_size, number_of_input_tokens, vocabulary_size).
    logits = outputs.logits

    # The predicted next token = the highest-scoring token at the LAST input position.
    next_token_id = int(logits[0, -1].argmax())
    next_token = tokenizer.decode(next_token_id)

    # Print a small report so we can see at a glance that everything worked.
    print("=== Sanity check ===")
    print(f"Model:            {cfg['model_id']}")
    print(f"Device:           {device}")
    print(f"Prompt:           {cfg['prompt']!r}")
    print(f"Logits shape:     {tuple(logits.shape)}")
    print(f"Predicted next:   {next_token!r}")
    print(f"MoE blocks found: {count_moe_blocks(model)}")
    print("====================")

    return logits


if __name__ == "__main__":
    run_sanity(load_config())
