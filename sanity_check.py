"""
sanity_check.py

A first, minimal check that our setup works, plus the shared model-loading
helpers the other scripts reuse. It does four things:
  1. read settings from a config file (local.yaml by default)
  2. fix the random seed (so runs are reproducible)
  3. load the model (tiny on CPU locally, or real Mixtral in 4-bit on a GPU)
  4. run ONE forward pass on a prompt and print a few facts about the model

Run it from the Code folder with:

    python sanity_check.py                  # uses local.yaml (tiny model)
    python sanity_check.py --config server.yaml

The notebook (sanity.ipynb) and the other scripts import these same functions,
so the loading logic lives in exactly one place.
"""

import random
import argparse

import yaml
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path="local.yaml"):
    """Read the YAML settings file into a plain Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_arg(default="local.yaml"):
    """Parse a single --config argument (shared by all scripts)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default, help="path to a YAML config")
    return parser.parse_args().config


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


def get_device(requested="auto"):
    """
    Resolve the device. "auto" picks the GPU ('cuda') if available, else CPU.
    The tiny model runs fine on a laptop CPU, so "auto" works everywhere.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_model(cfg):
    """
    Load the tokenizer and model according to cfg["model"].

    Two paths, chosen by `quantization`:
      - "4bit": real Mixtral in 4-bit (nf4 + double quant) spread across the GPU
        with device_map="auto". This is what fits the ~90GB model on one A100.
        Needs a CUDA GPU (bitsandbytes is CUDA-only).
      - "none": plain load in the given dtype, moved onto the device. Used for the
        tiny model on CPU.

    Returns: tokenizer, model, device (a string like "cpu" or "cuda").
    """
    mcfg = cfg["model"]
    device = get_device(mcfg.get("device", "auto"))
    dtype = getattr(torch, mcfg.get("dtype", "float32"))

    tokenizer = AutoTokenizer.from_pretrained(mcfg["model_id"])

    quant = mcfg.get("quantization", "none")
    if quant == "4bit":
        if device != "cuda":
            raise RuntimeError(
                "quantization: 4bit needs a CUDA GPU (bitsandbytes is CUDA-only). "
                "Use quantization: none for local/CPU runs."
            )
        # Imported here so local CPU runs do not need bitsandbytes installed.
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,   # bfloat16 on the server
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            mcfg["model_id"],
            quantization_config=bnb,
            device_map="auto",   # places the sharded model across the GPU
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(mcfg["model_id"], dtype=dtype)
        model.to(device)

    model.eval()  # evaluation mode: we only run the model, we do not train it
    return tokenizer, model, device


def count_moe_blocks(model):
    """
    Count how many 'MixtralSparseMoeBlock' modules the model contains.

    This block holds the router + the experts - it is the exact place we modify
    to build our ensemble. Counting it confirms the model really has the
    Mixture-of-Experts structure we expect.
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
    inputs = tokenizer(cfg["prompt"], return_tensors="pt").to(model.device)

    # Run the model WITHOUT tracking gradients (faster, less memory - not training).
    with torch.no_grad():
        outputs = model(**inputs)

    # 'logits' are the model's raw scores for every possible next token.
    # Shape = (batch_size, number_of_input_tokens, vocabulary_size).
    logits = outputs.logits

    # The predicted next token = the highest-scoring token at the LAST position.
    next_token_id = int(logits[0, -1].argmax())
    next_token = tokenizer.decode(next_token_id)

    print("=== Sanity check ===")
    print(f"Model:            {cfg['model']['model_id']}")
    print(f"Device:           {device}")
    print(f"Prompt:           {cfg['prompt']!r}")
    print(f"Logits shape:     {tuple(logits.shape)}")
    print(f"Predicted next:   {next_token!r}")
    print(f"MoE blocks found: {count_moe_blocks(model)}")
    print("====================")

    return logits


if __name__ == "__main__":
    run_sanity(load_config(config_arg()))
