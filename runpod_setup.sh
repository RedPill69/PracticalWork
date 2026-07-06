#!/usr/bin/env bash
# One-shot setup for a fresh Runpod GPU pod (any recent PyTorch CUDA template,
# one 48 GB card is enough). Clone the repo to /workspace and run this from the
# repo folder:
#
#     cd /workspace/Code && bash runpod_setup.sh
#
# /workspace is the persistent network volume: the ~88 GB model download
# survives a pod Stop/Start there (a Terminate deletes the volume).
set -e

# Keep the Hugging Face cache on the persistent volume (and in future shells).
export HF_HOME=/workspace/hf_cache
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# Known-good versions from the 2026-06-16 GPU bring-up. transformers is pinned
# to 4.x because 5.x fuses Mixtral's experts into one 3D parameter that
# bitsandbytes cannot 4-bit quantize (see environment.yml). The unpinned
# packages resolved to bitsandbytes 0.49.2 / lm-eval 0.4.12 back then.
pip install -U "torch==2.5.1" "transformers==4.57.6" "typing_extensions>=4.15" \
    bitsandbytes accelerate lm-eval sentencepiece datasets pyyaml

# Mixtral weights, safetensors only (~88 GB). The repo also ships a redundant
# consolidated.*.pt copy; pulling everything (~180 GB) exceeds the volume
# quota. HF_HUB_DISABLE_XET=1 avoids an hf-xet disk-quota/reconstruction error.
HF_HUB_DISABLE_XET=1 hf download mistralai/Mixtral-8x7B-v0.1 --exclude "consolidated*"

echo
echo "Setup done. Verify the load, then smoke-test the benchmark:"
echo "    python sanity_check.py --config server.yaml"
echo "    python run_eval.py --config server.yaml --limit 2"
echo "    python analyze.py  --config server.yaml"
