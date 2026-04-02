#!/usr/bin/env bash
# Setup script for running CAP LLM inference on a Meta GPU devserver.
#
# Two inference pipelines:
#   - llm_inference_cuda.py: logprob-based P(Yes)/(P(Yes)+P(No)) scoring
#   - llm_inference_cuda_verbalized.py: verbalized confidence (0-100 scale)
#
# Usage:
#   1. Reserve a devserver (1 x A100 80G GPU)
#   2. Clone/pull the repo:  ssh $DEVSERVER && cd ~/mc_measurement && git pull
#   3. Run this script:      bash cap_analysis/setup_devserver.sh
#   4. Follow the printed commands to run inference
#   5. Copy results back:    scp -r $DEVSERVER:~/mc_measurement/cap_analysis/data/inference_output/ \
#                                 cap_analysis/data/inference_output/

set -euo pipefail

# Always run from the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== CAP LLM Inference: DevServer Setup ==="
echo "Working directory: $(pwd)"

# Proxy for external access (packages + CAP data downloads)
export HTTPS_PROXY=http://fwdproxy:8080
export HTTP_PROXY=http://fwdproxy:8080

# Create virtual environment
echo "Creating Python environment..."
python3 -m venv ~/cap_env
source ~/cap_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch transformers accelerate bitsandbytes tqdm pandas requests

# Verify GPU access
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: No CUDA GPU detected!')
"

# Download and prepare data
echo ""
echo "=== Downloading and preparing CAP data ==="
python3 cap_analysis/prepare_data.py --proxy http://fwdproxy:8080

# Create output directories
mkdir -p cap_analysis/data/inference_output/llama-70b
mkdir -p cap_analysis/data/inference_output/llama-70b-verbalized

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Activate the environment:"
echo "  source ~/cap_env/bin/activate"
echo ""
echo "--- Logprob scoring (original) ---"
echo ""
echo "Feasibility test (~10-30 min):"
echo "  python cap_analysis/llm_inference_cuda.py \\"
echo "    --input cap_analysis/data/feasibility_sample_70b.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b/feasibility_scores.csv \\"
echo "    --model meta-llama/Llama-3.3-70B-Instruct \\"
echo "    --no-resume"
echo ""
echo "Full run (~1-3 hours):"
echo "  python cap_analysis/llm_inference_cuda.py \\"
echo "    --input cap_analysis/data/full_sample.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b/full_scores.csv \\"
echo "    --model meta-llama/Llama-3.3-70B-Instruct \\"
echo "    --no-resume"
echo ""
echo "--- Verbalized confidence scoring (greedy, single pass) ---"
echo ""
echo "Feasibility test (~10-30 min):"
echo "  python cap_analysis/llm_inference_cuda_verbalized.py \\"
echo "    --input cap_analysis/data/feasibility_sample_70b.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b-verbalized/feasibility_scores.csv \\"
echo "    --model meta-llama/Llama-3.3-70B-Instruct \\"
echo "    --num-samples 1 --no-resume"
echo ""
echo "--- Verbalized confidence + temperature sampling (K=10, 2x GPU) ---"
echo ""
echo "Feasibility test (2 shards, ~15 min each):"
echo "  CUDA_VISIBLE_DEVICES=0 python cap_analysis/llm_inference_cuda_verbalized.py \\"
echo "    --input cap_analysis/data/feasibility_sample_70b.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b-verbalized/feasibility_sampled_shard0.csv \\"
echo "    --model meta-llama/Llama-3.3-70B-Instruct \\"
echo "    --num-samples 10 --temperature 0.7 --shard 0/2 --no-resume &"
echo ""
echo "  CUDA_VISIBLE_DEVICES=1 python cap_analysis/llm_inference_cuda_verbalized.py \\"
echo "    --input cap_analysis/data/feasibility_sample_70b.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b-verbalized/feasibility_sampled_shard1.csv \\"
echo "    --model meta-llama/Llama-3.3-70B-Instruct \\"
echo "    --num-samples 10 --temperature 0.7 --shard 1/2 --no-resume &"
echo ""
echo "  wait  # wait for both shards to finish"
echo ""
echo "Full run (2 shards, ~10-15h each — run overnight):"
echo "  CUDA_VISIBLE_DEVICES=0 python cap_analysis/llm_inference_cuda_verbalized.py \\"
echo "    --input cap_analysis/data/full_sample.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b-verbalized/full_sampled_shard0.csv \\"
echo "    --model meta-llama/Llama-3.3-70B-Instruct \\"
echo "    --num-samples 10 --temperature 0.7 --shard 0/2 --no-resume &"
echo ""
echo "  CUDA_VISIBLE_DEVICES=1 python cap_analysis/llm_inference_cuda_verbalized.py \\"
echo "    --input cap_analysis/data/full_sample.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b-verbalized/full_sampled_shard1.csv \\"
echo "    --model meta-llama/Llama-3.3-70B-Instruct \\"
echo "    --num-samples 10 --temperature 0.7 --shard 1/2 --no-resume &"
echo ""
echo "  wait"
echo ""
echo "Merge shards:"
echo "  head -1 cap_analysis/data/inference_output/llama-70b-verbalized/full_sampled_shard0.csv > \\"
echo "    cap_analysis/data/inference_output/llama-70b-verbalized/full_sampled_scores.csv"
echo "  tail -n +2 -q cap_analysis/data/inference_output/llama-70b-verbalized/full_sampled_shard*.csv >> \\"
echo "    cap_analysis/data/inference_output/llama-70b-verbalized/full_sampled_scores.csv"
echo ""
echo "--- Copy results back to MacBook ---"
echo "  scp -r \$DEVSERVER:~/mc_measurement/cap_analysis/data/inference_output/ \\"
echo "      cap_analysis/data/inference_output/"
