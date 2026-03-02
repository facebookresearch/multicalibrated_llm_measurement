#!/usr/bin/env bash
# Setup script for running CAP LLM inference on a Meta GPU devserver.
#
# Prerequisites:
#   - A100 80G devserver reserved
#   - Data files transferred from MacBook (see instructions below)
#
# Usage:
#   1. Reserve a devserver (1 x A100 80G GPU)
#
#   2. Clone the repo on the devserver:
#        git clone <repo-url> ~/mc_measurement
#
#   3. Transfer data files from your MacBook to the devserver:
#        DEVSERVER=your-devserver-hostname
#        scp cap_analysis/data/*_final.csv \
#            cap_analysis/data/feasibility_sample_70b.csv \
#            cap_analysis/data/full_sample.csv \
#            $DEVSERVER:~/mc_measurement/cap_analysis/data/
#
#   4. SSH into the devserver and run this script:
#        ssh $DEVSERVER
#        cd ~/mc_measurement
#        bash cap_analysis/setup_devserver.sh
#
#   5. Run the feasibility test (~10-30 min on A100):
#        cd ~/mc_measurement
#        source ~/cap_env/bin/activate
#        python cap_analysis/llm_inference_cuda.py \
#          --input cap_analysis/data/feasibility_sample_70b.csv \
#          --output cap_analysis/data/inference_output/llama-70b/feasibility_scores.csv \
#          --model meta-llama/Llama-3.1-70B-Instruct \
#          --no-resume
#
#   6. If feasibility looks good, run the full 105K sample (~1-3 hours on A100):
#        python cap_analysis/llm_inference_cuda.py \
#          --input cap_analysis/data/full_sample.csv \
#          --output cap_analysis/data/inference_output/llama-70b/full_scores.csv \
#          --model meta-llama/Llama-3.1-70B-Instruct \
#          --no-resume
#
#   7. Copy results back to MacBook:
#        # From your MacBook:
#        scp -r $DEVSERVER:~/mc_measurement/cap_analysis/data/inference_output/llama-70b/ \
#            cap_analysis/data/inference_output/llama-70b/

set -euo pipefail

echo "=== CAP LLM Inference: DevServer Setup ==="

# Proxy for external package downloads
export HTTPS_PROXY=http://fwdproxy:8080
export HTTP_PROXY=http://fwdproxy:8080

# Create virtual environment
echo "Creating Python environment..."
python3 -m venv ~/cap_env
source ~/cap_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch transformers accelerate bitsandbytes tqdm

# Verify GPU access
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Create output directories
mkdir -p cap_analysis/data/inference_output/llama-70b

# Verify data files exist
echo ""
echo "=== Data Check ==="
for f in cap_analysis/data/feasibility_sample_70b.csv cap_analysis/data/full_sample.csv; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  OK: $f ($lines lines)"
    else
        echo "  MISSING: $f — transfer from MacBook (see instructions above)"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "Run the feasibility test with:"
echo "  source ~/cap_env/bin/activate"
echo "  python cap_analysis/llm_inference_cuda.py \\"
echo "    --input cap_analysis/data/feasibility_sample_70b.csv \\"
echo "    --output cap_analysis/data/inference_output/llama-70b/feasibility_scores.csv \\"
echo "    --model meta-llama/Llama-3.1-70B-Instruct \\"
echo "    --no-resume"
