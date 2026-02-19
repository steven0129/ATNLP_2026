#!/usr/bin/env bash
set -euo pipefail

DATASET=""
METHOD=""
MODEL=""
DATACONFIG=""

for TOP_P in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
  OUTPUT_FOLDER="final_results-top_p=${TOP_P}"
  conda run -n atnlp python -u main.py \
    --dataset "mmlu-redux-college_mathematics" \
    --method "comat" \
    --model "gpt" \
    --dataconfig "normal" \
    --top-p "${TOP_P}" \
    --output-folder "${OUTPUT_FOLDER}"
done
