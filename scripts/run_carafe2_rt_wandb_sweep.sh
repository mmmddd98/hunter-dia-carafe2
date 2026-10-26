#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: $0 <hours> <prepared-carafe-output> [sweep-output] [python] [wandb-config]" >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
default_root=${CARAFE2_FINETUNE_OUTPUT_ROOT:-"$root/finetune_output"}
sweep_output=${3:-"$default_root/carafe2_rt_wandb_sweep_$(date +%Y%m%d_%H%M%S)"}
python_bin=${4:-${CARAFE2_PYTHON:-python3}}
wandb_config=${5:-"$root/configs/wandb.json"}
input_dir=$2
if [[ -d "$input_dir/carafe2_input" ]]; then
  input_dir="$input_dir/carafe2_input"
fi

exec "$python_bin" "$root/run_carafe2_wandb_sweep.py" \
  --hours "$1" --in-dir "$input_dir" --output-dir "$sweep_output" --python "$python_bin" \
  --tf-type rt --sweep-config "$root/configs/carafe2_rt_finetune_sweep.json" \
  --wandb-config "$wandb_config"
