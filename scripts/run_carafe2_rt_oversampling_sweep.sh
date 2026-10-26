#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 <hours> <prepared-carafe-output> [sweep-output] [seeds] [python] [wandb-config]" >&2
  echo "Example seeds: 2024,2025,2026,2027,2028,2029,2030,2031,2032,2033" >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
default_root=${CARAFE2_FINETUNE_OUTPUT_ROOT:-"$root/finetune_output"}
sweep_output=${3:-"$default_root/carafe2_rt_oversampling_sweep_$(date +%Y%m%d_%H%M%S)"}
seeds=${4:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}
python_bin=${5:-${CARAFE2_PYTHON:-python3}}
wandb_config=${6:-"$root/configs/wandb.json"}
input_dir=$2
if [[ -d "$input_dir/carafe2_input" ]]; then
  input_dir="$input_dir/carafe2_input"
fi

exec "$python_bin" "$root/run_carafe2_wandb_sweep.py" \
  --hours "$1" --in-dir "$input_dir" --output-dir "$sweep_output" --python "$python_bin" \
  --tf-type rt --seeds "$seeds" \
  --sweep-config "$root/configs/carafe2_rt_oversampling_sweep.json" \
  --wandb-config "$wandb_config"
