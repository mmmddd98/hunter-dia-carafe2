#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: $0 <hours> <prepared-carafe-output> [sweep-output] [python] [wandb-config]" >&2
  exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
default_root=${CARAFE2_FINETUNE_OUTPUT_ROOT:-"$root/finetune_output"}
sweep_output=${3:-"$default_root/carafe2_wandb_sweep_$(date +%Y%m%d_%H%M%S)"}
python_bin=${4:-${CARAFE2_PYTHON:-python3}}
wandb_config=${5:-"$root/configs/wandb.json"}
exec "$python_bin" "$root/run_carafe2_wandb_sweep.py" \
  --hours "$1" --in-dir "$2" --output-dir "$sweep_output" --python "$python_bin" --wandb-config "$wandb_config"
