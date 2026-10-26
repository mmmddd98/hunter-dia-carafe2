#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 <prepared-carafe-output> <gpu> [output-dir] [seeds] [python] [wandb-config]" >&2
  exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${CARAFE2_FINETUNE_OUTPUT_ROOT:-"$root/finetune_output"}
output=${3:-"$output_root/carafe2_rt_controlled_weights_$(date +%Y%m%d_%H%M%S)"}
seeds=${4:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}
python_bin=${5:-${CARAFE2_PYTHON:-python3}}
wandb=${6:-$root/configs/wandb.json}
input=$1
if [[ -d "$input/carafe2_input" ]]; then input="$input/carafe2_input"; fi
exec env CUDA_VISIBLE_DEVICES="$2" "$python_bin" "$root/run_carafe2_wandb_sweep.py" \
  --hours 12 --in-dir "$input" --output-dir "$output" --python "$python_bin" \
  --tf-type rt --device gpu --seeds "$seeds" \
  --sweep-config "$root/configs/carafe2_rt_controlled_weights_sweep.json" \
  --configuration-map "$root/configs/carafe2_rt_controlled_weights_map.json" \
  --wandb-config "$wandb"
