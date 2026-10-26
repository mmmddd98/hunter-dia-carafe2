#!/usr/bin/env bash
set -euo pipefail
# Bash may otherwise continue reading a modified script after waiting for hours
# on the shared GPU lock. Buffer the complete source into argv once at startup.
script_path=${ORGANISM_SCALING_SCRIPT_PATH:-$(readlink -f -- "${BASH_SOURCE[0]}")}
if [[ ${ORGANISM_SCALING_SCRIPT_BUFFERED:-0} != 1 ]]; then
  script_text=$(<"$script_path")
  exec env ORGANISM_SCALING_SCRIPT_BUFFERED=1 \
    ORGANISM_SCALING_SCRIPT_PATH="$script_path" \
    bash -c "$script_text" "$script_path" "$@"
fi
root=$(cd -- "$(dirname -- "$script_path")/.." && pwd)
data_collect_dir=${CARAFE_DATA_COLLECT_DIR:-}
[[ -n "$data_collect_dir" ]] || {
  echo "Set CARAFE_DATA_COLLECT_DIR to the MA data_collect directory." >&2
  exit 1
}
study_root=${1:-"${CARAFE2_FINETUNE_OUTPUT_ROOT:-$root/finetune_output}/organism_scaling_v1"}
prep="$data_collect_dir/prepare_organism_scaling_study.py"
[[ -f "$prep" ]] || {
  echo "Missing MA study preparation script: $prep" >&2
  exit 1
}
prepared="$study_root/prepared"
rt_target=${RT_TARGET:-global_60min}
has_column() {
  local path=$1 column=$2
  awk -F '\t' -v wanted="$column" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == wanted) found = 1; exit !found }' "$path"
}
prepared_complete() {
  [[ -f "$prepared/study_manifest.json" ]] || return 1
  grep -q '"rt_target": "'"$rt_target"'"' "$prepared/study_manifest.json" || return 1
  local seed organism
  for seed in 2024 2025 2026 2027 2028; do
    for organism in celegans triplemix human; do
      [[ -s "$prepared/seed_${seed}/tests/${organism}.tsv" ]] || return 1
      has_column "$prepared/seed_${seed}/tests/${organism}.tsv" rt || return 1
      has_column "$prepared/seed_${seed}/tests/${organism}.tsv" source_organism || return 1
    done
  done
}
snapshot_tests() {
  local source_dir=$1 destination_dir=$2 organism
  mkdir -p "$destination_dir"
  for organism in celegans triplemix human; do
    [[ -s "$source_dir/${organism}.tsv" ]] || {
      echo "Missing prepared test input: $source_dir/${organism}.tsv" >&2
      return 1
    }
    cp -f "$source_dir/${organism}.tsv" "$destination_dir/${organism}.tsv"
  done
  cp -f "$(dirname "$source_dir")/rt_alignment.json" "$destination_dir/rt_alignment.json"
}

if ! prepared_complete; then
  prep_python=${PREP_PYTHON:-python3}
  "$prep_python" "$prep" --output-dir "$prepared" --rt-target "$rt_target"
fi
prepared_complete || { echo "Prepared organism test inputs are incomplete: $prepared" >&2; exit 1; }
carafe2_python=${CARAFE2_PYTHON:-}
[[ -n "$carafe2_python" ]] || {
  echo "Set CARAFE2_PYTHON to the Python executable for this Carafe2 fork." >&2
  exit 1
}
[[ -x "$carafe2_python" ]] || {
  echo "Carafe2 Python is not executable: $carafe2_python" >&2
  exit 1
}
"$carafe2_python" -c 'import torch, alphabase' || {
  echo "Carafe2 Python lacks required dependencies (torch, alphabase): $carafe2_python" >&2
  exit 1
}
gpu_lock=${STUDY_GPU_LOCK:-/tmp/rt_organism_scaling_all_gpus.lock}
exec 8>"$gpu_lock"
echo "Waiting for exclusive all-GPU study lock: $gpu_lock"
flock 8
dispatcher="$data_collect_dir/run_parallel_organism_scaling.py"
gpu_ids=${STUDY_GPUS:-0,1,2}
jobs_per_gpu=${STUDY_JOBS_PER_GPU:-2}
epochs=${STUDY_EPOCHS:-60}
warmup=${STUDY_WARMUP_EPOCHS:-5}
echo "Carafe2 will use GPUs $gpu_ids with $jobs_per_gpu independent runs per GPU"
python3 "$dispatcher" --family carafe2 --study-root "$study_root" \
  --gpus "$gpu_ids" --jobs-per-gpu "$jobs_per_gpu" \
  --epochs "$epochs" --warmup-epochs "$warmup" \
  --carafe2-python "$carafe2_python"
echo "Carafe2 organism-scaling study complete: $study_root"
diann_study=${DIANN_STUDY_ROOT:-}
[[ -n "$diann_study" ]] || exit 0
diann_results=$(find "$diann_study/runs" -name organism_test_metrics.json -type f 2>/dev/null | wc -l)
carafe2_results=$(find "$study_root/runs" -name organism_test_metrics.json -type f 2>/dev/null | wc -l)
if (( diann_results >= 70 && carafe2_results >= 70 )); then
  plotter=${DIANN_FINETUNE_ROOT:-}/scripts/plot_organism_scaling_study.sh
  if [[ -x "$plotter" ]]; then
    "$plotter" "$diann_study" "$study_root" "${ORGANISM_SCALING_PLOTS_DIR:-$diann_study/plots}"
  else
    echo "Skipping comparison plots; set DIANN_FINETUNE_ROOT to the DIA-NN fine-tuning repo." >&2
  fi
fi
