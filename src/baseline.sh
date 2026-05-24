#!/bin/bash

set -Eeuo pipefail

# Ensure conda works in non-interactive shells (e.g., tmux windows).
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
export PYTHONNOUSERSITE=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/out"
RUN_LOG="$OUT_DIR/run_status.log"
FAIL_LOG="$OUT_DIR/failed_runs.log"
GPU_ID="${GPU_ID:-0}"
STUDY="main"
ONLY_DATASETS="${ONLY_DATASETS:-}"

mkdir -p "$OUT_DIR"

backup_existing_baseline_logs() {
  [[ "${SKIP_BASELINE_BACKUP:-0}" == "1" ]] && return 0

  local study_dir="$OUT_DIR/log/$STUDY"
  [[ -d "$study_dir" ]] || return 0

  local -a baseline_logs=()
  local path
  shopt -s nullglob
  for path in "$study_dir"/*.csv; do
    [[ "$(basename "$path")" == signedpu_* ]] && continue
    baseline_logs+=("$path")
  done
  shopt -u nullglob

  [[ ${#baseline_logs[@]} -gt 0 ]] || return 0

  local backup_dir="$OUT_DIR/log/backups/$STUDY/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$backup_dir"
  mv "${baseline_logs[@]}" "$backup_dir"/
  printf '[%(%F %T)T] BACKUP study=%s moved=%d dir=%s\n' -1 "$STUDY" "${#baseline_logs[@]}" "$backup_dir" | tee -a "$RUN_LOG"
}

dataset_extra_args() {
  local dataset="$1"
  case "$dataset" in
    chameleon-filtered)
      echo "--patience 5"
      ;;
    twitch-en)
      echo "--wo-preprocess --patience 5"
      ;;
    pubmed|amazon-computers|amazon-photo)
      echo "--wo-preprocess --patience 100"
      ;;
    *)
      echo "--patience 100"
      ;;
  esac
}

run_main_oom_ok() {
  local dataset=""
  local prev=""
  local arg
  for arg in "$@"; do
    if [[ "$prev" == "--data" ]]; then
      dataset="$arg"
      break
    fi
    prev="$arg"
  done
  if [[ -n "$ONLY_DATASETS" && -n "$dataset" ]]; then
    case ",$ONLY_DATASETS," in
      *,"$dataset",*) ;;
      *) return 0 ;;
    esac
  fi

  local -a cmd=(-s main.py "$@" --gpu "$GPU_ID")
  local cmd_str="python ${cmd[*]}"
  local tmp_log
  tmp_log="$(mktemp)"

  printf '[%(%F %T)T] START cwd=%s cmd=%s\n' -1 "$PWD" "$cmd_str" >> "$RUN_LOG"

  set +e
  if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" command python "${cmd[@]}" 2>&1 | tee "$tmp_log"
  else
    command python "${cmd[@]}" 2>&1 | tee "$tmp_log"
  fi
  local cmd_status=${PIPESTATUS[0]}
  set -e

  if [[ $cmd_status -eq 0 ]]; then
    rm -f "$tmp_log"
    return 0
  fi

  if grep -Eq 'torch\.OutOfMemoryError|CUDA out of memory|CUDA error: out of memory' "$tmp_log"; then
    printf '[%(%F %T)T] OOM_SKIP exit=%d cwd=%s cmd=%s\n' -1 "$cmd_status" "$PWD" "$cmd_str" | tee -a "$FAIL_LOG" >&2
    command python - <<'PYTORCH_CLEAR'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PYTORCH_CLEAR
    rm -f "$tmp_log"
    return 0
  fi

  printf '[%(%F %T)T] FAIL exit=%d cwd=%s cmd=%s\n' -1 "$cmd_status" "$PWD" "$cmd_str" | tee -a "$FAIL_LOG" >&2
  rm -f "$tmp_log"
  return "$cmd_status"
}

run_dataset_suite() {
  local seed="$1"
  local dataset="$2"
  local extra_args
  extra_args="$(dataset_extra_args "$dataset")"

  run_main_oom_ok --data "$dataset" --model mlp --loss ce --seed "$seed" $extra_args
  run_main_oom_ok --data "$dataset" --model gat --loss ce --seed "$seed" $extra_args
  run_main_oom_ok --data "$dataset" --model mlp --loss ure --seed "$seed" --prior given $extra_args
  run_main_oom_ok --data "$dataset" --model gat --loss ure --seed "$seed" --prior given $extra_args
  run_main_oom_ok --data "$dataset" --model mlp --loss nre --seed "$seed" --prior given $extra_args
  run_main_oom_ok --data "$dataset" --model gat --loss nre --seed "$seed" --prior given $extra_args
  run_main_oom_ok --data "$dataset" --model pulp --loss distpu --seed "$seed" --prior given $extra_args
  run_main_oom_ok --data "$dataset" --model lsdan --loss nre --seed "$seed" --prior given $extra_args
  run_main_oom_ok --data "$dataset" --model grab --loss bre --seed "$seed" $extra_args
  run_main_oom_ok --data "$dataset" --model gpl --loss wce --seed "$seed" $extra_args
}

backup_existing_baseline_logs

for i in 0 1 2 3 4 5 6 7 8 9
do
  run_dataset_suite "$i" chameleon-filtered
  run_dataset_suite "$i" twitch-en
  run_dataset_suite "$i" facebook
  run_dataset_suite "$i" pubmed
  run_dataset_suite "$i" actor
  run_dataset_suite "$i" roman-empire
  run_dataset_suite "$i" amazon-computers
  run_dataset_suite "$i" amazon-photo
done
