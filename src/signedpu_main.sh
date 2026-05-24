#!/bin/bash
set -Eeuo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")"

OUT_DIR="../out"
FAIL_LOG="$OUT_DIR/failed_runs.log"

LSP_LAMBDA_MAIN="${LSP_LAMBDA_MAIN:-1.0}"
STUDY_MAIN="${STUDY_MAIN:-main}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ONLY_DATASETS="${ONLY_DATASETS:-}"

mkdir -p "$OUT_DIR"

run_signedpu_main_keep_going() {
  local dataset=""
  local gpu_id=""
  local prev=""
  local arg
  for arg in "$@"; do
    if [[ "$prev" == "--data" ]]; then
      dataset="$arg"
    elif [[ "$prev" == "--gpu" ]]; then
      gpu_id="$arg"
    fi
    prev="$arg"
  done

  if [[ -n "$ONLY_DATASETS" && -n "$dataset" ]]; then
    case ",$ONLY_DATASETS," in
      *,"$dataset",*) ;;
      *) return 0 ;;
    esac
  fi

  local -a cmd=(-s main.py "$@")
  local cmd_str="python ${cmd[*]}"
  local tmp_log
  tmp_log="$(mktemp)"

  set +e
  command python "${cmd[@]}" 2>&1 | tee "$tmp_log"
  local cmd_status=${PIPESTATUS[0]}
  set -e

  if [[ $cmd_status -eq 0 ]]; then
    rm -f "$tmp_log"
    return 0
  fi

  printf '[%(%F %T)T] FAIL exit=%d cwd=%s data=%s gpu=%s cmd=%s\n' -1 "$cmd_status" "$PWD" "${dataset:-unknown}" "${gpu_id:-unknown}" "$cmd_str" | tee -a "$FAIL_LOG" >&2
  rm -f "$tmp_log"
  return 0
}

for i in $SEEDS
do
  run_signedpu_main_keep_going --data chameleon-filtered --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 5 --gpu 0
  run_signedpu_main_keep_going --data twitch-en --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 5 --gpu 0 --wo-preprocess
  run_signedpu_main_keep_going --data facebook --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 100 --gpu 1
  run_signedpu_main_keep_going --data pubmed --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 100 --gpu 1 --wo-preprocess
  run_signedpu_main_keep_going --data actor --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 100 --gpu 2
  run_signedpu_main_keep_going --data roman-empire --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 100 --gpu 2
  run_signedpu_main_keep_going --data amazon-computers --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 100 --gpu 3 --wo-preprocess
  run_signedpu_main_keep_going --data amazon-photo --model signedpu --loss sbre-lsp --lsp-lambda "$LSP_LAMBDA_MAIN" --study "$STUDY_MAIN" --seed "$i" --save-em-log --patience 100 --gpu 3 --wo-preprocess
done
