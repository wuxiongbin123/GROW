#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export MAX_RESETTING_ENV_COUNT="${MAX_RESETTING_ENV_COUNT:-1}"

openha_root="${OPENHA_ROOT:-${REPO_ROOT}/third_party/OpenHA}"
run_id="${RUN_ID:-$(date +%y-%m-%d-%H-%M-%S)}"
base_record_path="${RECORD_PATH:-${REPO_ROOT}/outputs/openha_single_gold_shovel/${run_id}}"

temperature="${TEMPERATURE:-1}"
history_num="${HISTORY_NUM:-0}"
action_chunk_len="${ACTION_CHUNK_LEN:-4}"
instruction_type="${INSTRUCTION_TYPE:-normal}"
use_keyboard="${USE_KEYBOARD:-true}"
record_video="${RECORD_VIDEO:-false}"
benchmark_profile="${BENCHMARK_PROFILE:-}"

num_rollouts=1
parallel_workers=1
gpu_per_worker="${GPU_PER_WORKER:-1}"
rollout_pool_size=1
standby_pool_size=0
env_startup_workers=1
inference_batch_size=1
partial_batch_timeout="${PARTIAL_BATCH_TIMEOUT:-0.0}"
collector_poll_interval="${COLLECTOR_POLL_INTERVAL:-0.05}"
max_startup_retries=1
max_rollout_retries=0

task="${TASK:-craft_item:golden_shovel}"
max_steps_num="${MAX_STEPS_NUM:-600}"
difficulty="${DIFFICULTY:-zero}"

model_local_path="${MODEL_LOCAL_PATH:-/mnt/shared-storage-user/steai-share/wuxiongbin/checkpoints/GROW_7B}"

echo "Single-env OpenHA evaluation"
echo "  model: ${model_local_path}"
echo "  task: ${task}"
echo "  rollouts: ${num_rollouts}"
echo "  record_path: ${base_record_path}"
echo "  fps metric: printed as [REALTIME_FPS] and saved in success.json/loss.json"

cmd=(
    python -m jarvisvla.evaluate.openha_eval
    --openha-root "$openha_root"
    --checkpoints "$model_local_path"
    --record_path "$base_record_path"
    --task "$task"
    --difficulty "$difficulty"
    --num_rollouts "$num_rollouts"
    --max_steps_num "$max_steps_num"
    --parallel-workers "$parallel_workers"
    --gpu_per_worker "$gpu_per_worker"
    --rollout-pool-size "$rollout_pool_size"
    --standby-pool-size "$standby_pool_size"
    --env-startup-workers "$env_startup_workers"
    --inference-batch-size "$inference_batch_size"
    --partial-batch-timeout "$partial_batch_timeout"
    --collector-poll-interval "$collector_poll_interval"
    --max-startup-retries "$max_startup_retries"
    --max-rollout-retries "$max_rollout_retries"
    --temperature "$temperature"
    --history_num "$history_num"
    --instruction-type "$instruction_type"
    --action-chunk-len "$action_chunk_len"
)

if [ "$record_video" = true ]; then
    cmd+=(--record-video)
else
    cmd+=(--no-record-video)
fi

if [ "$use_keyboard" = true ]; then
    cmd+=(--use-keyboard)
fi

if [ -n "$benchmark_profile" ]; then
    cmd+=(--benchmark-profile "$benchmark_profile")
fi

PYTHONUNBUFFERED=1 "${cmd[@]}"
