#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export MAX_RESETTING_ENV_COUNT=20

openha_root="${REPO_ROOT}/third_party/OpenHA"
base_record_path="${REPO_ROOT}/outputs/openha_eval"

temperature=1
history_num=0
action_chunk_len=4 # 模型每次输出多少步的动作
instruction_type="normal"
use_keyboard=true
record_video=false

num_rollouts=2 # 每个任务的目标 rollout 数
resume_incomplete_only=true

parallel_workers=2 # Ray worker 数，通常对应可并行使用的 GPU 数
gpu_per_worker=1
rollout_pool_size=2
standby_pool_size=2
env_startup_workers=2
inference_batch_size=2
partial_batch_timeout=1.0
collector_poll_interval=0.05
max_startup_retries=3

task="kill_entity:*,craft_item:*,mine_block:*"
max_steps_num=1000
difficulty="zero"

model_local_path="sys555/GROW"

if [ "$resume_incomplete_only" = true ]; then
    model_ref=$(basename "$model_local_path")
    pending_tasks_output=$(python - <<PY
import sys

sys.path.insert(0, "$REPO_ROOT")

from jarvisvla.evaluate.openha_eval import count_existing_rollouts, ensure_openha_importable, expand_task_patterns

openha_root = "$openha_root"
record_path = "$base_record_path"
task_pattern = "$task"
model_ref = "$model_ref"
num_rollouts = int("$num_rollouts")

ensure_openha_importable(openha_root)
from openagents.envs.tasks.task_manager import get_available_tasks

all_tasks = get_available_tasks()
resolved_tasks = expand_task_patterns(task_pattern, all_tasks)

pending_tasks = []
for task_name in resolved_tasks:
    existing_rollouts = count_existing_rollouts(record_path, model_ref, task_name)
    if existing_rollouts < num_rollouts:
        pending_tasks.append(task_name)

print("__PENDING_TASKS__=" + ",".join(pending_tasks))
PY
)
    pending_tasks_status=$?
    if [ $pending_tasks_status -ne 0 ]; then
        echo "Failed to compute pending tasks."
        printf '%s\n' "$pending_tasks_output"
        exit $pending_tasks_status
    fi

    pending_tasks=$(printf '%s\n' "$pending_tasks_output" | grep '^__PENDING_TASKS__=' | tail -n 1 | sed 's/^__PENDING_TASKS__=//')

    if [ -z "$pending_tasks" ]; then
        echo "All requested tasks already have at least ${num_rollouts} rollouts. Nothing to do."
        exit 0
    fi

    echo "Only evaluating incomplete tasks:"
    echo "$pending_tasks"
    task="$pending_tasks"
fi

PYTHONUNBUFFERED=1 python -m jarvisvla.evaluate.openha_eval \
    --openha-root "$openha_root" \
    --checkpoints "$model_local_path" \
    --record_path "$base_record_path" \
    --task "$task" \
    --difficulty "$difficulty" \
    --num_rollouts "$num_rollouts" \
    --max_steps_num "$max_steps_num" \
    --parallel-workers "$parallel_workers" \
    --gpu_per_worker "$gpu_per_worker" \
    --rollout-pool-size "$rollout_pool_size" \
    --standby-pool-size "$standby_pool_size" \
    --env-startup-workers "$env_startup_workers" \
    --inference-batch-size "$inference_batch_size" \
    --partial-batch-timeout "$partial_batch_timeout" \
    --collector-poll-interval "$collector_poll_interval" \
    --max-startup-retries "$max_startup_retries" \
    --temperature "$temperature" \
    --history_num "$history_num" \
    --instruction-type "$instruction_type" \
    --action-chunk-len "$action_chunk_len" \
    $( [ "$record_video" = true ] && echo "--record-video" || echo "--no-record-video" ) \
    $( [ "$use_keyboard" = true ] && echo "--use-keyboard" )
