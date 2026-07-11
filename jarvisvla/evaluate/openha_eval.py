import argparse
import json
import logging
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ray
import torch
from ray.util.queue import Empty, Queue
from jarvisvla.evaluate.benchmark_profiles import (
    apply_benchmark_profile,
    summarize_benchmark_profile,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_OPENHA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "OpenHA"


def ensure_openha_importable(openha_root: str) -> Path:
    openha_path = Path(openha_root).expanduser().resolve()
    if not openha_path.exists():
        raise FileNotFoundError(f"OpenHA root not found: {openha_path}")
    if str(openha_path) not in sys.path:
        sys.path.insert(0, str(openha_path))
    return openha_path


def normalize_task_descriptions(task_description, task_name: str) -> list[str]:
    del task_description
    return [task_name]


class JarvisOpenHABridge:
    action_type = "agent"

    def __init__(self, agent_cls, checkpoint_path: str, model_config: dict, can_print: bool = False):
        self.agent = agent_cls(checkpoint_path=checkpoint_path, can_print=can_print, **model_config)
        self._points = [[320, 180]]
        self._policy_type = "jarvis_vla"

    def get_actions_batch(self, infos: list[dict], instructions: list[str], verbose: bool = False):
        observations = [info["pov"] for info in infos]
        return self.agent.generate_action_chunks_batch(
            observations,
            instructions,
            verbos=verbose,
        )


def infer_openha_segment_type(task_name: str) -> str:
    if task_name.startswith("mine_block:"):
        return "Mine"
    if task_name.startswith("kill_entity:"):
        return "Hunt"
    if task_name.startswith("craft_item:") or task_name.startswith("smelt_item:"):
        return "Craft"
    return "Explore"


class OpenHAEvalBridge:
    action_type = "env"

    def __init__(self, args, can_print: bool = False):
        ensure_openha_importable(args.openha_root)
        logging.info("Importing OpenHA agent module for backend=%s...", args.agent_backend)
        from openagents.agents.openha import OpenHA
        logging.info("OpenHA agent module imported.")

        self.args = args
        self.can_print = can_print
        self._agent_cls = OpenHA

    def prepare_session(self, session):
        if session.runtime_agent is not None:
            return

        model_url = (
            f"http://{random.choice(self.args.model_ips.split(','))}:"
            f"{random.choice(self.args.model_ports.split(','))}/v1"
        )
        session.runtime_agent = self._agent_cls(
            output_mode=self.args.output_mode,
            raw_action_type=self.args.raw_action_type,
            vlm_client_mode=self.args.vlm_client_mode,
            model_path=self.args.model_path,
            grounding_policy_path=self.args.grounding_policy_path,
            motion_policy_path=self.args.motion_policy_path,
            sam_path=self.args.sam_path,
            segment_type=infer_openha_segment_type(session.task_name),
            model_id=self.args.model_id,
            model_url=model_url,
            maximum_history_length=self.args.maximum_history_length,
            action_chunk_len=self.args.action_chunk_len,
            instruction_type=self.args.instruction_type,
            LLM_backbone=self.args.LLM_backbone,
            VLM_backbone=self.args.VLM_backbone,
            tokenizer_path=self.args.tokenizer_path,
            system_message=self.args.system_message,
            system_message_tag=self.args.system_message_tag,
            enforce_format=self.args.enforce_format,
            enforce_prefix=self.args.enforce_prefix,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            grounding_inference_interval=self.args.grounding_inference_interval,
            motion_inference_interval=self.args.motion_inference_interval,
        )
        session.runtime_agent.reset(
            instruction=session.current_instruction(),
            task_name=session.task_name,
        )

    def get_responses_for_requests(self, requests: list, verbose: bool = False):
        responses = []
        for request in requests:
            session = request.session
            self.prepare_session(session)
            action = session.runtime_agent.get_action(
                obs=session.obs,
                info=request.info,
                instruction=request.instruction,
                verbose=verbose,
            )
            responses.append(
                {
                    "actions": [action],
                    "raw_response": session.runtime_agent.response or "",
                    "points": session.runtime_agent.points or [[320, 180]],
                    "policy_type": session.runtime_agent.policy_type or "openha",
                }
            )
        return responses


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def create_runtime_components(args):
    ensure_openha_importable(args.openha_root)
    if args.agent_backend == "openha":
        bridge = OpenHAEvalBridge(args, can_print=args.verbose)
        return bridge, None, None

    from jarvisvla.evaluate.agent_wrapper import VLLM_AGENT
    from openagents.agents.base_agent import MineCraftAgent

    bridge = JarvisOpenHABridge(
        agent_cls=VLLM_AGENT,
        checkpoint_path=args.checkpoints,
        model_config={
            "temperature": args.temperature,
            "history_num": args.history_num,
            "instruction_type": args.instruction_type,
            "action_chunk_len": args.action_chunk_len,
            "use_keyboard": args.use_keyboard,
        },
        can_print=args.verbose,
    )
    action_mapper, action_transformer = MineCraftAgent.init_action_mapper_and_transformer()
    return bridge, action_mapper, action_transformer


def expand_task_patterns(task_patterns: str, all_tasks: list[str]) -> list[str]:
    resolved_tasks = []
    seen = set()
    for pattern in [p.strip() for p in task_patterns.split(",") if p.strip()]:
        if pattern == "random":
            candidates = list(all_tasks)
        else:
            regex = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
            compiled = re.compile(regex)
            candidates = [task_name for task_name in all_tasks if compiled.match(task_name)]
            if pattern == "smelt_item:*":
                candidates = [task_name for task_name in all_tasks if task_name.startswith("smelt_item:")]
        if not candidates:
            raise ValueError(f"No task matched pattern: {pattern}")
        for task_name in candidates:
            if task_name not in seen:
                seen.add(task_name)
                resolved_tasks.append(task_name)
    return resolved_tasks


def count_existing_rollouts(record_path: str, model_ref: str, task_name: str) -> int:
    task_dir = os.path.join(record_path, f"{model_ref}-jarvis-openha", task_name)
    if not os.path.isdir(task_dir):
        return 0

    count = 0
    for rollout_dir in os.listdir(task_dir):
        rollout_path = os.path.join(task_dir, rollout_dir)
        if not os.path.isdir(rollout_path):
            continue
        if (
            os.path.exists(os.path.join(rollout_path, "success.json"))
            or os.path.exists(os.path.join(rollout_path, "loss.json"))
        ):
            count += 1
    return count


class RolloutSession:
    TRACE_POINTS = [[320, 180]]
    TRACE_POLICY_TYPE = "jarvis_vla"

    def __init__(self, args, task_name: str):
        self.args = args
        self.task_name = task_name
        self.task_item = None
        self.task_config = None
        self.instructions = []
        self.rollout_path = ""
        self.raw_action_file_path = ""
        self.max_buffer_size = 1

        self.env = None
        self.env_closed = False
        self.noop_action = None
        self.obs = None
        self.info = None

        self.success_state = False
        self.subgoal_reward = 0
        self.subgoal_idx = 0
        self.processed_raw_actions = [{"raw_action": ""}]
        self.start_time = 0.0
        self.fps_timer_start = 0.0
        self.fps_timer_start_wall = 0.0
        self.fps_timer_end = 0.0
        self.realtime_wall_time_sec = 0.0
        self.realtime_fps = 0.0
        self.run_frame_idx = 0
        self.pending_actions = []
        self.current_raw_response = ""
        self.done = False
        self.prewarm_started_at = 0.0
        self.ready_at = 0.0
        self.current_points = list(self.TRACE_POINTS)
        self.current_policy_type = self.TRACE_POLICY_TYPE
        self.runtime_agent = None

    def start(self):
        ensure_openha_importable(self.args.openha_root)

        from openagents.envs import DEFAULT_MAXIMUM_BUFFER_SIZE
        from openagents.envs.env import env_init
        from openagents.envs.tasks.task_manager import gen_task_config

        self.prewarm_started_at = time.time()
        self.task_config = gen_task_config(self.task_name, difficulty=self.args.difficulty)
        self.task_config = apply_benchmark_profile(self.task_config, self.args)
        benchmark_summary = summarize_benchmark_profile(self.task_config, self.args)
        if benchmark_summary:
            logging.info(
                "Applied benchmark profile %s for %s: %s",
                self.args.benchmark_profile,
                self.task_name,
                benchmark_summary,
            )
        model_ref = self.args.model_id or Path(self.args.checkpoints).name
        rollout_tag = f"{model_ref}-jarvis-openha"
        self.max_buffer_size = DEFAULT_MAXIMUM_BUFFER_SIZE if self.args.num_rollouts > 10 else 1

        self.rollout_path = os.path.join(
            self.args.record_path,
            rollout_tag,
            self.task_name,
            datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            + "-"
            + self.task_name.replace(":", "_")
            + "-"
            + str(uuid.uuid4()).replace("-", "")[:8],
        )
        os.makedirs(self.rollout_path, exist_ok=True)
        self.raw_action_file_path = os.path.join(self.rollout_path, "raw_action.jsonl")

        self.env = env_init(self.task_config, self.rollout_path, self.args)
        self.noop_action = self.env.noop_action()
        self.obs, _, _, _, self.info = self.env.step(self.noop_action)

        self.instructions = normalize_task_descriptions(
            self.task_config.get("task_description"),
            self.task_name,
        )
        self.start_time = time.time()
        self.ready_at = self.start_time
        logging.info("OpenHA task %s -> %s", self.task_name, self.instructions)

    def current_instruction(self) -> str:
        return self.instructions[self.subgoal_idx]

    def start_fps_timer(self):
        if self.fps_timer_start > 0:
            return
        self.fps_timer_start = time.perf_counter()
        self.fps_timer_start_wall = time.time()
        logging.info(
            "Realtime FPS timer started: task=%s frame=%s",
            self.task_name,
            self.run_frame_idx,
        )

    def stop_fps_timer(self) -> dict:
        if self.fps_timer_start <= 0:
            self.start_fps_timer()
        self.fps_timer_end = time.perf_counter()
        self.realtime_wall_time_sec = max(self.fps_timer_end - self.fps_timer_start, 1e-6)
        self.realtime_fps = self.run_frame_idx / self.realtime_wall_time_sec
        return {
            "frames": self.run_frame_idx,
            "wall_time_sec": self.realtime_wall_time_sec,
            "fps": self.realtime_fps,
            "timer_start_unix": self.fps_timer_start_wall,
            "definition": (
                "agent_control_frames divided by wall-clock seconds; timer starts when "
                "the active rollout runner begins after env reset/prewarm and stops "
                "before end pause or video rendering"
            ),
        }

    def needs_model_action(self) -> bool:
        return not self.done and not self.pending_actions

    def set_pending_actions(self, actions: list, raw_response: str, points=None, policy_type=None):
        self.pending_actions = list(actions or ["no_action"])
        self.current_raw_response = raw_response or ""
        self.current_points = list(points or self.TRACE_POINTS)
        self.current_policy_type = policy_type or self.TRACE_POLICY_TYPE

    def append_raw_action(self):
        self.processed_raw_actions.append(
            {
                "points": self.current_points,
                "raw_action": self.current_raw_response,
                "action_type": self.current_policy_type,
            }
        )
        if len(self.processed_raw_actions) % self.max_buffer_size == 0:
            self.flush_raw_actions()

    def flush_raw_actions(self):
        if not self.processed_raw_actions:
            return
        with open(self.raw_action_file_path, "a", encoding="utf-8") as f:
            for processed_raw_action in self.processed_raw_actions:
                f.write(json.dumps(processed_raw_action, ensure_ascii=False) + "\n")
        self.processed_raw_actions = []

    def step(self, action_mapper, action_transformer, action_mode: str = "agent"):
        ensure_openha_importable(self.args.openha_root)
        if action_mode == "agent":
            from openagents.agents.base_agent import MineCraftAgent

        if self.done:
            return

        action = self.pending_actions.pop(0) if self.pending_actions else "no_action"
        if action_mode == "agent":
            action = MineCraftAgent.agent_action_to_env_action(
                action=action,
                action_mapper=action_mapper,
                action_transformer=action_transformer,
            )
        if action is None or action == "no_action" or action == "no_op":
            action = self.noop_action

        self.append_raw_action()
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.subgoal_reward += reward
        self.run_frame_idx += 1

        if terminated or truncated:
            self.success_state = True
            self.done = True
            return

        if self.subgoal_reward >= 1:
            if self.subgoal_idx < len(self.instructions) - 1:
                self.subgoal_idx += 1
                self.subgoal_reward = 0
            else:
                self.success_state = True
                self.done = True
                return

        if self.run_frame_idx >= self.args.max_steps_num:
            self.done = True

    def finalize(self):
        ensure_openha_importable(self.args.openha_root)
        from openagents.utils.file_op import clip_action_info, save_render_videos
        from openagents.utils.render import render_video

        try:
            fps_metrics = self.stop_fps_timer()
            logging.info(
                "Realtime evaluation FPS: task=%s frames=%s wall_time=%.4fs fps=%.4f success=%s",
                self.task_name,
                fps_metrics["frames"],
                fps_metrics["wall_time_sec"],
                fps_metrics["fps"],
                self.success_state,
            )
            print(
                "[REALTIME_FPS] "
                f"task={self.task_name} "
                f"frames={fps_metrics['frames']} "
                f"wall_time_sec={fps_metrics['wall_time_sec']:.4f} "
                f"fps={fps_metrics['fps']:.4f} "
                f"success={self.success_state}",
                flush=True,
            )

            end_pause = random.randint(5, 20) if self.success_state else 1
            for _ in range(end_pause):
                time.sleep(0.1)
                self.processed_raw_actions.append({"raw_action": ""})
                self.env.step(self.noop_action)

            self.flush_raw_actions()

            marker_path = "success.json" if self.success_state else "loss.json"
            payload = {
                "frames": self.run_frame_idx,
                "wall_time_sec": fps_metrics["wall_time_sec"],
                "fps": fps_metrics["fps"],
                "fps_metrics": fps_metrics,
                "args": vars(self.args),
            }
            if self.success_state:
                payload["success"] = True
            with open(os.path.join(self.rollout_path, marker_path), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self.close_env()

            if self.args.record_video:
                clip_action_info(self.rollout_path, assert_video_action_num_same=True)
                rendered_frames = render_video(
                    self.rollout_path,
                    enable_task_name=True,
                    enable_rawaction=True,
                    enable_thought=False,
                    enable_envaction=True,
                    enable_point_cot=False,
                    raw_action_max_length=5,
                )
                save_render_videos(self.rollout_path, rendered_frames, fps=self.args.fps)
            return self.rollout_path
        except Exception:
            logging.exception(
                "Rollout failed for task %s, removing partial directory: %s",
                self.task_name,
                self.rollout_path,
            )
            self.close_env()
            shutil.rmtree(self.rollout_path, ignore_errors=True)
            raise

    def close_env(self):
        if self.env is not None and not self.env_closed:
            self.env.close()
            self.env_closed = True


@dataclass
class InferenceRequest:
    session: RolloutSession
    info: dict
    instruction: str
    response_queue: queue.Queue
    submitted_at: float


@dataclass
class InferenceResponse:
    actions: list
    raw_response: str
    points: list | None = None
    policy_type: str | None = None


class GPUInferenceEngine(threading.Thread):
    def __init__(self, worker, bridge: JarvisOpenHABridge):
        super().__init__(daemon=True)
        self.worker = worker
        self.bridge = bridge
        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()

    def submit(self, request: InferenceRequest):
        self.request_queue.put(request)
        logging.info(
            "Inference request queued: task=%s frame=%s queue_depth=%s",
            request.session.task_name,
            request.session.run_frame_idx,
            self.request_queue.qsize(),
        )

    def run(self):
        batch_size = max(self.worker.args.inference_batch_size, 1)
        while True:
            if self.worker.stop_event.is_set() and self.request_queue.empty():
                return

            try:
                first_request = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first_request]
            batch_started_at = time.time()

            while len(batch) < batch_size:
                if self.worker.should_flush_batch(len(batch), batch_started_at):
                    break
                try:
                    request = self.request_queue.get(timeout=self.worker.args.collector_poll_interval)
                except queue.Empty:
                    continue
                batch.append(request)

            infos = [request.info for request in batch]
            instructions = [request.instruction for request in batch]
            logging.info(
                "Inference batch size=%s target=%s pending_queue=%s waited=%.3fs oldest_req_age=%.3fs",
                len(batch),
                batch_size,
                self.request_queue.qsize(),
                time.time() - batch_started_at,
                time.time() - min(request.submitted_at for request in batch),
            )

            if hasattr(self.bridge, "get_responses_for_requests"):
                bridge_outputs = self.bridge.get_responses_for_requests(
                    batch,
                    verbose=self.worker.args.verbose,
                )
            else:
                action_chunks, raw_responses = self.bridge.get_actions_batch(
                    infos,
                    instructions,
                    verbose=self.worker.args.verbose,
                )
                bridge_outputs = [
                    {"actions": list(actions), "raw_response": raw_response}
                    for actions, raw_response in zip(action_chunks, raw_responses)
                ]

            for request, output in zip(batch, bridge_outputs):
                request.response_queue.put(
                    InferenceResponse(
                        actions=list(output.get("actions", [])),
                        raw_response=output.get("raw_response", ""),
                        points=output.get("points"),
                        policy_type=output.get("policy_type"),
                    )
                )


class EnvironmentRunner(threading.Thread):
    def __init__(self, worker, session: RolloutSession):
        super().__init__(daemon=True)
        self.worker = worker
        self.session = session

    def run(self):
        try:
            self.session.start_fps_timer()
            while not self.session.done:
                if self.session.needs_model_action():
                    response_queue = queue.Queue(maxsize=1)
                    self.worker.inference_engine.submit(
                        InferenceRequest(
                            session=self.session,
                            info=self.session.info,
                            instruction=self.session.current_instruction(),
                            response_queue=response_queue,
                            submitted_at=time.time(),
                        )
                    )
                    response = response_queue.get()
                    self.session.set_pending_actions(
                        response.actions,
                        response.raw_response,
                        points=response.points,
                        policy_type=response.policy_type,
                    )

                self.session.step(
                    action_mapper=self.worker.action_mapper,
                    action_transformer=self.worker.action_transformer,
                    action_mode=self.worker.bridge.action_type,
                )

            rollout_path = self.session.finalize()
            self.worker.completion_queue.put(
                {
                    "task_name": self.session.task_name,
                    "rollout_path": rollout_path,
                    "runner_id": id(self),
                    "task_item": self.session.task_item,
                }
            )
        except Exception as exc:
            logging.exception("Environment runner failed for task %s", self.session.task_name)
            try:
                self.session.close_env()
                if self.session.rollout_path:
                    shutil.rmtree(self.session.rollout_path, ignore_errors=True)
            except Exception:
                logging.exception("Failed to cleanup broken rollout for task %s", self.session.task_name)
            self.worker.completion_queue.put(
                {
                    "task_name": self.session.task_name,
                    "rollout_path": None,
                    "runner_id": id(self),
                    "error": repr(exc),
                    "task_item": self.session.task_item,
                }
            )


class ParallelTaskWorker:
    def __init__(self, args_dict: dict):
        self.args = argparse.Namespace(**args_dict)
        self.bridge, self.action_mapper, self.action_transformer = create_runtime_components(self.args)

        self.stop_event = threading.Event()
        self.completion_queue: queue.Queue = queue.Queue()
        self.ready_sessions: queue.Queue[RolloutSession] = queue.Queue()
        self.inference_engine = GPUInferenceEngine(self, self.bridge)
        self.startup_executor = ThreadPoolExecutor(max_workers=max(self.args.env_startup_workers, 1))

        self._state_lock = threading.Lock()
        self._active_runners: dict[int, EnvironmentRunner] = {}
        self._ready_count = 0
        self._starting_count = 0
        self._task_queue_exhausted = False
        self._fatal_error = None

    def _get_task_item(self, task_queue):
        try:
            return task_queue.get(block=False)
        except Empty:
            self._task_queue_exhausted = True
            return None

    def _start_session(self, task_item: dict):
        session = RolloutSession(self.args, task_item["task_name"])
        session.task_item = dict(task_item)
        session.start()
        if hasattr(self.bridge, "prepare_session"):
            self.bridge.prepare_session(session)
        return session

    def _submit_startup(self, task_item: dict):
        task_item = dict(task_item)
        task_item["startup_attempt"] = task_item.get("startup_attempt", 0) + 1
        task_item.setdefault("startup_submitted_at", time.time())
        with self._state_lock:
            self._starting_count += 1

        future = self.startup_executor.submit(self._start_session, task_item)
        future.add_done_callback(lambda fut, item=task_item: self._on_startup_done(fut, item))

    def _on_startup_done(self, future, task_item: dict):
        with self._state_lock:
            self._starting_count -= 1

        try:
            session = future.result()
        except Exception as exc:
            logging.warning(
                "Env prewarm failed for task=%s rollout=%s/%s attempt=%s error=%s",
                task_item["task_name"],
                task_item["rollout_idx"],
                task_item["total_rollouts"],
                task_item["startup_attempt"],
                exc,
            )
            if task_item["startup_attempt"] < self.args.max_startup_retries and not self.stop_event.is_set():
                time.sleep(random.uniform(0.5, 1.5))
                self._submit_startup(task_item)
            else:
                self._fatal_error = RuntimeError(
                    f"Env prewarm failed permanently for task {task_item['task_name']}: {exc}"
                )
            return

        self.ready_sessions.put(session)
        with self._state_lock:
            self._ready_count += 1
        logging.info(
            "Standby ready: task=%s ready=%s active=%s starting=%s prewarm_latency=%.3fs queue_wait=%.3fs",
            session.task_name,
            self.ready_count,
            self.active_count,
            self.starting_count,
            session.ready_at - session.prewarm_started_at,
            time.time() - task_item["startup_submitted_at"],
        )

    @property
    def active_count(self) -> int:
        with self._state_lock:
            return len(self._active_runners)

    @property
    def ready_count(self) -> int:
        with self._state_lock:
            return self._ready_count

    @property
    def starting_count(self) -> int:
        with self._state_lock:
            return self._starting_count

    def _desired_inventory(self) -> int:
        return self.args.rollout_pool_size + self.args.standby_pool_size

    def refill_inventory(self, task_queue):
        while not self.stop_event.is_set():
            with self._state_lock:
                inventory = len(self._active_runners) + self._ready_count + self._starting_count
                startup_slots = max(self.args.env_startup_workers - self._starting_count, 0)
                missing = max(self._desired_inventory() - inventory, 0)

            if startup_slots <= 0 or missing <= 0 or self._task_queue_exhausted:
                return

            task_item = self._get_task_item(task_queue)
            if task_item is None:
                return

            logging.info(
                "Schedule prewarm: task=%s rollout=%s/%s inventory=%s target=%s ready=%s active=%s starting=%s",
                task_item["task_name"],
                task_item["rollout_idx"],
                task_item["total_rollouts"],
                inventory,
                self._desired_inventory(),
                self.ready_count,
                self.active_count,
                self.starting_count,
            )
            self._submit_startup(task_item)

    def activate_ready_sessions(self):
        while not self.stop_event.is_set():
            with self._state_lock:
                active_full = len(self._active_runners) >= self.args.rollout_pool_size
            if active_full:
                return

            try:
                session = self.ready_sessions.get_nowait()
            except queue.Empty:
                return

            with self._state_lock:
                self._ready_count -= 1

            runner = EnvironmentRunner(self, session)
            with self._state_lock:
                self._active_runners[id(runner)] = runner

            logging.info(
                "Activate rollout env: task=%s active=%s ready=%s starting=%s standby_wait=%.3fs",
                session.task_name,
                self.active_count,
                self.ready_count,
                self.starting_count,
                time.time() - session.ready_at,
            )
            runner.start()

    def should_flush_batch(self, pending_batch_size: int, batch_started_at: float) -> bool:
        if pending_batch_size >= self.args.inference_batch_size:
            return True

        waited = time.time() - batch_started_at
        if waited < self.args.partial_batch_timeout:
            return False

        logging.info(
            "Timed batch flush: size=%s target=%s waited=%.3fs active=%s ready=%s starting=%s",
            pending_batch_size,
            self.args.inference_batch_size,
            waited,
            self.active_count,
            self.ready_count,
            self.starting_count,
        )
        return True

    def _requeue_task_item(self, task_queue, task_item: dict | None, reason: str):
        if task_item is None:
            return

        retry_count = task_item.get("rollout_retry_count", 0) + 1
        if retry_count > self.args.max_rollout_retries:
            raise RuntimeError(
                f"Task {task_item['task_name']} exceeded max_rollout_retries={self.args.max_rollout_retries} after {reason}"
            )

        retry_item = dict(task_item)
        retry_item["rollout_retry_count"] = retry_count
        task_queue.put(retry_item)
        self._task_queue_exhausted = False
        logging.warning(
            "Requeue rollout item: task=%s rollout=%s/%s retry=%s reason=%s",
            retry_item["task_name"],
            retry_item["rollout_idx"],
            retry_item["total_rollouts"],
            retry_count,
            reason,
        )

    def _handle_completion(self, result: dict, result_queue, task_queue):
        runner_id = result["runner_id"]
        with self._state_lock:
            self._active_runners.pop(runner_id, None)

        if result.get("error"):
            self._requeue_task_item(
                task_queue,
                result.get("task_item"),
                reason=f"env_runner_error:{result['error']}",
            )
            logging.warning(
                "Rollout crashed and will be replaced from standby when available: task=%s active=%s ready=%s starting=%s",
                result["task_name"],
                self.active_count,
                self.ready_count,
                self.starting_count,
            )
            self.activate_ready_sessions()
            self.refill_inventory(task_queue)
            return

        result_queue.put(
            {
                "task_name": result["task_name"],
                "rollout_path": result["rollout_path"],
            }
        )

        logging.info(
            "Rollout completed: task=%s active=%s ready=%s starting=%s",
            result["task_name"],
            self.active_count,
            self.ready_count,
            self.starting_count,
        )

        # Replace a finished rollout env immediately from the hot-standby pool,
        # then kick off background prewarm to refill standby capacity.
        self.activate_ready_sessions()
        self.refill_inventory(task_queue)

    def run_forever(self, task_queue, result_queue):
        self.inference_engine.start()
        self.refill_inventory(task_queue)

        try:
            while not self.stop_event.is_set():
                if self._fatal_error is not None:
                    raise self._fatal_error

                self.activate_ready_sessions()
                self.refill_inventory(task_queue)

                try:
                    result = self.completion_queue.get(timeout=0.05)
                except queue.Empty:
                    result = None

                if result is not None:
                    self._handle_completion(result, result_queue, task_queue)
                    continue

                with self._state_lock:
                    is_idle = (
                        self._task_queue_exhausted
                        and len(self._active_runners) == 0
                        and self._ready_count == 0
                        and self._starting_count == 0
                    )
                if is_idle:
                    break
        finally:
            self.stop_event.set()
            self.inference_engine.join(timeout=5.0)
            self.startup_executor.shutdown(wait=True, cancel_futures=False)

        return True


class LocalTaskQueue:
    def __init__(self, items: list[dict]):
        self._items = list(items)
        self._lock = threading.Lock()

    def get(self, block=False):
        del block
        with self._lock:
            if not self._items:
                raise Empty
            return self._items.pop(0)

    def put(self, item):
        with self._lock:
            self._items.append(item)


class LocalResultQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def rollout(args):
    ensure_openha_importable(args.openha_root)
    from openagents.envs.tasks.task_manager import get_available_tasks

    all_tasks = get_available_tasks()
    resolved_tasks = expand_task_patterns(args.task, all_tasks)
    logging.info("Resolved %s tasks from pattern %s", len(resolved_tasks), args.task)
    model_ref = args.model_id or Path(args.checkpoints).name

    task_queue = []
    for idx, task_name in enumerate(resolved_tasks, start=1):
        existing_rollouts = count_existing_rollouts(args.record_path, model_ref, task_name)
        if existing_rollouts >= args.num_rollouts:
            logging.info(
                "[%s/%s] Skip task %s: existing rollouts %s >= required %s",
                idx,
                len(resolved_tasks),
                task_name,
                existing_rollouts,
                args.num_rollouts,
            )
            continue

        missing_rollouts = args.num_rollouts - existing_rollouts
        logging.info(
            "[%s/%s] Queue task %s: existing=%s missing=%s target=%s",
            idx,
            len(resolved_tasks),
            task_name,
            existing_rollouts,
            missing_rollouts,
            args.num_rollouts,
        )
        for rollout_idx in range(missing_rollouts):
            task_queue.append(
                {
                    "task_name": task_name,
                    "rollout_idx": rollout_idx + 1,
                    "total_rollouts": args.num_rollouts,
                    "task_position": idx,
                    "task_count": len(resolved_tasks),
                }
            )

    if not task_queue:
        logging.info("All tasks already satisfy the requested rollout count. Nothing to do.")
        return []

    if args.parallel_workers <= 1:
        local_worker = ParallelTaskWorker(vars(args))
        local_result_queue = LocalResultQueue()
        local_worker.run_forever(LocalTaskQueue(task_queue), local_result_queue)
        return [item["rollout_path"] for item in local_result_queue.items]

    num_gpus = torch.cuda.device_count()
    logging.info("Available GPUs: %s", num_gpus)
    ray.init(num_gpus=num_gpus, log_to_driver=True, ignore_reinit_error=True)

    worker_count = min(args.parallel_workers, len(task_queue))
    worker_cls = ray.remote(num_gpus=args.gpu_per_worker)(ParallelTaskWorker)
    workers = [worker_cls.remote(vars(args)) for _ in range(worker_count)]

    rollout_paths = []
    result_queue = Queue()
    shared_task_queue = Queue()
    total_rollouts = len(task_queue)

    for task_item in task_queue:
        shared_task_queue.put(task_item)

    try:
        worker_futures = [worker.run_forever.remote(shared_task_queue, result_queue) for worker in workers]

        for completed_idx in range(total_rollouts):
            result = result_queue.get()
            rollout_paths.append(result["rollout_path"])
            if (completed_idx + 1) % max(args.rollout_pool_size, 1) == 0 or completed_idx + 1 == total_rollouts:
                logging.info(
                    "Completed rollouts: %s/%s (latest task=%s)",
                    completed_idx + 1,
                    total_rollouts,
                    result["task_name"],
                )

        ray.get(worker_futures)
    finally:
        ray.shutdown()

    return rollout_paths


def rollout_wrapper(args):
    try:
        return rollout(args)
    except Exception as exc:
        print(f"\n[ERROR] Worker failed with error: {exc}")
        traceback.print_exc()
        raise


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openha-root", type=str, default=os.environ.get("OPENHA_ROOT", str(DEFAULT_OPENHA_ROOT)))
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="")
    parser.add_argument("--record_path", type=str, required=True)
    parser.add_argument("--agent-backend", type=str, choices=["jarvis", "openha"], default="jarvis")

    parser.add_argument("--task", type=str, default="mine_block:*")
    parser.add_argument("--difficulty", type=str, default="zero")
    parser.add_argument("--max_steps_num", type=int, default=300)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--parallel-workers", dest="parallel_workers", type=int, default=1)
    parser.add_argument("--parallel_tasks", dest="parallel_workers", type=int)
    parser.add_argument("--gpu_per_worker", type=float, default=1.0)

    parser.add_argument("--rollout-pool-size", dest="rollout_pool_size", type=int, default=5)
    parser.add_argument("--envs_per_worker", dest="rollout_pool_size", type=int)
    parser.add_argument("--standby-pool-size", dest="standby_pool_size", type=int, default=3)
    parser.add_argument("--env-pool-target-ready", dest="standby_pool_size", type=int)
    parser.add_argument("--env-pool-min-ready", dest="env_pool_min_ready", type=int, default=0)
    parser.add_argument("--env-startup-workers", dest="env_startup_workers", type=int, default=3)
    parser.add_argument("--env-pool-max-starting", dest="env_startup_workers", type=int)
    parser.add_argument("--inference-batch-size", dest="inference_batch_size", type=int, default=5)
    parser.add_argument("--partial-batch-timeout", dest="partial_batch_timeout", type=float, default=2.0)
    parser.add_argument("--collector-poll-interval", dest="collector_poll_interval", type=float, default=0.05)
    parser.add_argument("--max-startup-retries", dest="max_startup_retries", type=int, default=3)
    parser.add_argument("--max-rollout-retries", dest="max_rollout_retries", type=int, default=3)

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--record-video", dest="record_video", action="store_true")
    parser.add_argument("--no-record-video", dest="record_video", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    parser.add_argument("--history_num", type=int, default=0)
    parser.add_argument("--instruction-type", type=str, default="normal")
    parser.add_argument("--action-chunk-len", type=int, default=4)
    parser.add_argument("--use-keyboard", action="store_true")
    parser.add_argument("--benchmark-profile", type=str, default="")
    parser.add_argument("--output_mode", type=str, default="text_action")
    parser.add_argument("--vlm_client_mode", type=str, default="online")
    parser.add_argument("--system_message_tag", type=str, default="text_action")
    parser.add_argument("--system_message", type=str, default=None)
    parser.add_argument("--model_ips", type=str, default="localhost")
    parser.add_argument("--model_ports", type=str, default="11000")
    parser.add_argument("--raw_action_type", type=str, default="text")
    parser.add_argument("--model_path", type=str, default="CraftJarvis/minecraft-openha-qwen2vl-7b-2509")
    parser.add_argument("--sam_path", type=str, default="facebook/sam2-hiera-base-plus")
    parser.add_argument("--grounding_policy_path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--motion_policy_path", type=str, default="CraftJarvis/Minecraft-Motion_policy-2509")
    parser.add_argument("--grounding_inference_interval", type=int, default=4)
    parser.add_argument("--motion_inference_interval", type=int, default=4)
    parser.add_argument("--maximum_history_length", type=int, default=15)
    parser.add_argument("--LLM_backbone", type=str, default="")
    parser.add_argument("--VLM_backbone", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--enforce_format", action="store_true")
    parser.add_argument("--enforce_prefix", type=str, default="")
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.set_defaults(record_video=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    ensure_openha_importable(args.openha_root)
    rollout(args)
