import time
import json
import random
import copy
import re
import os
from pathlib import Path
from collections import Counter
from typing import Literal

import numpy as np
import torch
from PIL import Image
from rich import print

# This rollout path is PyTorch/vLLM-only. Disabling TF avoids
# transformers probing tensorflow in environments with incompatible protobuf.
os.environ.setdefault("USE_TF", "0")

# Transformers & vLLM
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# JarvisVLA / Project Specific Imports
from jarvisvla.inference import action_mapping
from jarvisvla.inference.action_converter import ActionFromLLMConverter
from jarvisvla.utils.file_utils import load_json_file

import sys



class VLLM_AGENT:
    def __init__(self, checkpoint_path, 
                 history_num=0, action_chunk_len=4, bpe=0,
                 instruction_type: Literal['simple', 'recipe', 'normal'] = 'normal',
                 temperature=1.0, 
                 use_keyboard=False,
                 action_reuse_times=0,
                 _obs_shape=(640, 360, 3),
                 can_print=False):
        
        # 1. 加载 Tokenizer
        print(f"[INFO] Loading tokenizer from {checkpoint_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,  
            trust_remote_code=True,
        )
        
        self.LLM_backbone = "qwen2_vl" 
        
        # 2. 初始化本地 vLLM 引擎
        # TP=1 避免多头注意力切分问题，对于7B模型单卡显存足够
        print(f"[INFO] Initializing local vLLM engine with TP=1...")
        self.llm = LLM(
            model=checkpoint_path,
            tensor_parallel_size=1, 
            trust_remote_code=True,
            gpu_memory_utilization=0.90, 
            max_model_len=32768, 
            enable_prefix_caching=True
        )

        # 3. 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=1024,
            top_p=0.99,
            skip_special_tokens=False,  # 关键：必须保留特殊token以解析动作
            # repetition_penalty=1.15,
            # logprobs=1  # 设置为1以获取每个生成token的对数概率
        )

        # 动作编解码器
        self.action_tokenizer = action_mapping.OneActionTokenizer(tokenizer_type=self.LLM_backbone)

        self.use_keyboard = use_keyboard
        self._obs_shape = _obs_shape
        
        # 加载 prompt 库 (保留原逻辑以免报错，虽然 forward 中不再依赖)
        self.prompt_library = load_json_file(Path(__file__).parent/"assets"/"instructions.json")
        
        self.actions = []
        self.action_chunk_len = action_chunk_len
        self.history_num = history_num # 暂时未在 forward 中启用 history，以对齐单步训练
        self.history = []
        
        self.instruction_type = instruction_type

        # 辅助转换器
        self.action_converter = ActionFromLLMConverter(
            map_camera_to_11=False,
            return_numpy=True
        )
        self.can_print = can_print
        self.have_print = False

    def _build_messages(self, task_text, image_pil):
        return [
            {
                "role": "system",
                "content": f"""You are an experienced Minecraft player. Based on the current game screen, plan the next 200ms of actions, composed of 4 steps, each spaced 50ms apart. Each action begins at its execution time and lasts for 50ms until the next step starts.

# Output Format
<|reserved_special_token_178|>...<|reserved_special_token_179|><|reserved_special_token_178|>...<|reserved_special_token_179|><|reserved_special_token_178|>...<|reserved_special_token_179|><|reserved_special_token_178|>...<|reserved_special_token_179|>

# Instructions
1. The output consists of exactly 4 action steps. Each step is marked by a start token <|reserved_special_token_178|> and an end token <|reserved_special_token_179|>, with the specific action command placed between them.
2. Output ONLY the plain string in the exact format above. Do not add line breaks, quotes, or any additional text. Your current task is: {task_text}"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Your observation: "},
                    {"type": "image", "image": image_pil}
                ]
            }
        ]

    def _build_llm_input(self, observation, instruction):
        image_pil = Image.fromarray(observation.astype('uint8'), 'RGB')
        messages = self._build_messages(instruction, image_pil)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return {
            "prompt": prompt_text,
            "multi_modal_data": {
                "image": image_pil
            },
        }

    def _decode_output_ids(self, outputs_ids):
        actions = []
        if self.LLM_backbone in {"qwen2_vl", "Qwen2-VL-7B-Instruct"}:
            actions = self.action_tokenizer.decode(outputs_ids)
        len_action = min(self.action_chunk_len, len(actions))
        return actions[:len_action]

    def generate_action_chunks_batch(self, observations, instructions, verbos=True):
        llm_inputs = []
        for observation, instruction in zip(observations, instructions):
            if verbos or self.can_print:
                print(f'[INFO] Current Task: {instruction}')
            llm_inputs.append(self._build_llm_input(observation, instruction))

        request_output = self.llm.generate(llm_inputs, self.sampling_params)

        batch_actions = []
        raw_responses = []
        for output in request_output:
            output_obj = output.outputs[0]
            outputs_ids = output_obj.token_ids
            actions = self._decode_output_ids(outputs_ids)
            raw_text = output_obj.text
            if not actions:
                print("[WARNING] No valid actions parsed. Returning no_action.")
                actions = ["no_action"]
            batch_actions.append(actions)
            raw_responses.append(raw_text)
            if verbos:
                print(f'[DEBUG] Parsed Actions: {actions}')

        return batch_actions, raw_responses

    def reset(self):
        """重置 Agent 状态"""
        self.history = []
        self.actions = []
        print("[INFO] Agent reset.")

    def forward(self, observations, instructions, verbos=True, need_crafting_table=False):
        """
        核心推理步骤：
        严格模仿训练数据的结构：User(Text) -> Assistant(Empty) -> User(Image) -> Assistant(Gen)
        """
        
        # --- 1. 处理缓存的动作块 (Action Chunking) ---
        if self.actions:
            if verbos:
                print(f'[DEBUG] Cached actions: {self.actions}')
            if len(self.actions) >= 1:
                return self.actions.pop(0)
            else:
                # 理论上不应进入这里，但作为防守编程
                action = self.actions[0]
                self.actions = []
                return action
        
        batch_actions, raw_responses = self.generate_action_chunks_batch(
            observations,
            instructions,
            verbos=verbos,
        )
        self.actions = batch_actions[0]
        self._response = raw_responses[0]

        if self.actions:
            return self.actions.pop(0)
        else:
            print("[WARNING] No valid actions parsed. Returning no_action.")
            return "no_action"

    # --- 保留旧的辅助函数接口，防止外部调用报错，但不再使用 ---
    def rule_based_instruction(self, env_prompt:str):
        pass 
    def create_basic_instruction(self, env_prompt:str):
        pass
    def get_recipe_item_name(self, ingredient:dict):
        pass
    def create_recipe_prompt_from_library(self, item_name:str):
        pass
    def create_recipe_prompt(self, env_prompt:str, method:str="crafting_table"):
        pass
    def create_instruction(self, env_prompt, method):
        pass
    def create_thought(self, env_prompt):
        pass
    def split_camera_movement(self, action_dict, reuse_times):
        pass
