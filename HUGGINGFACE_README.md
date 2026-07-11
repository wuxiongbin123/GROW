---
tags:
- minecraft
- vision-language-action
- reinforcement-learning
- openha
- vllm
- qwen2-vl
- embodied-ai
library_name: transformers
base_model: Qwen/Qwen2-VL-7B-Instruct
---

# GROW

<p align="center">
  <a href="https://github.com/wuxiongbin123/GROW"><img src="https://img.shields.io/badge/Code-GitHub-blue" alt="GitHub code"></a>
  <a href="https://arxiv.org/abs/2605.20246"><img src="https://img.shields.io/badge/arXiv-2605.20246-b31b1b.svg" alt="arXiv"></a>
</p>

This model repository hosts the released GROW checkpoint for Minecraft agent evaluation. The accompanying evaluation code is available at:

```text
https://github.com/wuxiongbin123/GROW
```

The current release provides the model weights and OpenHA evaluation pipeline. Training code will be released in a later update.

## Quick Start

Clone the evaluation repository with submodules:

```bash
git clone --recurse-submodules https://github.com/wuxiongbin123/GROW.git
cd GROW
```

Create the environment and install dependencies:

```bash
conda create -n grow python=3.10 -y
conda activate grow

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
conda install --channel=conda-forge openjdk=8 -y
pip install -e third_party/OpenHA
```

Run OpenHA evaluation:

```bash
bash scripts/evaluate/openha_rollout.sh
```

By default, the script loads this Hugging Face checkpoint:

```bash
model_local_path="sys555/GROW"
```

You can edit `scripts/evaluate/openha_rollout.sh` to change task patterns, rollout count, batch size, parallel workers, and output directory.

## Evaluation Entry Points

The main evaluation script is:

```text
scripts/evaluate/openha_rollout.sh
```

The Python rollout driver is:

```text
jarvisvla/evaluate/openha_eval.py
```

The evaluator supports OpenHA task patterns such as:

```text
kill_entity:*
craft_item:*
mine_block:*
```

## Notes

- The checkpoint is intended for Minecraft/OpenHA evaluation.
- The rollout pipeline uses vLLM for batched local inference.
- OpenHA is included in the code repository as a git submodule under `third_party/OpenHA`.
- Evaluation requires a working Minecraft/OpenHA runtime, Java 8, CUDA-compatible PyTorch, and GPUs suitable for serving a 7B vision-language model.

## Acknowledgement

We thank the following projects for their excellent work:

- [OpenHA](https://github.com/CraftJarvis/OpenHA)
- [JarvisVLA](https://github.com/CraftJarvis/JarvisVLA)
- [minerl](https://github.com/minerllabs/minerl)
- [malmo](https://github.com/microsoft/malmo)
- [MineStudio](https://github.com/CraftJarvis/MineStudio/tree/master)
- [ROCKET-1](https://github.com/CraftJarvis/ROCKET-1)
- [SAM2](https://github.com/facebookresearch/sam2)

## Citation

If you find GROW useful, please consider citing our paper:

```bibtex
@article{grow2026,
  title   = {GROW: TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```
