# Anonymous Code Repository (Minecraft VLA)

This repository contains code for an anonymous conference submission.  
It implements a Vision-Language-Action (VLA) agent for first-person Minecraft and provides training and evaluation scripts for single-task and short composite rollouts.

> **Anonymity note:** This repository is anonymized. Please do not attempt to de-anonymize the authors.

---

## Repository layout

- `minestudio/external/cab_vla/` — Core implementation of the proposed VLA (training + inference).
- `minestudio/models/cab_vla/` — MineStudio/MineRL policy wrapper for running the proposed VLA inside the simulator.
- `eval/` — Evaluation code and scripts (E1/E2 rollouts and metrics).

---

## Contents (what you can run)

- **Evaluation rollouts**
  - `E2_single` (single-task)
  - `E2_composite` (two-step composites)
- **Training (optional)** from public data (VPT demonstrations) with a public pretrained backbone (`paligemma-3b-pt-224`).
- Docker-based environment setup (recommended) and pip-based setup (optional).

> Checkpoints are provided to reviewers via supplemental material and will be released upon acceptance.

---

## System requirements

- OS: Linux recommended
- GPU: **≥ 13 GB** VRAM for inference
- Docker (recommended) or a Python environment (pip option)

---

## Installation (recommended: Docker)

We strongly recommend Docker to reduce setup burden and improve reproducibility.

### Build

~~~bash
docker build ./ -t cab
~~~

### Run

~~~bash
docker run --gpus all -it \
  -v {your_model_dir}:/model \
  -v {your_out_dir}:/out_dir \
  --ulimit memlock=-1 --shm-size=16gb \
  cab:latest /bin/bash
~~~

**Inside the container**, set:

~~~bash
export PRISMATIC_MODEL_DIR=/model
~~~

---

## Installation (optional: pip)

### System dependencies

#### Java

~~~bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk
sudo update-alternatives --config java
~~~

#### xvfb / OpenGL

~~~bash
sudo apt-get update
sudo apt-get install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev
~~~

### Python packages

From the repository root:

~~~bash
pip install -e minestudio/external/cab_vla
pip install -e .
~~~

Set the model directory (required):

~~~bash
export PRISMATIC_MODEL_DIR={your_model_dir}
~~~

---

## Model setup

1. Download / place model files under `{your_model_dir}` (mounted to `/model` in Docker):
   - `prism-paligemma-3b-pt-224_cab`
   - `paligemma-3b-pt-224`
   - `paligemma-3b-pt-224_vision_tower.pth`

2. Ensure `PRISMATIC_MODEL_DIR` points to the directory containing the above files:
   - Docker: `export PRISMATIC_MODEL_DIR=/model`
   - pip: `export PRISMATIC_MODEL_DIR={your_model_dir}`

---

## Model checkpoints

Model checkpoints are provided to reviewers via the supplemental material.  
Please extract the provided `.tar.gz` files into `{your_model_dir}` (or `/model` inside Docker).

---

## Inference (evaluation rollouts)

All commands below write outputs under `/out_dir/...` (Docker) or `{your_out_dir}/...` (pip).

### E2_single (single-task rollouts)

~~~bash
cd eval
CUDA_VISIBLE_DEVICES=0 bash scripts/rollout-E2_single.sh /out_dir/E2_single
~~~

### E2_composite (two-step composite rollouts)

~~~bash
cd eval
CUDA_VISIBLE_DEVICES=0 bash scripts/rollout-E2_composite.sh /out_dir/E2_composite
~~~

---

## Training (optional)

We provide training code in this repository. Training is significantly more compute-intensive than evaluation (multi-GPU recommended).

### Prerequisites
Before training, please prepare the following:
- **Training data:** the public **VPT Minecraft Demonstration Dataset** (not redistributed in this repository).
- **Pretrained backbone:** the public checkpoint **paligemma-3b-pt-224** (place under PRISMATIC_MODEL_DIR).

### Train

~~~bash
cd minestudio/external/cab_vla
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc-per-node 2 train_vla.py --data_dir {your_data_dir} --run_root_dir {your_run_dir}
~~~

---

## Notes on licensing and anonymity

- This repository incorporates open-source components. License notices and attributions are included in the source tree in accordance with their licenses.
- No identifying information is intentionally included in this repository.
