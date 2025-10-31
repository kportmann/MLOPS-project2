# MLOPS Project 2 – Containerized GLUE Training

Minimal instructions for anyone cloning this repository and reproducing the training run.

## Requirements
- Docker (Desktop or CLI)
- Weights & Biases account + API key (`WANDB_API_KEY`)
- Optional: Python 3.12 and [`uv`](https://github.com/astral-sh/uv) for local execution without Docker

## 1. Local Run (optional)
```bash
python -m venv .venv
source .venv/bin/activate
uv sync          # or: pip install -r requirements.txt
export WANDB_API_KEY=...
uv run main.py --help
uv run main.py --wandb_project MLOPS_Project2 --max_epochs 3 --learning_rate 3e-5 \
    --train_batch_size 64 --eval_batch_size 64 --weight_decay 1e-4
```
All CLI flags are defined in `src/train.py`.

## 2. Build Docker Image
```bash
docker build -t mlops-project2 .
```
The image installs dependencies with `uv sync` and copies the source code. `.dockerignore` keeps local venvs, logs, and secrets out of the build context.

## 3. Run Docker Container
Provide your W&B credentials via env vars or an `.env` file on the host.

```bash
# Using inline env variable
docker run --rm -e WANDB_API_KEY=... mlops-project2

# Or using a host .env file (not copied into the image)
docker run --rm --env-file .env mlops-project2
```

The container executes the baked-in command from the Dockerfile:
```
uv run main.py --wandb_project MLOPS_Project2 --wandb_run_name Project1_Best_run_recreation_dockerfile \
  --max_epochs 3 --learning_rate 3e-5 --train_batch_size 64 --eval_batch_size 64 --weight_decay 0.0001 \
  --no-save_checkpoints
```

Override hyperparameters by appending a custom command:
```bash
docker run --rm -e WANDB_API_KEY=... mlops-project2 \
  uv run main.py --max_epochs 1 --train_batch_size 32
```

## 4. Codespaces / Docker Playground
Clone the repo, run the same `docker build` and `docker run` commands. No code changes required.

## 5. Notes
- PyTorch is installed with CPU runtimes. GPU usage requires rebuilding the image with CUDA-enabled wheels and running with `--gpus all` on a GPU host.
- Checkpoints are disabled by default (`--no-save_checkpoints`) to save space and time in containerized runs. Metrics are logged to W&B.
- Artifacts (`logs/`, `wandb/`) stay on the host; they are excluded from the image by `.dockerignore`.
