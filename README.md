# MLOPS Project 2

Minimal instructions for anyone cloning this repository and reproducing the training run.

## Requirements
- Docker (Desktop or CLI)
- Weights & Biases account + API key (`WANDB_API_KEY`)

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

The container executes the baked-in command from the Dockerfile: (these are the best Hyperparameters I found in Project1)
```
uv run main.py --wandb_project MLOPS_Project2 --wandb_run_name Project1_Best_run_recreation_dockerfile \
  --max_epochs 3 --learning_rate 3e-5 --train_batch_size 64 --eval_batch_size 64 --weight_decay 0.0001 \
  --no-save_checkpoints
```

Override hyperparameters by appending a custom command:
```bash
docker run --rm -e WANDB_API_KEY=<key> mlops-project2 \
  uv run main.py --wandb_project MLOPS_Project2 \
  --wandb_run_name Project1_Best_run_recreation_codespaces \
  --max_epochs 3 --learning_rate 3e-5 \
  --train_batch_size 64 --eval_batch_size 64 \
  --weight_decay 0.0001 --no-save_checkpoints
```

## 4. Codespaces
Clone the repo, run the same `docker build` and `docker run` commands. No code changes required.

## 5. Notes
- PyTorch 2.9.0 with CUDA 12 support is installed. It automatically falls back to CPU execution when no GPU is available. For GPU usage in Docker, run with `--gpus all` flag and ensure NVIDIA Container Toolkit is installed on the host.
- Checkpoints are disabled by default (`--no-save_checkpoints`) to save space and time in containerized runs. Metrics are logged to W&B.
- Artifacts (`logs/`, `wandb/`) stay on the host; they are excluded from the image by `.dockerignore`.
- Docker Playground has insufficient disk space for this project. Use GitHub Codespaces instead.
