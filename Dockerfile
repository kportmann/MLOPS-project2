FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy project
COPY . .

CMD ["uv", "run", "main.py", "--wandb_project", "MLOPS_Project2", "--wandb_run_name", "Project1_Best_run_recreation_dockerfile", "--max_epochs", "3", "--learning_rate", "3e-5", "--train_batch_size", "64", "--eval_batch_size", "64", "--weight_decay", "0.0001"]