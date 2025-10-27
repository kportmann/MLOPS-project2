import argparse
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import wandb

from src.data_module import GLUEDataModule
from src.model import GLUETransformer


class TrainingConfig(BaseModel):
    """Hyperparameters and runtime settings for training."""

    model_name_or_path: str = "distilbert-base-uncased"
    task_name: str = "mrpc"
    max_epochs: int = 3
    accelerator: str = "auto"
    devices: int | str = 1
    seed: int = 42
    save_checkpoints: bool = False

    learning_rate: float = 2e-5
    warmup_steps: int = 0
    weight_decay: float = 0.0
    train_batch_size: int = 32
    eval_batch_size: int = 32
    max_seq_length: int = 128

    checkpoint_dir: Path = Path("models")
    log_dir: Path = Path("logs")


class LoggerConfig(BaseSettings):
    """Configuration for the W&B logger."""

    model_config = SettingsConfigDict(env_prefix="WANDB_", env_file=".env", extra="ignore")

    project: str = "Default_Project"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] | None = None
    mode: str | None = None  # e.g. "offline"
    log_model: bool | str = False
    api_key: str | None = None


def parse_training_args() -> tuple[TrainingConfig, LoggerConfig]:
    parser = argparse.ArgumentParser(description="Train a GLUE model with PyTorch Lightning.")

    # Training hyperparameters
    parser.add_argument("--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--task_name", default="mrpc")
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models"))
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))

    # Logger options
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--wandb_log_model", default=None)
    parser.add_argument("--wandb_api_key", default=None)

    args = parser.parse_args()
    args_dict: dict[str, Any] = vars(args)

    training_config = TrainingConfig.model_validate(
        {field: args_dict[field] for field in TrainingConfig.model_fields if field in args_dict}
    )
    wandb_overrides = {
        k.removeprefix("wandb_"): v
        for k, v in args_dict.items()
        if k.startswith("wandb_") and v is not None
    }
    logger_config = LoggerConfig(**wandb_overrides)

    return training_config, logger_config


def build_logger(config: LoggerConfig, save_dir: Path) -> WandbLogger:
    return WandbLogger(
        project=config.project,
        entity=config.entity,
        name=config.run_name,
        tags=config.tags,
        mode=config.mode,
        log_model=config.log_model,
        save_dir=str(save_dir),
    )


def login_to_wandb(api_key: str | None) -> None:
    if not api_key:
        raise RuntimeError(
            "Weights & Biases API key not provided. Set the WANDB_API_KEY environment variable, add it to the "
            ".env file, or pass --wandb_api_key to train.py."
        )

    wandb.login(key=api_key, relogin=True)


def run_training(training_config: TrainingConfig, logger_config: LoggerConfig) -> None:
    if training_config.save_checkpoints:
        training_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_config.log_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(training_config.seed)

    login_to_wandb(logger_config.api_key)
    wandb_logger = build_logger(logger_config, training_config.log_dir)
    wandb_logger.experiment.config.update(training_config.model_dump(mode="json"))

    data_module = GLUEDataModule(
        model_name_or_path=training_config.model_name_or_path,
        task_name=training_config.task_name,
        max_seq_length=training_config.max_seq_length,
        train_batch_size=training_config.train_batch_size,
        eval_batch_size=training_config.eval_batch_size,
    )
    data_module.setup("fit")

    model = GLUETransformer(
        model_name_or_path=training_config.model_name_or_path,
        num_labels=data_module.num_labels,
        task_name=data_module.task_name,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        train_batch_size=training_config.train_batch_size,
        eval_batch_size=training_config.eval_batch_size,
        eval_splits=data_module.eval_splits,
    )

    trainer = L.Trainer(
        max_epochs=training_config.max_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        logger=wandb_logger,
        default_root_dir=str(training_config.checkpoint_dir)
        if training_config.save_checkpoints
        else None,
        enable_checkpointing=training_config.save_checkpoints,
    )

    try:
        trainer.fit(model, datamodule=data_module)
    finally:
        wandb.finish()