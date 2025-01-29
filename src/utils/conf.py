from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Union

class DynamicConfig(BaseModel):
    strategy: str
    n_losses: int

class WorkerConfig(BaseModel):
    sync_steps: int
    local_steps: int
    batch_size: int
    dynamic: Optional[DynamicConfig] = None

class SlowMOConfig(BaseModel):
    learning_rate: float
    momentum: float

class ModelConfig(BaseModel):
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    loss: str
    optimizer: str
    scheduler: str
    weight_decay: float
    slowmo: Optional[SlowMOConfig] = None
    warmup: Optional[int] = None
    patience: Optional[int] = None
    pretrained: Optional[Path] = None
    work: Optional[WorkerConfig] = None
    num_workers: Optional[int] = 0

class ExperimentConfig(BaseModel):
    name: str
    resume: bool
    test_only: Optional[bool] = False
    version: float
    output: Optional[Path] = "results.csv"
    checkpoint_dir: Optional[Path] = "./checkpoints"

class Config(BaseModel):
    model:      ModelConfig
    experiment: ExperimentConfig
