from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Union


class WorkerConfig(BaseModel):
    sync_steps: int
    local_steps: int
    batch_size: int

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
    slowmo:Optional[SlowMOConfig] = None
    test: Optional[bool] = False
    warmup: Optional[int] = None
    patience: Optional[int] = None
    pretrained: Optional[Path] = None
    work: Optional[WorkerConfig] = None
    num_workers: Optional[int] = 0

class ExperimentConfig(BaseModel):
    name: str
    resume: bool
    version: float
    output: Optional[Path] = "results.csv"

class CheckpointConfig(BaseModel):
    dir: Path
    save_top_k: int

class Config(BaseModel):
    model:      ModelConfig
    experiment: ExperimentConfig
    checkpoint: CheckpointConfig