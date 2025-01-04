from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Union


class ModelConfig(BaseModel):
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    loss: str
    optimizer: str
    scheduler: str
    pretreined: Optional[Path] = None
    test: bool
    num_nodes: int

class ExperimentConfig(BaseModel):
    name: str
    resume: bool
    version: float

class CheckpointConfig(BaseModel):
    dir: Path
    save_top_k: int

class Config(BaseModel):
    model:      ModelConfig
    experiment: ExperimentConfig
    checkpoint: CheckpointConfig