import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from models.lenet5 import leNet5
from utils.optim import LAMB, LARS
from utils.conf import ModelConfig

def sel_model(config: ModelConfig) -> torch.nn.Module:
    assert config.name, "Model not selected"
    # Define a LeNet-5 model
    if config.name == 'LeNet':
        model = leNet5(config.learning_rate)
    return model

def sel_device(device: str) -> str:
    assert device, "Invalid selected Device"
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
    
def sel_optimizer(config:ModelConfig, model: torch.nn.Module)-> torch.optim.Optimizer:
    assert config.optimizer, "Optimizer not selected"
    weight_decay = config.weight_decay
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
    elif config.optimizer == "SGDM":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif config.optimizer == "LARS":
        optimizer = LARS(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif config.optimizer == "LAMB":
        optimizer = LAMB(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return optimizer

def sel_loss(config: ModelConfig):
    assert config.loss, "Loss not selected"
    if config.loss == 'CSE':
        criterion = CrossEntropyLoss()
    return criterion

def sel_scheduler(config:ModelConfig, optimizer: torch.optim.Optimizer, len_train_data: int)-> torch.optim.lr_scheduler._LRScheduler:
    assert config.scheduler, "Scheduler not selected"

    warmup_epochs = config.warmup
    total_epochs = config.epochs
    warmup_iters = warmup_epochs * len_train_data

    if config.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=0
        )
    elif config.scheduler == 'WarmUpCosineAnnealingLR':
        # Warmup scheduler 
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_iters
        )
        # Cosine Annealing Scheduler 
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=0
        )
        # combination
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iters]
        )
    elif config.scheduler == 'WarmUpPolynomialDecayLR':
        # Warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_iters
        )
        # Polynomial Decay Scheduler
        polynomial_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                (1 - (step - warmup_iters) / (total_epochs * len_train_data - warmup_iters)) ** 2
            ) if step >= warmup_iters else 1.0
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, polynomial_scheduler],
            milestones=[warmup_iters]
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    return scheduler