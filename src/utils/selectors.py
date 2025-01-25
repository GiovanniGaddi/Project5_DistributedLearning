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



def asc_linear_ls(config: ModelConfig, meta_config: dict)-> None:
    meta_config['ls'] = (int(meta_config['epoch']/(config.epochs-1)*120+1)//2 + 4)
    meta_config['tot_ss'] += meta_config['budget'] // meta_config['ls']

def desc_linear_ls(config: ModelConfig, meta_config: dict)-> None:
    meta_config['ls'] = int(((1.0-meta_config['epoch']/(config.epochs-1))*120+1)//2 + 4)
    meta_config['tot_ss'] += meta_config['budget'] // meta_config['ls']

def asc_binary_ls(config: ModelConfig, meta_config: dict)-> None:
    power2 = (int(meta_config['epoch']/(config.epochs-1)*8+1)//2 + 2)
    meta_config['ls'] = 2**power2
    meta_config['tot_ss'] += meta_config['budget'] // meta_config['ls']

def desc_binary_ls(config: ModelConfig, meta_config: dict)-> None:
    power2 = int(((1.0-meta_config['epoch']/(config.epochs-1))*8+1)//2 + 2)
    meta_config['ls'] = 2**power2
    meta_config['tot_ss'] += meta_config['budget'] // meta_config['ls']

def improvement_ls(config: ModelConfig, meta_config: dict) -> None:
    if meta_config['no_impr_count'] > 4:
        if meta_config['ls']> 4:    
                meta_config['ls'] //= 2
    elif meta_config['no_impr_count'] < 2:
        if meta_config['ls'] < 64:
            meta_config['ls'] *= 2

def loss_ls(config: ModelConfig, meta_config: dict,) -> None: 
    # if at least 2 (could become an hyperparam) losses have been computed
    if meta_config['3loss']:
        # if the loss has increased during the last 2 epochs
        if meta_config['3loss'][-1] > meta_config['3loss'][-2] and meta_config['3loss'][-2] > meta_config['3loss'][-3]: #17m25s - 56.23 - 2 workers | 15m22s - 56.60 - 4 workers | 13m32s - 52.7 - 8 workers
            # halve the local steps (min 4 bu could be as low as 1)
            if meta_config['ls']> 4:    
                meta_config['ls'] //= 2
        # if the loss has decreased over the last 2 epochs
        elif meta_config['3loss'][-1] < meta_config['3loss'][-2] and meta_config['3loss'][-2] < meta_config['3loss'][-3]:
            # double the local steps (max 64 but could be higher)
            if meta_config['ls'] < 64:
                meta_config['ls'] *= 2
        # adapt the synchronization steps to the new value
        meta_config['tot_ss'] += meta_config['budget'] // meta_config['ls']

def sel_dynamic_ls(config: ModelConfig)-> callable:
    if config.work.dynamic == 'LossLS':
        return loss_ls
    elif config.work.dynamic == 'ImprLS':
        return improvement_ls
    elif config.work.dynamic == 'ALin':
        return asc_linear_ls
    elif config.work.dynamic == 'DLin':
        return desc_linear_ls
    elif config.work.dynamic == 'ABin':
        return asc_binary_ls
    elif config.work.dynamic == 'Dlin':
        return desc_binary_ls
    else:
        raise ValueError(f"Unsupported scheduler: {config.work.dynamic}")