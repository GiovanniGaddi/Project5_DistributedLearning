import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import math
from models.lenet5 import leNet5
from utils.optim import LAMB, LARS
from utils.conf import ModelConfig

def sel_model(config: ModelConfig) -> torch.nn.Module:
    assert config.name, "Model not selected"
    if config.name == 'LeNet':
        model = leNet5(config.learning_rate)
    else:
        raise ValueError(f"Unsupported model: {config.name}")
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
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "SGDM":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    elif config.optimizer == "LARS":
        optimizer = LARS(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    elif config.optimizer == "LAMB":
        optimizer = LAMB(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return optimizer

def sel_loss(config: ModelConfig):
    assert config.loss, "Loss not selected"
    if config.loss == 'CSE':
        criterion = CrossEntropyLoss()
    else: 
        raise ValueError(f"Unsupported Loss Function: {config.loss}")
    return criterion

def sel_scheduler(config:ModelConfig, optimizer: torch.optim.Optimizer, len_train_data: int)-> torch.optim.lr_scheduler._LRScheduler:
    assert config.scheduler, "Scheduler not selected"

    warmup_iters = config.warmup * len_train_data

    # Main selection
    if config.scheduler == 'CosineAnnealingLR':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs - config.warmup, eta_min=0
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    # Full
    warmup_scheduler = create_warmup_scheduler(optimizer, 0.1, config.warmup, warmup_iters)

    if warmup_scheduler:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.warmup]
        )
    else:
        scheduler = main_scheduler

    return scheduler


def create_warmup_scheduler(optimizer: torch.optim.Optimizer, start: float, warmup_epochs: int, warmup_iters: int) -> torch.optim.lr_scheduler._LRScheduler:
    if warmup_epochs > 0:
        return optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start, total_iters=warmup_epochs
        )
    return None

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                       Methods for dynamic local steps scheduling
#------------------------------------------------------------------------------------------------------------------------------------------------------------


def asc_linear_ls(config: ModelConfig, meta_config: dict)-> None:
    '''
    Linearly maps the epoch to the range of local steps: from 4 to 64 
    '''
    meta_config['ls'] = (int(meta_config['epoch']/(config.epochs-1)*120+1)//2 + 4)
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']

def desc_linear_ls(config: ModelConfig, meta_config: dict)-> None:
    '''
    Linearly maps the epoch to the range of local steps: from 64 to 4 
    '''
    meta_config['ls'] = int(((1.0-meta_config['epoch']/(config.epochs-1))*120+1)//2 + 4)
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']

def asc_binary_ls(config: ModelConfig, meta_config: dict)-> None:
    '''
    Linearly maps the epoch to the power of 2 of local steps: from 2**2=4 to 2**6=64 
    '''
    power2 = (int(meta_config['epoch']/(config.epochs-1)*8+1)//2 + 2)
    meta_config['ls'] = 2**power2
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']

def desc_binary_ls(config: ModelConfig, meta_config: dict)-> None:
    '''
    Linearly maps the epoch to the power of 2 of local steps: from 2**6=64 to 2**2=4 
    '''
    power2 = int(((1.0-meta_config['epoch']/(config.epochs-1))*8+1)//2 + 2)
    meta_config['ls'] = 2**power2
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']

def improvement_ls(config: ModelConfig, meta_config: dict) -> None:
    '''
    In case of improvement (validation accuracy increase), no improvement = 0 reduce synchronisations steps increasing the power of 2 of the local steps
    In case of no improvement for at least 4 Epochs, no improvement > 4 increase synchronisations steps decreasing the power of 2 of local steps
    local steps have a maximum of 64 steps and a minimum of 4
    '''
    if meta_config['no_impr_count'] > 3:
        if meta_config['ls']> 4:    
                meta_config['ls'] //= 2
    elif meta_config['no_impr_count'] < 1:
        if meta_config['ls'] < 64:
            meta_config['ls'] *= 2

def loss_ls(config: ModelConfig, meta_config: dict,) -> None: 
    # if at least 2 (could become an hyperparam) losses have been computed
    if meta_config['loss_history']:
        # if the loss has increased during the last 2 epochs
        if meta_config['loss_history'][-1] > meta_config['loss_history'][-2] and meta_config['loss_history'][-2] > meta_config['loss_history'][-3]: #17m25s - 56.23 - 2 workers | 15m22s - 56.60 - 4 workers | 13m32s - 52.7 - 8 workers
            # half the local steps (min 4 but could be as low as 1)
            if meta_config['ls']> 4:    
                meta_config['ls'] //= 2
        # if the loss has decreased over the last 2 epochs
        elif meta_config['loss_history'][-1] < meta_config['loss_history'][-2] and meta_config['loss_history'][-2] < meta_config['loss_history'][-3]:
            # double the local steps (max 64 but could be higher)
            if meta_config['ls'] < 64:
                meta_config['ls'] *= 2
    # adapt the synchronization steps to the new value
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']

def sigmoid_based_ls(config: ModelConfig, meta_config: dict, midpoint: int = 75, steepness: float = 0.5)-> None:
    """
    Compute the number of local steps (H) to take using a sigmoid function.
    
    Args:
        midpoint (float): The epoch at which the transition occurs most rapidly.
        steepness (float): Controls how sharp the transition is.
    """
    total_steps = meta_config['budget']
    epoch = meta_config['epoch']
    H_max = total_steps/2
    H_min = total_steps/22 #might be a parameter

    # Sigmoid function for smooth transition
    sigmoid_value = 1 / (1 + math.exp(-steepness * (epoch - midpoint)))
    H = H_min + (H_max - H_min) * sigmoid_value
    H = max(H_min, min(H_max, round(H)))  # Clamp H within bounds

    # Compute synchronization steps
    sync_steps = total_steps // H
    remainder = total_steps % H

    if remainder > 0 and (remainder / total_steps) <= 0.05:
        sync_steps += 1
        H = math.ceil(total_steps / sync_steps)
    else:
        H = math.floor(total_steps / sync_steps)

    meta_config['ls'] = H
    # adapt the synchronization steps to the new value
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']

def reverse_cosine_annealing_ls(config: ModelConfig, meta_config: dict)-> None: 
    """Compute the inverted Cosine Annealing value for H with exact total_steps."""
    total_steps = meta_config['budget']
    t = meta_config['epoch']
    T = config.epochs
    H_max = total_steps/2
    H_min = total_steps/22 #might be a parameter

    H = H_max - (H_max - H_min) / 2 * (1 + math.cos(math.pi * t / T))
    H = max(H_min, min(H_max, round(H)))  

    sync_steps = total_steps // H
    remainder = total_steps % H

    if remainder > 0 and (remainder / total_steps) <= 0.05:
        sync_steps += 1
        H = math.ceil(total_steps / sync_steps)
    else:
        H = math.floor(total_steps / sync_steps)

    meta_config['ls'] = H
    # adapt the synchronization steps to the new value
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']


def avg_loss_ls(config: ModelConfig, meta_config: dict) -> None:
    if meta_config['loss_history']: #16m03s - 57.19 - 2 workers | 13m40s - 56.50 - 4 workers | 14m58s / 13m14s - 52.34 / 52.25 - 8 workers 
        # compute last 5 epochs average loss
        new_avg = sum(meta_config['loss_history'][-config.work.dynamic.n_losses//2:])/(config.work.dynamic.n_losses//2)
        # compute second-to-last group of 5 epochs average loss
        old_avg = sum(meta_config['loss_history'][-config.work.dynamic.n_losses:-config.work.dynamic.n_losses//2])/(config.work.dynamic.n_losses//2)
        if new_avg > old_avg and meta_config['ls'] > 4:
            # half the local steps (min 4 but could be as low as 1)
                meta_config['ls'] //= 2
        elif new_avg < old_avg and meta_config['ls'] < 64:
            # double the local steps (max 64 but could be higher)
                meta_config['ls'] *= 2
    # adapt the synchronization steps to the new value
    meta_config['tot_ss'] += (meta_config['budget'] + meta_config['ls'] - 1)// meta_config['ls']



def sel_dynamic_ls(config: ModelConfig)-> callable:
    if config.work.dynamic.strategy == 'LossLS':
        return loss_ls
    if config.work.dynamic.strategy == 'AvgLossLS':
        return avg_loss_ls
    elif config.work.dynamic.strategy == 'RevCosAnn':
        return reverse_cosine_annealing_ls
    elif config.work.dynamic.strategy == 'Sigmoid':
        return sigmoid_based_ls
    elif config.work.dynamic.strategy == 'ImprLS':
        return improvement_ls
    elif config.work.dynamic.strategy == 'ALin':
        return asc_linear_ls
    elif config.work.dynamic.strategy == 'DLin':
        return desc_linear_ls
    elif config.work.dynamic.strategy == 'ABin':
        return asc_binary_ls
    elif config.work.dynamic.strategy == 'DBin':
        return desc_binary_ls
    else:
        raise ValueError(f"Unsupported local steps scheduler: {config.work.dynamic.strategy}")
