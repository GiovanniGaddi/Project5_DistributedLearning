import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import math
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

    # Main selection
    if config.scheduler == 'CosineAnnealingLR':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=0
        )
    elif config.scheduler == 'PolynomialDecayLR':
        main_scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_epochs * len_train_data, power=2.0
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    # Full
    warmup_scheduler = create_warmup_scheduler(optimizer, 0.1, warmup_epochs, warmup_iters)

    if warmup_scheduler:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = main_scheduler
    print(scheduler)
    return scheduler


def create_warmup_scheduler(optimizer: torch.optim.Optimizer, start: float, warmup_epochs: int, warmup_iters: int):
    if warmup_epochs > 0:
        return optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start, total_iters=warmup_epochs
        )
    return None


# Methods for dynamic local steps scheduling

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
    meta_config['tot_ss'] += meta_config['budget'] // meta_config['ls']

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
    meta_config['ss'] = sync_steps
    meta_config['tot_ss'] += sync_steps


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
    meta_config['ss'] = sync_steps
    meta_config['tot_ss'] += sync_steps

def avg_param_dev_ls(config: ModelConfig, meta_config: dict, threshold=0.5): #threshold to be determined
    """
    Update sync steps (T) and local steps (H) based on avg params deviation 

    Args:
        threshold: to decide whether to update
    """
    K = config.num_workers
    H = config.work.local_steps
    total_steps = meta_config['budget']
    T = total_steps/H

    num_params = len(meta_config['avg_params'])
    total_deviation = 0.0

    # Compute deviation w.r.t. average pparams
    for k in range(K):
        worker_deviation = 0.0
        for param_grad, avg_grad in zip(meta_config['list_params'][k], meta_config['avg_params']):
            worker_deviation += torch.norm(param_grad - avg_grad).item()  # applying norm
        total_deviation += worker_deviation / num_params

    avg_deviation = total_deviation / K

    print(avg_deviation)

    # Updates based on avg dev
    if avg_deviation < threshold:  # if gradient is steady, increase H
        H = min(H * 2, total_steps//2)  # double H
        T = max(2, total_steps//H)  # update T to keep total_steps constant
    elif avg_deviation > threshold*2:  # otherwise reduce H (so more sync)
        H = max(4, H // 2)  # half H
        T = max(4, total_steps // H)  # update T to keep total_steps constant

    meta_config['ls'] = H
    meta_config['ss'] = T
    meta_config['tot_ss'] += T

def avg_loss_ls(config: ModelConfig, meta_config: dict) -> None:
    if meta_config['loss_history']: #16m03s - 57.19 - 2 workers | 13m40s - 56.50 - 4 workers | 14m58s / 13m14s - 52.34 / 52.25 - 8 workers 
        # compute last 5 epochs average loss
        new_avg = sum(meta_config['loss_history'][-config.model.work.dynamic.n_losses//2:])/(config.model.work.dynamic.n_losses//2)
        # compute second-to-last group of 5 epochs average loss
        old_avg = sum(meta_config['loss_history'][-config.model.work.dynamic.n_losses:-config.model.work.dynamic.n_losses//2])/(config.model.work.dynamic.n_losses//2)
        if new_avg > old_avg and config.model.work.local_steps > 4:
            # half the local steps (min 4 but could be as low as 1)
                meta_config['ls'] //= 2
        elif new_avg < old_avg and config.model.work.local_steps < 64:
            # double the local steps (max 64 but could be higher)
                meta_config['ls'] *= 2
    # adapt the synchronization steps to the new value
    meta_config['tot_ss'] += meta_config['budget'] // meta_config['ls']

def sel_dynamic_ls(config: ModelConfig)-> callable:
    if config.work.dynamic.strategy == 'LossLS':
        return loss_ls
    if config.work.dynamic.strategy == 'AvgLossLS':
        return avg_loss_ls
    elif config.work.dynamic.strategy == 'RevCosAnn':
        return reverse_cosine_annealing_ls
    elif config.work.dynamic.strategy == 'Sigmoid':
        return sigmoid_based_ls
    elif config.work.dynamic.strategy == 'AvgParamDev':
        return avg_param_dev_ls
    elif config.work.dynamic.strategy == 'ImprLS':
        return improvement_ls
    elif config.work.dynamic.strategy == 'ALin':
        return asc_linear_ls
    elif config.work.dynamic.strategy == 'DLin':
        return desc_linear_ls
    elif config.work.dynamic.strategy == 'ABin':
        return asc_binary_ls
    elif config.work.dynamic.strategy == 'Dlin':
        return desc_binary_ls
    else:
        raise ValueError(f"Unsupported local steps scheduler: {config.work.dynamic.strategy}")
