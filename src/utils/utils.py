import os
import csv
import torch
from copy import deepcopy
from utils.conf import Config
import pickle


#Save checkpoint
def save_checkpoint(config: Config, epoch: int, model: torch.nn.Module, best_model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,val_acc:float, best_acc: float)-> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'learning_rate': model.learning_rate,
        'val_acc': val_acc,
        'best_acc': best_acc
    }
    torch.save(checkpoint, os.path.join(config.experiment.checkpoint_dir, f'{config.experiment.version}.pth'))

def load_checkpoint(config:Config, model: torch.nn.Module, best_model:torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, pretrained: bool = False) -> dict:
        checkpoint = torch.load(os.path.join(config.experiment.checkpoint_dir, f'{config.experiment.version}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model.load_state_dict(checkpoint['best_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.learning_rate = checkpoint['learning_rate']
        train_state = {}
        train_state['start_epoch'] = checkpoint['epoch'] + 1
        train_state['val_acc'] = checkpoint['val_acc']
        train_state['best_acc'] = checkpoint['best_acc']

        return train_state

def load_pretrain(config:Config, model: torch.nn.Module) -> None:
        checkpoint = torch.load(config.model.pretrained, weights_only = True)
        tmp_model = deepcopy(model)
        tmp_model.load_state_dict(checkpoint['model_state_dict'])
        for param, tmp_param in zip(model.parameters(), tmp_model.parameters()):
            param.data = tmp_param.data.clone()

def deepcopy_model(model: torch.nn.Module) -> torch.nn.Module:
    tmp_model = deepcopy(model)
    for tmp_para, para in zip(tmp_model.parameters(), model.parameters()):
        if para.grad is not None:
            tmp_para.grad = para.grad.clone()
    return tmp_model

def save_to_csv(config: Config, meta_config: dict, model_accuracy: float, best_model_accuracy: float)-> None:
    with open(config.experiment.output, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Check if file is empty (write header only if the file is empty)
        if file.tell() == 0:
            writer.writerow([
                'Model Name', 'Epochs', 'Batch Size', 'Learning Rate', 'Loss', 'Optimizer', 
                'Scheduler', 'Test', 'Warmup', 'Patience', 'Weight Decay', 'Pretrained',
                'Total Sync Steps','Work Sync Steps', 'Work Local Steps', 'Work Batch Size', 'Num Workers',
                'Slowmo LR', 'SlowMo Momentum', 'Dynamic LS',
                'Model Accuracy', 'Best Model Accuracy'
            ])
        
        # Write model configuration and results to CSV
        row = [
            config.model.name, config.model.epochs, config.model.batch_size, config.model.learning_rate, 
            config.model.loss, config.model.optimizer, config.model.scheduler, config.model.test, 
            config.model.warmup, config.model.patience, config.model.weight_decay, config.model.pretrained, 
            meta_config['tot_ss']if config.model.work else None,
            config.model.work.sync_steps if config.model.work else None, 
            config.model.work.local_steps if config.model.work else None, 
            config.model.work.batch_size if config.model.work else None,
            config.model.num_workers,
            config.model.slowmo.learning_rate if config.model.slowmo else config.model.learning_rate,
            config.model.slowmo.momentum if config.model.slowmo else 0,
            config.model.work.dynamic if config.model.work else False,
            model_accuracy,  # Pass model accuracy
            best_model_accuracy  # Pass best model accuracy
        ]
        writer.writerow(row)
        print(row)

def save_to_pickle(config: dict, meta_config: dict) -> None:
    strategy = ""
    if config.model.work.dynamic:
        strategy = f"strat-{config.model.work.dynamic.strategy}_nLosses-{config.model.work.dynamic.n_losses}_"
    slowmo = ""
    if config.model.slowmo:
        slowmo = f"slr-{config.model.slowmo.learning_rate}_sm-{config.model.slowmo.momentum}_"
    filepath = f"{strategy}{slowmo}K-{config.model.num_workers}.pkl"
    datatypes = {
        "val_acc": "val_accuracies", 
        "val_loss": "val_losses", 
        "train_acc": "train_accuracies", 
        "train_loss": "train_losses", 
        "train_H": "train_local_steps", 
        "train_lr": "train_learning_rates"
    }
    for directory, index in datatypes.items():
        if not os.path.exists(f"pickles/{directory}"):
            os.makedirs(f"pickles/{directory}")
        file = open(f"pickles/{directory}/{filepath}", "wb+")
        pickle.dump(meta_config[index], file)
        file.close()
