import os
import csv
import torch
from copy import deepcopy
from utils.conf import Config


#Save checkpoint
def save_checkpoint(config: Config, epoch: int, model: torch.nn.Module, best_model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,val_acc:float, best_acc: float)-> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'best_acc': best_acc
    }
    torch.save(checkpoint, os.path.join(config.checkpoint.dir, f'{config.experiment.version}.pth'))

def load_checkpoint(config:Config, model: torch.nn.Module, best_model:torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler) -> dict:
        checkpoint = torch.load(os.path.join(config.checkpoint.dir, f'{config.experiment.version}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model.load_state_dict(checkpoint['best_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        train_state = {}
        train_state['start_epoch'] = checkpoint['epoch'] + 1
        train_state['val_acc'] = checkpoint['val_acc']
        train_state['best_acc'] = checkpoint['best_acc']

        return train_state

def deepcopy_model(model: torch.nn.Module) -> torch.nn.Module:
    tmp_model = deepcopy(model)
    for tmp_para, para in zip(tmp_model.parameters(), model.parameters()):
        if para.grad is not None:
            tmp_para.grad = para.grad.clone()
    return tmp_model

def save_to_csv(config: Config, model_accuracy: float, best_model_accuracy: float)-> None:
    with open(config.experiment.output, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Check if file is empty (write header only if the file is empty)
        if file.tell() == 0:
            writer.writerow([
                'Model Name', 'Epochs', 'Batch Size', 'Learning Rate', 'Loss', 'Optimizer', 
                'Scheduler', 'Test', 'Warmup', 'Patience', 'Weight Decay','Pretrained',
                'Work Sync Steps', 'Work Local Steps', 'Work Batch Size', 'Num Workers',
                'Slowmo LR', 'SlowMo Momentum', 'Dynamic LS',
                'Model Accuracy', 'Best Model Accuracy'
            ])
        
        # Write model configuration and results to CSV
        row = [
            config.model.name, config.model.epochs, config.model.batch_size, config.model.learning_rate, 
            config.model.loss, config.model.optimizer, config.model.scheduler, config.model.test, 
            config.model.warmup, config.model.patience, config.model.weight_decay, config.model.pretrained, 
            config.model.work.sync_steps if config.model.work else None, 
            config.model.work.local_steps if config.model.work else None, 
            config.model.work.batch_size if config.model.work else None,
            config.model.num_workers,
            config.model.slowmo.learning_rate if config.model.slowmo else config.model.learning_rate,
            config.model.slowmo.momentum if config.model.slowmo else 0,
            config.model.work.dynamic,
            model_accuracy,  # Pass model accuracy
            best_model_accuracy  # Pass best model accuracy
        ]
        writer.writerow(row)
        print(row)