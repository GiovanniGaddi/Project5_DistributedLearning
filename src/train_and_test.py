import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from pathlib import Path
from tqdm import tqdm
from utils.conf import Config, ModelConfig
from utils.parser import Parser
from utils.plot import plot_metrics
from utils.utils import save_checkpoint, load_checkpoint, deepcopy_model, save_to_csv
from utils.load_dataset import load_cifar100
from utils.selectors import sel_device, sel_loss, sel_model, sel_optimizer, sel_scheduler, sel_dynamic_ls

[ ] # Documentation
[ ] # Comments
[ ] # Finishing paper
[ ] # Run guide

def centralized(config: ModelConfig, meta_config: dict,  model: torch.nn.Module, train_loader: list[DataLoader], optimizer: torch.optim.Optimizer, criterion, device: str, pbar: tqdm) -> tuple[list[float], list[float]]:
    """
    Performs centralized training using a single training DataLoader.

    Args:
        config (ModelConfig): The configuration object containing model parameters.
        meta_config (dict): A dictionary containing other configuration.
        model (torch.nn.Module): The model to be trained.
        train_loader (list[DataLoader]): A list containing DataLoaders, with the first (and only) DataLoader used for training
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
        criterion: The function used to compute the loss.
        device (str): The device to run the computations on (e.g., 'cpu' or 'cuda').
        pbar (tqdm): A progress bar instance for tracking training progress (not used at the moment)

    Returns:
        tuple[list[float], list[float]]: A tuple containing:
            - A list containing the average training loss for the epoch.
            - A list containing the training accuracy as a percentage.
    """
    
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader= train_loader[0]
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss and accuracy
        running_loss += loss.item()/len(train_loader)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return [running_loss], [100 * correct/total]

def local(config: ModelConfig, meta_config: dict, model: torch.nn.Module, training_data_splits: DataLoader, optimizer: torch.optim.Optimizer, criterion, device: str, pbar: tqdm) -> tuple[list[float], list[float]]:
    """
    Performs local Stochastic Gradient Descent (SGD) with momentum across multiple workers with optional SlowMo synchronization.

    Args:
        config (ModelConfig): The configuration object containing model parameters.
        meta_config (dict): A dictionary containing other configuration.
        model (torch.nn.Module): The model to be trained.
        training_data_splits (DataLoader): A list of DataLoaders for each worker.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights (unused here, since we have local methods)
        criterion: The loss function used to compute the training loss.
        device (str): The device to run the computations on (e.g., 'cpu' or 'cuda').
        pbar (tqdm): A progress bar instance for tracking training progress.

    Returns:
        tuple[list[float], list[float], list[torch.Tensor], list[list[torch.Tensor]]]: A tuple containing:
            - A list of training losses for each worker.
            - A list of training accuracies for each worker as a percentage.
            - A list of averaged parameters across workers.
            - A nested list of model parameters for each worker.
    """
    
    losses = [0]*config.num_workers
    totals = [0]*config.num_workers
    corrects = [0]*config.num_workers
    
    alpha =  0.0 if config.slowmo is None else config.slowmo.learning_rate
    beta = 0.0 if config.slowmo is None else config.slowmo.momentum
    slowmo_buffer = [torch.zeros_like(param) for param in model.parameters()]    
    
    list_models = [deepcopy_model(model) for _ in range(config.num_workers)]
    list_optimizers = [optim.SGD(list_models[worker_id].parameters(), lr=list_models[worker_id].learning_rate, momentum=0.9, weight_decay=config.weight_decay) for worker_id in range(config.num_workers)]
    iters = [iter(train_loader) for train_loader in training_data_splits]
    
    with tqdm(range(meta_config['budget']), desc='Sync', position=0) as pbar_t:
        for t in pbar_t:
            # workers step
            for worker_id in range(config.num_workers):
                try:
                    inputs, labels = next(iters[worker_id])
                except StopIteration:
                    iters[worker_id] = iter(training_data_splits[worker_id])
                    inputs, labels = next(iters[worker_id])
                # transfer them to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                # reset gradients
                list_optimizers[worker_id].zero_grad()
                # local forward pass
                outputs = list_models[worker_id](inputs)
                loss = criterion(outputs, labels)
                # local backward pass and optimization
                loss.backward()
                list_optimizers[worker_id].step()
                losses[worker_id] += loss.item()/meta_config['budget']
                _, predicted = torch.max(outputs, 1)
                totals[worker_id] += labels.size(0)
                corrects[worker_id] += (predicted == labels).sum().item()

                #pbar.write(f"Work{worker_id+1:02} Loss: {loss}")
            
            # syncronization
            if (t+1)% meta_config['ls'] == 0:
                list_params = [[param for param in list_models[worker_id].parameters()] for worker_id in range(config.num_workers)]

                # Average gradients across all nodes
                avg_params = [torch.mean(torch.stack(worker_param), dim=0) for worker_param in zip(*list_params)]
                slowmo_buffer = [beta*slowmo_param + (1/model.learning_rate)*(param - avg_param) for slowmo_param, avg_param, param in zip(slowmo_buffer, avg_params, model.parameters())]
                # Apply averaged gradients to the global model
                if config.slowmo is None:
                    alpha = model.learning_rate
                with torch.no_grad():
                    for param, slowmo_param in zip(model.parameters(), slowmo_buffer):
                        param.grad = slowmo_param.clone()
                        param.data -= alpha *model.learning_rate * param.grad
                
                if t+meta_config['ls'] > meta_config['budget'] and t+1 != meta_config['budget']:
                    pbar_t.close()
                    break

    return losses, [100*correct/total for correct, total in zip(corrects, totals)], avg_params, list_params


# Function to define the training method
def defineTraining(config: ModelConfig) -> callable:
    if config.num_workers > 0:
        return local
    else:
        return centralized

# Training loop with validation
def train_model(config: Config, train_data: torchvision.datasets, val_data: torchvision.datasets, model: torch.nn.Module, best_model: torch.nn.Module, device: str, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, criterion, checkpoint: dict = None) -> dict:
    
    train_losses = [[] for _ in range(config.model.num_workers if config.model.num_workers > 0 else 1)]
    train_accuracies = [[] for _ in range(config.model.num_workers if config.model.num_workers > 0 else 1)]
    val_losses = []
    val_accuracies = []

    if config.model.patience <= 0:
        patience = config.model.epochs
    else:
        patience = config.model.patience

    #load values form checkpoint
    val_acc = 0.0 if checkpoint is None else checkpoint['val_acc']
    best_acc = 0.0 if checkpoint is None else checkpoint['best_acc']
    start_epoch = 0 if checkpoint is None else checkpoint['start_epoch']

    # Setup for data loaders based on the number of workers
    if config.model.num_workers > 0:
        train_loader = [DataLoader(data, batch_size=config.model.work.batch_size, shuffle=True, drop_last=False) for data in train_data]
    else:
        train_loader = [DataLoader(train_data, batch_size=config.model.batch_size, shuffle=False, drop_last=False)]
    val_loader = DataLoader(val_data, batch_size=config.model.batch_size, shuffle=False)

    train_func = defineTraining(config.model)

    # update meta_config accordingly, setting values such as budget, local steps, total syncs
    if config.model.num_workers > 0:
        meta_config= {
            'budget': max([len(loader) for loader in train_loader]),
            'ls': config.model.work.local_steps,
            'no_impr_count':0,
            '3loss': None,
            'tot_ss': 0 if config.model.work.dynamic else config.model.work.sync_steps*config.model.epochs
        }
    else:
        meta_config = {}
    update_steps = sel_dynamic_ls(config.model)

    # training loop
    postfix = {'Val_Acc': f'{val_acc:.2f}%', 'Best_Acc': f'{best_acc:.2f}%'}
    with tqdm(range(start_epoch, config.model.epochs), desc='Epoch', position=1, postfix=postfix) as pbar:
        for epoch in pbar:
            model.learning_rate = scheduler.get_last_lr()[0]
            print(model.learning_rate)
            model.train()

            #Training phase
            epoch_loss, epoch_acc, avg_params, list_params = train_func(config.model, meta_config, model, train_loader, optimizer, criterion, device, pbar)

            for train_loss, ep_loss in zip(train_losses, epoch_loss):
                train_loss.append(ep_loss)
            for train_acc, ep_acc in zip(train_accuracies, epoch_acc):
                train_acc.append(ep_acc)

            scheduler.step()

            # Validate the model on the validation set
            val_loss, val_acc = validate_model(val_loader, criterion, model)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # update meta_config according to the dynamic local step adjustment method
            if config.model.work.dynamic and len(val_losses)> 2:
                meta_config['3loss'] = val_losses[-3:]
            if config.model.work.dynamic and len(val_losses)> 10:
                meta_config['10loss'] = val_losses[-10:] # TODO generalize history lenght
            if config.model.work.dynamic == 'AvgParamDev':
                meta_config['avg_params'] = avg_params
                meta_config['list_params'] = list_params
            # Save the model with the best accuracy on the validation set
            if val_acc > best_acc:
                best_acc = val_acc
                pbar.write(f"Saved new best model at epoch {epoch+1}")
                best_model = deepcopy_model(model)
                meta_config['no_impr_count'] = 0
            else:
                meta_config['no_impr_count'] += 1

            postfix['Val_Acc'] = f'{val_acc:.2f}%'
            postfix['Best_Acc'] = f'{best_acc:.2f}%'
            pbar.set_postfix(postfix)

            #if early stopping is set
            if meta_config['no_impr_count'] >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            # print local steps for current epoch
            if config.model.num_workers > 0:
                if config.model.work.dynamic:
                    meta_config['epoch'] = epoch
                    update_steps(config.model, meta_config)
                    pbar.write(f'Epoch {meta_config['epoch']}: {meta_config['ls']}')
            
            save_checkpoint(config, epoch, model, best_model, optimizer, scheduler, val_acc, best_acc)

    plot_metrics("distributed" if config.model.num_workers > 0 else "centralized", config, train_losses, train_accuracies, val_losses, val_accuracies)
    return meta_config
    

# Validation function
def validate_model(val_loader: DataLoader, criterion, model: torch.nn.Module) -> tuple[float,float]:
    model.eval()  
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss = running_loss / len(val_loader)
    return loss, accuracy
    

# Evaluate model performance on test set
def evaluate_model(config: ModelConfig, test_data: torchvision.datasets, model: torch.nn.Module) -> float:
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy



if __name__ == '__main__':
    script_dir = Path(__file__).parent

    # setup for config and parser
    yaml_path = script_dir / 'config' / 'Distributed_Lenet.yaml'    
    parser = Parser(yaml_path)
    config, device = parser.parse_args()

    # load dataset
    train_data, val_data, test_data = load_cifar100(config.model)

    # all the selectors
    model = sel_model(config.model)
    device = sel_device(device)
    model = model.to(device)
    loss_function = sel_loss(config.model)
    optimizer = sel_optimizer(config.model, model)
    scheduler = sel_scheduler(config.model, optimizer, len(train_data))
    best_model = deepcopy_model(model)
    checkpoint = None
    if config.experiment.resume:     
        checkpoint = load_checkpoint(config, model, best_model, optimizer, scheduler)
    
    # call functions for train and test (bpth last and best validation accuracy)
    meta_config = train_model(config, train_data, val_data, model, best_model, device, optimizer, scheduler, loss_function, checkpoint)

    model_acc = evaluate_model(config.model,test_data, model)
    print(f'Model Test Accuracy: {model_acc:.2f}%')

    best_model_acc = evaluate_model(config.model,test_data, best_model)
    print(f'Best Model Test Accuracy: {best_model_acc:.2f}%')

    save_to_csv(config, meta_config, model_acc, best_model_acc)
    print(meta_config['tot_ss'])
