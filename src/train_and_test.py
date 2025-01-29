import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from pathlib import Path
from tqdm import tqdm
from utils.conf import Config, ModelConfig
from utils.parser import Parser
from utils.plot import plot_metrics
from utils.utils import save_checkpoint, load_checkpoint, deepcopy_model, save_to_csv, save_to_pickle, load_pretrain
from utils.load_dataset import load_cifar100
from utils.selectors import sel_device, sel_loss, sel_model, sel_optimizer, sel_scheduler, sel_dynamic_ls

# [ ] # Documentation
# [ ] # Comments
# [ ] # Finishing paper
# [ ] # Run guide

# [x] # Graph best cases (Silvano)
# [x] # Local Steps Graph (Grasso che cola ma mezza fatta, Silvano)
# [ ] # Graph centralized (Gio)
# [ ] # Add linear Experiments (Gio)
# [ ] # Slowmo beta (Silvano)
# [ ] # Tabelle Results (Silvano, Nicolò)
# [x] # Check save/load checkpoint (Gio)
# [x] # Check pretrained (Gio)
# [x] # Check only-Test (Gio)
# [x] # Skip fix (Gio)
# [x] # Parser (Gio)
# [ ] # ReadME (Nicolò)
# [ ] # Commenti
# [ ] # Public the github
# [ ] # Final check


last_step_skip = False


def centralized(config: ModelConfig, meta_config: dict,  model: torch.nn.Module, train_loader: list[DataLoader], optimizer: torch.optim.Optimizer, criterion, device: str, pbar: tqdm) -> tuple[list[float], list[float]]:
    ''' Centralized Model Training Function '''
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader= train_loader[0]
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset Gradiants
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


def localSDG(config: ModelConfig, meta_config: dict, model: torch.nn.Module, training_data_splits: DataLoader, optimizer: torch.optim.Optimizer, criterion, device: str, pbar: tqdm) -> tuple[list[float], list[float]]:
    ''' Distributed Model Training Function'''
    losses = [0]*config.num_workers
    totals = [0]*config.num_workers
    corrects = [0]*config.num_workers
    
    # SlowMo Initialization
    alpha =  1.0 if config.slowmo is None else config.slowmo.learning_rate
    beta = 0.0 if config.slowmo is None else config.slowmo.momentum
    slowmo_buffer = [torch.zeros_like(param) for param in model.parameters()]    
    
    # Create Workers and optimizer based on the model
    list_models = [deepcopy_model(model) for _ in range(config.num_workers)]
    list_optimizers = [optim.SGD(list_models[worker_id].parameters(), lr=list_models[worker_id].learning_rate, momentum=0.9, weight_decay=config.weight_decay) for worker_id in range(config.num_workers)]
    # Create Iterator for the Dataset
    iters = [iter(train_loader) for train_loader in training_data_splits]
    
    # Last Synch flag Variable
    updated = False

    with tqdm(range(meta_config['budget']), desc='Sync', position=0) as pbar_t:
        for t in pbar_t:
            # Workers step
            for worker_id in range(config.num_workers):
                try:
                    inputs, labels = next(iters[worker_id])
                except StopIteration:
                    iters[worker_id] = iter(training_data_splits[worker_id])
                    inputs, labels = next(iters[worker_id])
                # Transfer them to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                # Reset gradients
                list_optimizers[worker_id].zero_grad()
                # Local forward pass
                outputs = list_models[worker_id](inputs)
                loss = criterion(outputs, labels)
                # Local backward pass and optimization
                loss.backward()
                list_optimizers[worker_id].step()
                # Track local loss and accuracy
                losses[worker_id] += loss.item()/meta_config['budget']
                _, predicted = torch.max(outputs, 1)
                totals[worker_id] += labels.size(0)
                corrects[worker_id] += (predicted == labels).sum().item()

            # Set the update Flag to Model not updated
            updated = False

            # Syncronization
            if (t+1)% meta_config['ls'] == 0:
                # Get local models parameters
                list_params = [[param for param in list_models[worker_id].parameters()] for worker_id in range(config.num_workers)]

                # Average parameters across all nodes and update SlowMo Buffer
                avg_params = [torch.mean(torch.stack(worker_param), dim=0) for worker_param in zip(*list_params)]
                slowmo_buffer = [beta*slowmo_param + (1/model.learning_rate)*(param - avg_param) for slowmo_param, avg_param, param in zip(slowmo_buffer, avg_params, model.parameters())]

                # Apply averaged gradients to the global model
                with torch.no_grad():
                    for param, slowmo_param in zip(model.parameters(), slowmo_buffer):
                        param.grad = slowmo_param.clone()
                        param.data -= alpha *model.learning_rate * param.grad
                
                # Clone the weight of the  Global Model to the Workers
                for local_model in list_models:
                    for param, local_param in zip(model.parameters(), local_model.parameters()):
                        local_param.data = param.data.clone()
    
                # Set the update Flag to Model updated
                updated = True
                
                # If skip is enable, it's not the last iteration and the next synchronization would be after the dataset exploration
                # End the Epoch Training
                if t+meta_config['ls'] > meta_config['budget'] and t+1 != meta_config['budget'] and last_step_skip:
                    pbar_t.close()
                    break
        # In case of skip not enabled and last synchronization was not on the last iteration
        # Synchronize the global Model 
        if not last_step_skip and not updated:
                # Get local models parameters
                list_params = [[param for param in list_models[worker_id].parameters()] for worker_id in range(config.num_workers)]

                # Average parameters across all nodes and update SlowMo Buffer
                avg_params = [torch.mean(torch.stack(worker_param), dim=0) for worker_param in zip(*list_params)]
                slowmo_buffer = [beta*slowmo_param + (1/model.learning_rate)*(param - avg_param) for slowmo_param, avg_param, param in zip(slowmo_buffer, avg_params, model.parameters())]

                # Apply averaged gradients to the global model
                with torch.no_grad():
                    for param, slowmo_param in zip(model.parameters(), slowmo_buffer):
                        param.grad = slowmo_param.clone()
                        param.data -= alpha *model.learning_rate * param.grad


    # Meta Parameters for Dynamic Local Step Scheduler
    if config.work.dynamic is not None:
        if config.work.dynamic.strategy == "AvgParamDev":
            meta_config['avg_params'] = avg_params
            meta_config['list_params'] = list_params

    return losses, [100*correct/total for correct, total in zip(corrects, totals)]


def defineTraining(config: ModelConfig) -> callable:
    if config.num_workers > 0:
        return localSDG
    else:
        return centralized


# Training loop with validation
def train_model(config: Config, train_data: torchvision.datasets, val_data: torchvision.datasets, model: torch.nn.Module, best_model: torch.nn.Module, device: str, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, criterion, checkpoint: dict = None) -> tuple[torch.nn.Module, dict]:
    
    train_losses = [[] for _ in range(config.model.num_workers if config.model.num_workers > 0 else 1)]
    train_accuracies = [[] for _ in range(config.model.num_workers if config.model.num_workers > 0 else 1)]
    val_losses = []
    val_accuracies = []
    train_local_steps = []
    train_learning_rates = []

    if config.model.patience <= 0:
        patience = config.model.epochs
    else:
        patience = config.model.patience

    #load values form checkpoint
    val_acc = 0.0 if checkpoint is None else checkpoint['val_acc']
    best_acc = 0.0 if checkpoint is None else checkpoint['best_acc']
    start_epoch = 0 if checkpoint is None else checkpoint['start_epoch']
    
    # Create the Training Dataloaders for Centralized or Distributed
    if config.model.num_workers > 0:
        train_loader = [DataLoader(data, batch_size=config.model.work.batch_size, shuffle=True, drop_last=False) for data in train_data]
    else:
        train_loader = [DataLoader(train_data, batch_size=config.model.batch_size, shuffle=False, drop_last=False)]
    val_loader = DataLoader(val_data, batch_size=config.model.batch_size, shuffle=False)

    train_func = defineTraining(config.model)

    # Define Meta configuration Parameters for Local Step Dynamic Scheduler
    if config.model.num_workers > 0:
        meta_config= {
            'budget': max([len(loader) for loader in train_loader]),
            'ls': config.model.work.local_steps,
            'no_impr_count':0,
            'loss_history': None,
            'tot_ss': 0 if config.model.work.dynamic else config.model.work.sync_steps*config.model.epochs
        }
        if config.model.work.dynamic:
            update_steps = sel_dynamic_ls(config.model)
    else:
        meta_config = {}

    postfix = {'Val_Acc': f'{val_acc:.2f}%', 'Best_Acc': f'{best_acc:.2f}%'}
    with tqdm(range(start_epoch, config.model.epochs), desc='Epoch', position=1, postfix=postfix) as pbar:
        for epoch in pbar:
            # Learning Rate Update
            model.learning_rate = scheduler.get_last_lr()[0] if scheduler.get_last_lr()[0] != 0 else model.learning_rate
            model.train()

            #Training phase
            epoch_loss, epoch_acc = train_func(config.model, meta_config, model, train_loader, optimizer, criterion, device, pbar)

            for train_loss, ep_loss in zip(train_losses, epoch_loss):
                train_loss.append(ep_loss)
            for train_acc, ep_acc in zip(train_accuracies, epoch_acc):
                train_acc.append(ep_acc)

            if config.model.num_workers > 0:
            # Add current number of local steps performed and current learning rate
                train_local_steps.append(meta_config['ls'])
                train_learning_rates.append(model.learning_rate)

            scheduler.step()

            # Validate the model on the validation set
            val_loss, val_acc = validate_model(val_loader, criterion, model)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if config.model.num_workers > 0:
                if config.model.work.dynamic:
                    if len(val_losses) > config.model.work.dynamic.n_losses:
                        meta_config['loss_history'] = val_losses[-config.model.work.dynamic.n_losses:]
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

            if meta_config['no_impr_count'] >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            if config.model.num_workers > 0:
                if config.model.work.dynamic:
                    meta_config['epoch'] = epoch
                    update_steps(config.model, meta_config)
                    pbar.write(f"Epoch {meta_config['epoch']}: {meta_config['ls']}")
            
            save_checkpoint(config, epoch, model, best_model, optimizer, scheduler, val_acc, best_acc)
    
    meta_config['val_losses'] = val_losses
    meta_config['val_accuracies'] = val_accuracies
    meta_config['train_losses'] = train_losses
    meta_config['train_accuracies'] = train_accuracies
    meta_config['train_local_steps'] = train_local_steps
    meta_config['train_learning_rates'] = train_learning_rates

    plot_metrics("distributed" if config.model.num_workers > 0 else "centralized", config, train_losses, train_accuracies, val_losses, val_accuracies)
    return best_model, meta_config
    

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
    # Set default Configuration File
    yaml_path = script_dir / 'config' / 'SlowMo_Lenet.yaml'    
    parser = Parser(yaml_path)
    # Read Configuration File and Parse possible arguments
    config, device = parser.parse_args()

    train_data, val_data, test_data = load_cifar100(config)

    model = sel_model(config.model)
    device = sel_device(device)
    model = model.to(device)
    # Skip other Selectors in case of Only Test
    if config.experiment.test_only == False:
        loss_function = sel_loss(config.model)
        optimizer = sel_optimizer(config.model, model)
        scheduler = sel_scheduler(config.model, optimizer, len(train_data))
        best_model = deepcopy_model(model)

        checkpoint = None
        # Load previous checkpoint in case of resume 
        if config.experiment.resume:     
            checkpoint = load_checkpoint(config, model, best_model, optimizer, scheduler)

    # Load pretrained weights in case of pretrained model
    if config.model.pretrained:
        load_pretrain(config, model)

    # Train the model and Test the Best Model if it's not Only Test
    if config.experiment.test_only == False:
        best_model, meta_config = train_model(config, train_data, val_data, model, best_model, device, optimizer, scheduler, loss_function, checkpoint)
        best_model_acc = evaluate_model(config.model,test_data, best_model)
        print(f'Best Model Test Accuracy: {best_model_acc:.2f}%')
    
    # Test the Model
    model_acc = evaluate_model(config.model,test_data, model)
    print(f'Model Test Accuracy: {model_acc:.2f}%')
    
    # Save Result and Settings
    if config.experiment.test_only == False:
        save_to_csv(config, meta_config, model_acc, best_model_acc)

        if config.model.num_workers > 0:
            save_to_pickle(config, meta_config)

