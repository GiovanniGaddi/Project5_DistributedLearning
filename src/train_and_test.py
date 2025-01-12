import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
from copy import deepcopy
import os
from pathlib import Path
from models.lenet5 import leNet5
from utils.parser import Parser
from utils.plot import plot_metrics
from utils.optim import LAMB, LARS


'''
Best results for centralized at 150 epochs:
- SGDM, lr 0.01, momentum 0.9, weight decay 4e-4: test accuracy 52.28%
- SGDM, lr 0.01, momentum 0.9, weight decay 4e-3: test accuracy 55.10%/55.50%
- AdamW, lr 0.001, weight decay 0.01: test accuracy 49.56%
- AdamW, lr 0.001, weight decay 0.04: test accuracy 52.09%
- AdamW, lr 0.0015, weight decay 0.04: test accuracy 48.4%
- AdamW, lr 0.003, weight decay 0.04: test accuracy 48.7%
- AdamW, lr 0.002, weight decay 0.04: test accuracy 48.6%


Best results with large batch and warmup scheduler:
- LAMB, lr 0.128, batch size 8192, 150 epochs (warmup 15), polynomial: test accuracy next 
- LAMB, lr 0.032, batch size 2048, 30 epochs: test accuracy 39.89%
- LAMB, lr 0.032, batch size 1024, 30 epochs: test accuracy 41.58%
- LAMB, lr 0.032, batch size 1024, 50 epochs: test accuracy 43.17%
- LAMB, lr 0.032, batch size 8192, 100 epochs (warmup 5): test accuracy 45.02%
- LAMB, lr 0.032, batch size 16384, 150 epochs (warmup 15): test accuracy 44.72%
- LAMB, lr 0.032, batch size 2048, 150 epochs (warmup 15) with early stopping: test accuracy 46.04%
- LAMB, lr 0.032, batch size 2048, 150 epochs (warmup 15), polynomial with early stopping: test accuracy 45.10% 
- LAMB, lr 0.008, batch size 2048, 100 epochs (warmup 5): test accuracy 46.43%
- LAMB, lr 0.008, batch size 2048, 150 epochs (warmup 15): test accuracy 47.09%
- LAMB, lr 0.008, batch size 2048, 150 epochs (warmup 15), polynomial: test accuracy 46.93% 
- LAMB, lr 0.005, batch size 1024, 150 epochs (warmup 15), polynomial: test accuracy 44.07% 
- LAMB, lr 0.005, batch size 512, 150 epochs (warmup 15), polynomial: test accuracy todo




Default momentum 0.9
- LARS, lr 10, batch size 256, 150 epochs (warmup 15): test accuracy 47.71%
- LARS, lr 12.8, batch size 8192, 150 epochs (warmup 15): test accuracy 40%
- LARS, lr 15, batch size 2048, 150 epochs (warmup 15): test accuracy 43.14%
- LARS, lr 15, batch size 1024, 150 epochs (warmup 15): test accuracy 46.47%
- LARS, lr 15, batch size 512, 150 epochs (warmup 15): test accuracy 46.93%
- LARS, lr 15, batch size 256, 150 epochs (warmup 15): test accuracy 49.04%/47.87%
- LARS, lr 15, batch size 256, 150 epochs (warmup 15), momentum 0.5: test accuracy 41.38%
- LARS, lr 15, batch size 128, 150 epochs (warmup 15): test accuracy 48.65%
- LARS, lr 20, batch size 256, 150 epochs (warmup 15): test accuracy 47%
- LARS, lr 20, batch size 1024, 150 epochs (warmup 15): test accuracy 46.05%
- LARS, lr 25, batch size 1024, 150 epoch (warmup 15): test accuracy 46.25%
- LARS, lr 30, batch size 512, 150 epoch (warmup 15): test accuracy 46.72%
- LARS, lr 30, batch size 8192, 150 epoch (warmup 15): test accuracy next


'''
# [X] Plot
# [X] Scheduler
# [X] Optimizer
# [X] LAMB&LARS

# Next steps to end part 3:
# [ ] hyperparameter tuning and comparison
# [ ] IID sharding and training with LocalSGD and PostLocalSGD
# [ ] Performing multiple local steps, scaling the number of iteration
# [ ] Using two optimizers: one for the outer loop and one for the inner loop (+ analysis)

validation_split = 0.1  # 10% of the training data will be used for validation
best_model = None

def load_cifar100(config):
    # Transform the training set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),    
        transforms.ToTensor(),                
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  
    ])

    # Transform the test set
    test_transform = transforms.Compose([
        transforms.CenterCrop(32),            
        transforms.ToTensor(),                
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  
    ])

    # Load CIFAR-100 dataset + transformation
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    # Split the training dataset into training and validation sets
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders for training, validation, and test sets
    train_loader = DataLoader(train_subset, batch_size=config.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.model.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.model.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader 

#Save checkpoint
def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, loss, config):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(config.checkpoint.dir, f'{config.experiment.version}.pth'))

def load_checkpoint(config, model, optimizer, scheduler):
        checkpoint = torch.load(os.path.join(config.checkpoint.dir, f'{config.experiment.version}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        trainin_state = {}
        trainin_state.start_epoch = checkpoint['epoch'] + 1
        trainin_state.best_acc = checkpoint['best_acc']
        trainin_state.loss = checkpoint['loss']

        return model, optimizer, scheduler, trainin_state

def sel_model(config):
    assert config.model.name, "Model not selected"
    # Define a LeNet-5 model
    if config.model.name == 'LeNet':
        model = leNet5()
    return model

def sel_device(device):
    assert device, "Invalid selected Device"
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
    
def sel_optimizer(config, model):
    assert config.model.optimizer, "Optimizer not selected"
    if config.model.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.model.learning_rate, weight_decay=0.04)
    elif config.model.optimizer == "SGDM":
        optimizer = optim.SGD(model.parameters(), lr=config.model.learning_rate, momentum=0.9, weight_decay=0.004)
    elif config.model.optimizer == "LARS":
        optimizer = LARS(model.parameters(), lr=config.model.learning_rate, momentum=0.9, weight_decay=0.004)
    elif config.model.optimizer == "LAMB":
        optimizer = LAMB(model.parameters(), lr=config.model.learning_rate, weight_decay=0.04)
    else:
        raise ValueError(f"Unsupported optimizer: {config.model.optimizer}")
    return optimizer

def sel_loss(config):
    assert config.model.loss, "Loss not selected"
    if config.model.loss == 'CSE':
        criterion = nn.CrossEntropyLoss()
    return criterion

def sel_scheduler(config, optimizer):
    assert config.model.scheduler, "Scheduler not selected"

    warmup_epochs = config.model.warmup
    total_epochs = config.model.epochs
    warmup_iters = warmup_epochs * len(train_loader)

    if config.model.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.model.epochs, eta_min=0
        )
    elif config.model.scheduler == 'WarmUpCosineAnnealingLR':
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
    elif config.model.scheduler == 'WarmUpPolynomialDecayLR':
        # Warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_iters
        )
        # Polynomial Decay Scheduler
        polynomial_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                (1 - (step - warmup_iters) / (total_epochs * len(train_loader) - warmup_iters)) ** 2
            ) if step >= warmup_iters else 1.0
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, polynomial_scheduler],
            milestones=[warmup_iters]
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler.name}")
    return scheduler


# Training loop with validation
def train_model_centralized(config, train_loader, val_loader, model, device, optimizer, scheduler, criterion, checkpoint = None):
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    no_improvement_count = 0 

    if config.model.patience <= 0:
        patience = config.model.epochs
    else:
        patience = config.model.patience

    best_acc = 0.0 if checkpoint is None else checkpoint.best_acc
    start_epoch = 0 if checkpoint is None else checkpoint.start_epoch
    train_time = time.time()

    for epoch in range(start_epoch, config.model.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # Training phase
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
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{config.model.epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        scheduler.step()

        # Validate the model on the validation set
        val_loss, val_acc = validate_model(val_loader, criterion, model)
        print(f'Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_acc:.2f}%')

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save the model with the best accuracy on the validation set
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(epoch, model, optimizer, scheduler, best_acc, epoch_loss, config)
            print(f"Saved new best model at epoch {epoch+1}")
            best_model = deepcopy(model)
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print(f"Epoch time: {time.time() - start_time:.2f} seconds")

        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Total train time: {time.time() - train_time:.2f} seconds")

    plot_metrics("centralized", config, train_losses, train_accuracies, val_losses, val_accuracies)
    save_checkpoint(epoch, model, optimizer, scheduler, val_acc, epoch_loss, config)

    print("Testing best model...")
    evaluate_model(test_loader, best_model)

# Validation function
def validate_model(val_loader, criterion, model):
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
def evaluate_model(test_loader, model):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    script_dir = Path(__file__).parent

    yaml_path = script_dir / 'config' / 'Lenet.yaml'    
    parser = Parser(yaml_path)
    config, device = parser.parse_args()
    
    train_loader, val_loader, test_loader = load_cifar100(config)

    model = sel_model(config)
    device = sel_device(config)
    model = model.to(device)
    loss_function = sel_loss(config)
    optimizer = sel_optimizer(config, model)
    scheduler = sel_scheduler(config, optimizer)
    
    
    if config.experiment.resume:
        model, optimizer, scheduler, checkpoint = load_checkpoint(config)
    
      
    train_model_centralized(config, train_loader, val_loader, model, device, optimizer, scheduler, loss_function, checkpoint = checkpoint if config.experiment.resume else None)

    print("Testing last model...")
    evaluate_model(test_loader, model)
