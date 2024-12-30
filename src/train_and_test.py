import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import os
from pathlib import Path
from models.lenet5 import leNet5
from utils.parser import Parser
from utils.plot import plot_metrics


# [X] Plot
# [ ] Worker/Sequential Training
# [X] Scheduler
# [X] Optimizer

validation_split = 0.1  # 10% of the training data will be used for validation

def load_cifar100(config):
    # Transformations to apply to the dataset (including normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # CIFAR-100 mean/std
    ])

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

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
        optimizer = optim.AdamW(model.parameters(), lr=config.model.learning_rate)
    elif config.model.optimizer == "SGDM":
        optimizer = optim.SGD(model.parameters(), lr=config.model.learning_rate, momentum=0.9)
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
    if config.model.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.model.epochs, eta_min=1e-5
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler.name}")
    return scheduler

# Training loop with validation
def train_model(config, train_loader, val_loader, model, device, optimizer, scheduler, criterion, checkpoint = None):
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_acc = 0.0 if checkpoint is None else checkpoint.best_acc
    start_epoch = 0 if checkpoint is None else checkpoint.start_epoch

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
        print(f'Validation Accuracy: {val_acc:.2f}%')

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save the model with the best accuracy on the validation set
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(epoch, model, optimizer, scheduler, best_acc, epoch_loss, config)

        print(f"Epoch time: {time.time() - start_time:.2f} seconds")

    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)

# Validation function
def validate_model(val_loader, criterion, model):
    model.eval()  # Set model to evaluation mode
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
    model.eval()  # Set model to evaluation mode
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
    
      
    train_model(config, train_loader, val_loader, model, device, optimizer, scheduler, loss_function, checkpoint = checkpoint if config.experiment.resume else None)
    evaluate_model(test_loader, model)