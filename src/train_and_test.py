import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import os
from models.lenet5 import leNet5
from utils.parser import Parser
from copy import deepcopy


# [ ] Plot
# [x] Worker/Sequential Training
# [ ] Scheduler
# [ ] Optimizer

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
    # Split train subset into num_nodes DataLoaders
    train_loaders = []
    #val_loaders = []
    num_nodes = config.model.num_nodes
    for i in range(num_nodes):
        train_indices = train_subset[(train_size / num_nodes) * i: (train_size / num_nodes) * (i+1)]
        train_loaders.append(DataLoader(train_indices, batch_size=config.model.batch_size, shuffle=True))
        
    #train_loader = DataLoader(train_subset, batch_size=config.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.model.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.model.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader 

#Save checkpoint
def save_checkpoint(epoch, model, optimizer, best_acc, loss, config):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(config.checkpoint.dir, f'{config.experiment.version}.pth'))

def load_checkpoint(config, model, optimizer):
        checkpoint = torch.load(os.path.join(config.checkpoint.dir, f'{config.experiment.version}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainin_state = {}
        trainin_state.start_epoch = checkpoint['epoch'] + 1
        trainin_state.best_acc = checkpoint['best_acc']
        trainin_state.loss = checkpoint['loss']

        return model, optimizer, trainin_state

def sel_model(config):
    assert config.model.name, "Model not selected"
    # Define a LeNet-5 model
    if config.model.name == 'LeNet':
        model = leNet5(config.model.learning_rate)
    return model

def sel_device(device):
    assert device, "Invalid selected Device"
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
    
def sel_optimizer(config, model):
    assert config.model.optimizer, "Model not selected"
    if config.model.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.model.learning_rate)
    return optimizer

def sel_loss(config):
    assert config.model.loss, "Loss not selected"
    if config.model.loss == 'CSE':
        criterion = nn.CrossEntropyLoss()
    return criterion

def localSDG(model, training_data, optimizer, criterion, minibatch_size, step_size, momentum, num_nodes, device, synch_steps=50, local_steps=90):
    for t in range(synch_steps):
        list_models = [deepcopy(model) for i in range(num_nodes)]
        list_gradients = []
        for k in range(num_nodes):
            for h in range(local_steps):
                # take images and labels from dataloader
                inputs, labels = training_data[k].dataset[t*h + h]
                # transfer them to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                # reset gradients
                optimizer.zero_grad()

                # local forward pass
                outputs = list_models[k](inputs)
                loss = criterion(outputs, labels)
                
                # local backward pass and optimization
                loss.backward()
                optimizer.step()

            # keep local model's gradients
            list_gradients.append(list_models[k].parameters())

        # local models' gradients aggregation
        reduced_gradients = [model.parameters() for _ in range(num_nodes)] - list_gradients

        # wordlwide model's update
        model = model.parameters() - model.learning_rate / num_nodes * sum(reduced_gradients)

# Training loop with validation
def train_model(config, train_loader, val_loader, model, device, optimizer, criterion, checkpoint = None):
    
    best_acc = 0.0 if checkpoint is None else checkpoint.best_acc
    start_epoch = 0 if checkpoint is None else checkpoint.start_epoch

    for epoch in range(start_epoch, config.model.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # Training phase
        # for inputs, labels in train_loader:
        #     inputs, labels = inputs.to(device), labels.to(device)

        #     optimizer.zero_grad()

        #     # Forward pass
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)

        #     # Backward pass and optimization
        #     loss.backward()
        #     optimizer.step()

        #     # Track the loss and accuracy
        #     running_loss += loss.item()
        #     _, predicted = torch.max(outputs, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()

        localSDG(model, train_loader, optimizer, criterion, 0, 10, 0.9, config.model.num_nodes, device)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{config.model.epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')

        # Validate the model on the validation set
        val_acc = validate_model(val_loader, model)
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Save the model with the best accuracy on the validation set
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(epoch, model, optimizer, best_acc, epoch_loss, config)

        print(f"Epoch time: {time.time() - start_time:.2f} seconds")

# Validation function
def validate_model(val_loader, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

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
    parser = Parser('src/config/Lenet.yaml')
    config, device = parser.parse_args()
    
    train_loader, val_loader, test_loader = load_cifar100(config)

    model = sel_model(config)
    device = sel_device(config)
    model = model.to(device)
    loss_function = sel_loss(config)
    optimizer = sel_optimizer(config, model)
    
    if config.experiment.resume:
        model, optimizer, checkpoint = load_checkpoint(config)
    
      
    train_model(config, train_loader, val_loader, model, device, optimizer, loss_function, checkpoint = checkpoint if config.experiment.resume else None)
    evaluate_model(test_loader, model)