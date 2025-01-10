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
from tqdm import tqdm


# [ ] Plot
# [x] Worker/Sequential Training
# [ ] Scheduler
# [ ] Optimizer

validation_split = 0.1  # 10% of the training data will be used for validation

def split_dataset(dataset, k):
    # Determine the sizes of each subset
    subset_size = len(dataset) // k
    remaining = len(dataset) % k
    lengths = [subset_size + 1 if i < remaining else subset_size for i in range(k)]
    
    # Split the dataset
    subsets = random_split(dataset, lengths)
    return subsets

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

    return train_subset, val_subset, test_dataset

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
    if config.model.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.model.learning_rate)
    return optimizer

def sel_loss(config):
    assert config.model.loss, "Loss not selected"
    if config.model.loss == 'CSE':
        criterion = nn.CrossEntropyLoss()
    return criterion

def deepcopy_model(model):
    # a dirty hack....
    tmp_model = deepcopy(model)
    for tmp_para, para in zip(tmp_model.parameters(), model.parameters()):
        if para.grad is not None:
            tmp_para.grad = para.grad.clone()
    return tmp_model

def localSDG(config, model, training_data_splits, optimizer, criterion, device, pbar):
    with tqdm(range(config.model.work.sync_steps), desc='Sync', position=1) as pbar_t:
        for t in pbar_t:
            list_models = [deepcopy_model(model) for i in range(config.model.num_workers)]
            #tmp_optimizer, tmp_scheduler = [optim.SGD(list_models[k].parameters(), lr=list_models[k].learning_rate) for k in range(K)]
            # scheduler = [optim.lr_scheduler.CosineAnnealingLR(
            #         tmp_optimizer, T_max=H, eta_min=0
            #     
            list_gradients = []
            for k in range(config.model.num_workers):
                for inputs, labels in training_data_splits[k][t]:
                    # take images and labels from dataloader
                    # transfer them to GPU
                    inputs, labels = inputs.to(device), labels.to(device)
                    # reset gradients
                    #tmp_optimizer[k].zero_grad()
                    #list_models[k].zero_grad()

                    # local forward pass
                    outputs = list_models[k](inputs)
                    loss = criterion(outputs, labels)
                    
                    # local backward pass and optimization
                    loss.backward()
                
                    # for param, global_param in zip(list_models[k].parameters(), model.parameters()):
                    #     param = param - list_models[k].learning_rate * param.grad

                    for param, global_param in zip(list_models[k].parameters(), model.parameters()):
                        param.data = global_param.data - list_models[k].learning_rate * param.grad

                    #tmp_optimizer[k].step()
                
            
                pbar.write(f"Work{k+1:02} Loss: {loss}")

            #list_gradients.append([param.grad for param in list_models[k].parameters()])
            list_gradients.append([global_param - param for global_param, param in zip(model.parameters(), list_models[k].parameters())])

            # Now, average gradients across all nodes
            averaged_gradients = []
            for param_gradients in zip(*list_gradients):
                avg_grad = torch.mean(torch.stack(param_gradients), dim=0)
                averaged_gradients.append(avg_grad)

            # Apply averaged gradients to the global model
            with torch.no_grad():
                for param, avg_grad in zip(model.parameters(), averaged_gradients):
                    param.grad = avg_grad
                    param.data -= model.learning_rate * param.grad

# Training loop with validation
def train_model(config, train_data, val_data, model, device, optimizer, criterion, checkpoint = None):
    
    best_acc = 0.0 if checkpoint is None else checkpoint.best_acc
    start_epoch = 0 if checkpoint is None else checkpoint.start_epoch

    work_data_splits = split_dataset(train_data, config.model.num_workers)
    sync_data_split = [split_dataset(data,config.model.work.sync_steps) for data in work_data_splits]
    sycn_data_loaders = [[DataLoader(data, batch_size=config.model.work.batch_size, shuffle=False, drop_last=False) for data in sync_step] for sync_step in sync_data_split]
    

    val_loader = DataLoader(val_data, batch_size=config.model.batch_size, shuffle=False)
    postfix = {'Val_Acc': '00.00%', 'Best_Acc': '00.00%'}
    with tqdm(range(start_epoch, config.model.epochs), desc='Epoch', position=0, postfix=postfix) as pbar:
        for epoch in pbar:
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

            

            localSDG(config, model, sycn_data_loaders, optimizer, criterion, device, pbar)

            # epoch_loss = running_loss / len(train_loader)
            # epoch_acc = 100 * correct / total
            # print(f'Epoch [{epoch+1}/{config.model.epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')

            # Validate the model on the validation set
            val_acc = validate_model(val_loader, model)
            #print(f'Validation Accuracy: {val_acc:.2f}%')
            
            
            # Save the model with the best accuracy on the validation set
            if val_acc > best_acc:
                best_acc = val_acc
                # save_checkpoint(epoch, model, optimizer, best_acc, epoch_loss, config)
            #print(f"Epoch time: {time.time() - start_time:.2f} seconds")
            postfix['Val_Acc'] = f'{val_acc:.2f}%'
            postfix['Best_Acc'] = f'{best_acc:.2f}%'
            pbar.set_postfix(postfix)

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
def evaluate_model(test_data, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loader = DataLoader(test_data, batch_size=config.model.batch_size, shuffle=False)
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
    parser = Parser('src/config/Distributed_Lenet.yaml')
    config, device = parser.parse_args()
    
    train_data, val_data, test_data = load_cifar100(config)

    model = sel_model(config)
    device = sel_device(config)
    model = model.to(device)
    loss_function = sel_loss(config)
    optimizer = sel_optimizer(config, model)
    
    if config.experiment.resume:
        model, optimizer, checkpoint = load_checkpoint(config)
    
      
    train_model(config, train_data, val_data, model, device, optimizer, loss_function, checkpoint = checkpoint if config.experiment.resume else None)
    evaluate_model(test_data, model)

    