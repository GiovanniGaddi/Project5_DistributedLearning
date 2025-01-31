from tqdm import trange
from torch.utils.data import random_split, Subset
from torchvision import datasets, transforms

from utils.conf import Config

validation_split = 0.1  # 10% of the training data will be used for validation

def iid_sharding(dataset, num_shards: int)-> list[Subset]:
    #reorder data based on label
    tmp_data = [[] for _ in range(100)]
    for j in trange(dataset.__len__(), desc= "Sharding"):
        tmp_data[dataset.__getitem__(j)[1]].append(j)
    
    data_idx = []
    for idxs in tmp_data:
        data_idx.extend(idxs)

    # Divide into shards
    data_shards = [data_idx[i::num_shards] for i in range(num_shards)]

    # Create Subsets for each shard
    subsets = [Subset(dataset, shard) for shard in data_shards]
    return subsets

def split_dataset(dataset, k: int)-> list[Subset]:
    # Determine the sizes of each subset
    subset_size = len(dataset) // k
    remaining = len(dataset) % k
    lengths = [subset_size + 1 if i < remaining else subset_size for i in range(k)]
    
    # Split the dataset
    subsets = random_split(dataset, lengths)
    return subsets

def load_cifar100(config: Config)-> tuple[Subset, Subset, datasets]:
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
    # Set traininng values to non in case of only test
    if config.experiment.test_only:
        train_subset = None
        val_subset = None
    else:
        # Load CIFAR-100 training dataset + transformation
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        
        # Split the training dataset into training and validation sets
        train_size = int((1 - validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        if config.model.num_workers > 0:
            train_subset = iid_sharding(train_subset, config.model.num_workers)
    
    # Load CIFAR-100 test dataset + transformation
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    return train_subset, val_subset, test_dataset

