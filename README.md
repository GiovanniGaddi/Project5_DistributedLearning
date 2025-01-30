# Understanding data-parallel approaches in Distributed Learning: objectives, challenges, and performance trade-offs

## Abstract

This paper explores distributed learning techniques for deep neural network training using data parallelism. We evaluate standard and large-batch training optimizers (SGDM, AdamW, LARS, LAMB) and local methods such as LocalSGD, analyzing their trade-offs in terms of scalability, communication overhead, and convergence stability. Experiments conducted on the CIFAR-100 dataset with LeNet5 architecture provide insights into optimizer selection, hyperparameter tuning, and dataset sharding strategies. Additionally, we propose dynamic local step adjustment strategies for LocalSGD to balance accuracy and communication cost.

## Table of Contents

- [Introduction](#introduction)  
- [Related works](#related-works)
- [Repository structure](#repository-structure) 
- [Experiments](#experiments)  
- [Results](#results)  
- [Conclusion](#conclusion) 

## Introduction

Deep learning models require significant computational resources. Distributed learning, particularly data parallelism, accelerates training but introduces challenges such as communication overhead and optimization stability. We analyze large-batch training and local update methods within synchronous distributed learning to find optimal training strategies.

## Related works

We review recent advancements in distributed learning, focusing on communication efficiency, large-batch optimization, and scalability, providing key insights into learning rate scaling, LocalSGD, and momentum-based optimization techniques.

## Repository structure

This section describes the structure of the project repository and explains the purpose of each file and folder.

### Directory Layout

The repository is organized as follows:

```
./
│
├── src/                                     Main source code folder
│   ├── train_and_test.py                    Main script for training and testing the model
│   ├── config/                              Configuration folder
│   │   └── Distributed_Lenet.yaml           Example of configuration file (YAML format)
│   ├── model/                               Folder containing model-related files
│   │   └── lenet5.py                        LeNet-5 model architecture definition
│   └── utils/                               Utility scripts and helper functions
│       ├── conf.py                          Configuration settings
│       ├── parser.py                        Argument parsing utility
│       ├── load_dataset.py                  Dataset loading and preprocessing
│       ├── optim.py                         Custom optimizers for large batch training
│       ├── plot.py                          Plotting and visualization functions
│       ├── selectors.py                     Utility to select optimizers, schedulers, etc
│       └── utils.py                         General utility functions, such as saving results, checkpoints, etc
│
├── checkpoints/                             Folder for saving model checkpoints
│   └── model_checkpoint.pth                 Example of saved model checkpoint
│
├── requirements.txt                         List of project dependencies (Python libraries)
├── README.md                                Project documentation (you are here)
└── RUN_GUIDE.md                             Run guide for executing the project

```


## Experiments

### Dataset

We use the CIFAR-100 dataset, consisting of 60,000 images across 100 classes, for evaluating our distributed learning techniques.

### Implementation details

We employ a modified LeNet-5 architecture with convolutional and fully connected layers. A centralized baseline is established using AdamW and SGDM optimizers.

### Large-batch optimization

We analyze the scalability of LARS and LAMB optimizers, evaluating their impact on training performance with increasing batch sizes.

### Local methods

We implement LocalSGD with varying numbers of workers (K) and local steps (H), balancing synchronization frequency and computational efficiency.

### Hybrid optimization

We integrate SlowMo, a hybrid optimizer, to mitigate performance degradation in LocalSGD by applying momentum correction.

### Dynamic local step adjustment

We propose and test multiple dynamic strategies to adjust the number of local steps during training, leveraging for example loss-based adjustments, learning rate scheduling, and other mathematical functions.

## Results

SGDM outperforms AdamW in centralized training.

Large-batch optimizers (LARS, LAMB) can maintain performance better than traditional optimizers as batch size increases, when an appropriate learning rate is chosen.

LocalSGD reduces communication overhead but can suffer performance degradation, which is mitigated by SlowMo.

Dynamic adjustment of local steps is performed with the purpose of improving convergence stability and reducing communication costs.

## Conclusion

This study provides practical insights into optimizing distributed learning. By evaluating large-batch training and local update methods, we propose strategies to improve efficiency and scalability in deep learning training pipelines.

# Run guide for the project

This guide provides detailed instructions on how to run the `train_and_test.py` script with various command-line arguments to customize the training and testing process.

## Step 1: Set up your environment

### Install the required dependencies
- pip install -r requirements.txt

## Step 2: Configuration file

Before running the script, you can have a look at configuration YAML file (`e.g., config/distributed_Lenet.yaml`). This file contains the settings for your model and experiment. 

## Step 3: Running the script

You can execute the script in the `src` folder using the following command:

- python train_and_test.py

### Available Arguments
- `-h, --help`: Show help for arguments and exit.
- `-c, --config`: Configuration file path. 
- `--cpu`: Use CPU as the device for training and testing.  
- `-lr, --learning-rate`: Override the default learning rate.  
- `-bs, --batch-size`: Override the default batch size.  
- `-ep, --epochs`: Override the default number of epochs for training (which is 150).  
- `-opt, --optimizer`: Override the default optimizer: you can choose among AdamW, SGDM, LARS, LAMB. Local methods will not use this argument, as LocalSGD is required.
- `-sch, --scheduler`: Override the default learning rate scheduler (we used CosineAnnealingLR).  
- `-p, --patience`: Override the early stopping patience: we generally set it to 0, but it can be useful sometimes in order to prevent overfitting.  
- `-wu, --warmup`: Override the number of warm-up steps for the learning rate scheduler: this is useful for large batch optimizers. 
- `-wd, --weight-decay`: Override the default weight decay.  
- `-sm, --slowmo-momentum`: Override the momentum (beta) for the SlowMo optimizer: if nothing is specified, we use 0.0
- `-slr, --slowmo-learning-rate`: Override the learning rate (alpha) for the SlowMo optimizer: if nothing is specified, we use 1.0
- `-nw, --number-workers`: Override the number of workers: use 0 for pure centralized, 1 for a distributed setting equivalent to centralized
- `-wss, --worker-sync-step`: Override the number of worker sync steps: useful only in distributed setting
- `-wls, --worker-local-step`: Override the number of local steps per worker: useful only in distributed setting  
- `-wbs, --worker-batch-size`: Override the batch size for workers: we use 64
- `-dls, --dynamic-local-step`: Override the dynamic local step function for workers: you can choose among those listed at the end of `selectors.py`, in `src/utils`  
- `-dnl, --dynamic-num-loss`: Override the dynamic number of losses taken into account in some strategies
- `-o, --output`: Path to the output file, format is csv
- `-P, --pretrained`: Path to the pretrained model checkpoint.  
- `-LC, --load-checkpoint`: Resume training from the last checkpoint.  
- `-TO, --test-only`: Skip training and run only the testing phase.  
- `-en, --experiment-name`: Change the name of the experiment.  
- `-v, --version`: Update the experiment version.

  
## Step 4: Example Full Command

Here is an example of a full command that customizes several arguments:

```bash
python train_and_test.py -lr 0.002 -bs 256 -ep 150 -opt LAMB -sch CosineAnnealingLR -nw 0
```

This command:
- Sets the learning rate to `0.002`
- Sets the batch size to `256`
- Trains for `150` epochs
- Uses the `LAMB` optimizer and `CosineAnnealingLR` scheduler
- Uses the centralized training function

## Step 5: Results

After running the script, you should see logs that indicate the training progress (if you are not using the `-T` flag). Otherwise, it will run the testing phase and output the results. 
The experiment results will be saved in the directory specified in the configuration file.
Finally, a plot will be saved in the corresponding directory.



