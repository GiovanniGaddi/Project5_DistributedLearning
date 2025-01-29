# Understanding data-parallel approaches in Distributed Learning: objectives, challenges, and performance trade-offs

## Abstract

This paper explores distributed learning techniques for deep neural network training using data parallelism. We evaluate large-batch training optimizers (SGDM, AdamW, LARS, LAMB) and local methods such as LocalSGD, analyzing their trade-offs in terms of scalability, communication overhead, and convergence stability. Experiments conducted on the CIFAR-100 dataset with LeNet5 architecture provide insights into optimizer selection, hyperparameter tuning, and dataset sharding strategies. Additionally, we propose dynamic local step adjustment strategies for LocalSGD to balance accuracy and communication cost.

## Table of Contents

- [Introduction](#introduction)  
- [Related works](#related-works)  
- [Experiments](#experiments)  
- [Results](#results)  
- [Conclusion](#conclusion) 

## Introduction

Deep learning models require significant computational resources. Distributed learning, particularly data parallelism, accelerates training but introduces challenges such as communication overhead and optimization stability. We analyze large-batch training and local update methods within synchronous distributed learning to find optimal training strategies.

## Related Works

We review recent advancements in distributed learning, focusing on communication efficiency, large-batch optimization, and scalability, providing key insights into learning rate scaling, LocalSGD, and momentum-based optimization techniques.

## Experiments

### Dataset

We use the CIFAR-100 dataset, consisting of 60,000 images across 100 classes, for evaluating our distributed learning techniques.

### Implementation Details

We employ a modified LeNet-5 architecture with convolutional and fully connected layers. A centralized baseline is established using AdamW and SGDM optimizers.

### Large-Batch Optimization

We analyze the scalability of LARS and LAMB optimizers, evaluating their impact on training performance with increasing batch sizes.

### Local Methods

We implement LocalSGD with varying numbers of workers (K) and local steps (H), balancing synchronization frequency and computational efficiency.

### Hybrid Optimization

We integrate SlowMo, a hybrid optimizer, to mitigate performance degradation in LocalSGD by applying momentum correction.

### Dynamic Local Step Adjustment

We propose and test multiple dynamic strategies to adjust the number of local steps during training, leveraging for example loss-based adjustments, learning rate scheduling, and parameter deviation tracking.

## Results

SGDM outperforms AdamW in centralized training.

Large-batch optimizers (LARS, LAMB) can maintain performance better than traditional optimizers as batch size increases, when an appropriate learning rate is chosen.

LocalSGD reduces communication overhead but can suffer performance degradation, which is mitigated by SlowMo.

Dynamic adjustment of local steps improves convergence stability and reduces communication costs.

## Conclusion

This study provides practical insights into optimizing distributed learning. By evaluating large-batch training and local update methods, we propose strategies to improve efficiency and scalability in deep learning training pipelines.


