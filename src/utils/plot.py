from matplotlib import pyplot as plt
import os

def plot_metrics(type:str, config, train_losses: list, train_accuracies: list, val_losses: list, val_accuracies: list):

    filename = f"{type}_opt-{config.model.optimizer}_sch-{config.model.scheduler}_lr-{config.model.learning_rate}_bs-{config.model.batch_size}"
    if config.model.num_workers > 0:
        filename += f"_K-{config.model.num_workers}_H-{config.model.work.local_steps}"
    filename += ".png"
    dirname = 'plots'
    filepath = os.path.join(dirname, filename)
    os.makedirs(dirname, exist_ok=True)
    # Training and validation loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for idx, loss in enumerate(train_losses):
        plt.plot(loss, marker= 'o',  linewidth = 1, markersize = 1, label=f'{type}{idx+1 if len(train_losses) > 1 else ""}')
    plt.plot(val_losses, marker= 'o', linewidth = 2, markersize = 2, label='Validation')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Training and validation accuracy
    plt.subplot(1, 2, 2)
    for idx, accuracy in enumerate(train_accuracies):
        plt.plot(accuracy, marker= 'o',  linewidth = 1, markersize = 1, label=f'{type}{idx+1 if len(train_accuracies) > 1 else ""}')
    plt.plot(val_accuracies, marker= 'o',  linewidth = 2, markersize = 2, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filepath)
    #plt.show()
    #plt.close()