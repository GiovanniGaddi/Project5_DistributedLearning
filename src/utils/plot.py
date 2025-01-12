from matplotlib import pyplot as plt
import os

def plot_metrics(type, config, train_losses, train_accuracies, val_losses, val_accuracies):

    #os.makedirs('../plots', exist_ok=True)
    filename = f"{type}_opt-{config.model.optimizer}_sched-{config.model.scheduler}_lr-{config.model.learning_rate}_bs-{config.model.batch_size}.png"
    filepath = os.path.join('../plots', filename)
    # Training and validation loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    #plt.savefig(filepath)
    plt.show()
    #plt.close()