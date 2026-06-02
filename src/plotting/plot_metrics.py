import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax.set_title('Training and Validation Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig