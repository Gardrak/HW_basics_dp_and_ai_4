import matplotlib.pyplot as plt
import seaborn as sns
from Settings import *

def plot_learning_curves(train_losses, test_losses, train_acc, test_acc, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title(f'{model_name} - Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_acc, label='Train Accuracy')
    ax2.plot(test_acc, label='Test Accuracy')
    ax2.set_title(f'{model_name} - Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/{model_name}_curves.png')
    plt.close()

def plot_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{PLOTS_PATH}/{model_name}_cm.png')
    plt.close()

def plot_gradient_histogram(gradients, model_name):
    plt.figure(figsize=(10, 6))
    plt.hist(gradients, bins=50, alpha=0.7, color='b', log=True)
    plt.title(f'{model_name} - Gradient Distribution')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency (log scale)')
    plt.savefig(f'{PLOTS_PATH}/{model_name}_gradients.png')
    plt.close()