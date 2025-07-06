import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.fc_models import FCNetworkMNIST, FCNetworkCIFAR10
from models.cnn_models import SimpleCNNMNIST, ResNetMNIST, ResNetCIFAR10, ResNetRegularizedCIFAR10
from utils.training_utils import train_model, evaluate_model, get_gradients
from utils.visualization_utils import plot_learning_curves, plot_confusion_matrix, plot_gradient_histogram
import os
import time
from Settings import *

# Создаем директории для результатов
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Установка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Функция для загрузки данных MNIST
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader

# Функция для загрузки данных CIFAR-10
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader

# Функция для расчета параметров модели
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 1.1 Сравнение на MNIST
def compare_mnist():
    train_loader, test_loader = load_mnist()
    models = {
        'FC': FCNetworkMNIST().to(device),
        'SimpleCNN': SimpleCNNMNIST().to(device),
        'ResNetMNIST': ResNetMNIST().to(device)
    }
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} on MNIST")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        start_time = time.time()
        
        # Обучение
        history = train_model(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS)
        
        # Инференс
        inference_start = time.time()
        test_loss, test_acc, inference_time, cm = evaluate_model(model, test_loader, criterion, device)
        total_inference_time = time.time() - inference_start
        
        # Сохранение результатов
        results[name] = {
            'parameters': count_parameters(model),
            'train_acc': history['train_acc'][-1],
            'test_acc': test_acc,
            'training_time': history['training_time'],
            'inference_time': total_inference_time,
            'history': history
        }
        
        # Визуализация
        plot_learning_curves(
            history['train_losses'], 
            history['test_losses'], 
            history['train_acc'], 
            [acc for acc in history['test_acc']], 
            f"MNIST_{name}"
        )
        
        print(f"{name} | Params: {results[name]['parameters']} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Train Time: {results[name]['training_time']:.2f}s | "
              f"Inference Time: {results[name]['inference_time']:.2f}s")
    
    return results

# 1.2 Сравнение на CIFAR-10
def compare_cifar10():
    train_loader, test_loader = load_cifar10()
    models = {
        'FC': FCNetworkCIFAR10().to(device),
        'ResNetCIFAR10': ResNetCIFAR10().to(device),
        'ResNetRegularized': ResNetRegularizedCIFAR10().to(device)
    }
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for name, model in models.items():
        print(f"\nTraining {name} on CIFAR-10")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        start_time = time.time()
        
        # Обучение
        history = train_model(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS)
        
        # Инференс
        inference_start = time.time()
        test_loss, test_acc, inference_time, cm = evaluate_model(model, test_loader, criterion, device)
        total_inference_time = time.time() - inference_start
        
        # Анализ градиентов
        optimizer.zero_grad()
        sample_data, _ = next(iter(train_loader))
        sample_data = sample_data.to(device)
        output = model(sample_data)
        loss = criterion(output, torch.randint(0, 10, (sample_data.size(0), )).to(device))
        loss.backward()
        gradients = get_gradients(model)
        
        # Сохранение результатов
        results[name] = {
            'train_acc': history['train_acc'][-1],
            'test_acc': test_acc,
            'training_time': history['training_time'],
            'inference_time': total_inference_time,
            'cm': cm,
            'gradients': gradients
        }
        
        # Визуализация
        plot_learning_curves(
            history['train_losses'], 
            history['test_losses'], 
            history['train_acc'], 
            [acc for acc in history['test_acc']], 
            f"CIFAR10_{name}"
        )
        plot_confusion_matrix(cm, class_names, f"CIFAR10_{name}")
        plot_gradient_histogram(gradients, f"CIFAR10_{name}")
        
        print(f"{name} | Test Acc: {test_acc:.2f}% | "
              f"Train Time: {results[name]['training_time']:.2f}s | "
              f"Inference Time: {results[name]['inference_time']:.2f}s")
    
    return results

# Основной блок
if __name__ == "__main__":
    print("="*50)
    print("Starting MNIST Comparison")
    print("="*50)
    mnist_results = compare_mnist()
    
    print("\n" + "="*50)
    print("Starting CIFAR-10 Comparison")
    print("="*50)
    cifar_results = compare_cifar10()
    
    # Сохранение результатов в текстовый файл
    with open('C:/Users/Dismas/Study/programming/practice/4/results/comparison_results.txt', 'w') as f:
        f.write("MNIST Results:\n")
        for model, res in mnist_results.items():
            f.write(f"{model}: Params={res['parameters']}, Test Acc={res['test_acc']:.2f}%, Train Time={res['training_time']:.2f}s\n")
        
        f.write("\nCIFAR-10 Results:\n")
        for model, res in cifar_results.items():
            f.write(f"{model}: Test Acc={res['test_acc']:.2f}%, Train Time={res['training_time']:.2f}s\n")