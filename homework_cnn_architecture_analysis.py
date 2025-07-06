import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from Settings import *


# Установка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Создаем директории для результатов
os.makedirs(PLOTS_PATH + '/kernel_analysis', exist_ok=True)
os.makedirs(PLOTS_PATH + '/depth_analysis', exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# 2.1 Модели для анализа размера ядра
class Kernel3x3CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Kernel5x5CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Kernel7x7CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MixedKernelCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 2.2 Модели для анализа глубины сети
class ShallowCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MediumCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 2 * 2, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
   
    
class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Уменьшаем количество слоев пулинга
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Автоматическая подстройка размера
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Первые 2 слоя с пулингом
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        
        # Следующие слои без пулинга
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        
        # Финал с адаптивным пулингом
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class ResNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Утилиты для обучения и оценки
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    train_losses = []
    train_acc = []
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    training_time = time.time() - start_time
    return train_losses, train_acc, training_time


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    activations = None  # Инициализируем как None
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            
            # Сохраняем активации только для первого батча
            if activations is None:
                activations = model.conv1(inputs).cpu().numpy()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    return test_loss, accuracy, activations


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Анализ рецептивных полей
def calculate_receptive_field(model):
    # Простая реализация расчета рецептивного поля
    rf = 1
    stride = 1
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            k = layer.kernel_size[0]
            s = layer.stride[0]
            rf = rf * s + (k - s)
            stride *= s
    return rf

# Визуализация активаций
def visualize_activations(activations, model_name, n=8):
    plt.figure(figsize=(16, 8))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        plt.imshow(activations[0, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    plt.suptitle(f'{model_name} - First Layer Activations')
    plt.savefig(f'{PLOTS_PATH}/kernel_analysis/{model_name}_activations.png')
    plt.close()

# Визуализация карт признаков
def visualize_feature_maps(model, input_tensor, model_name, layer_idx=0):
    # Регистрируем хук для захвата активаций
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Выбираем слой для визуализации
    target_layer = list(model.children())[layer_idx]
    hook = target_layer.register_forward_hook(get_activation('target'))
    
    # Пропускаем данные через модель
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    
    # Снимаем хук
    hook.remove()
    
    # Визуализация
    fmaps = activations['target'][0].cpu().numpy()
    n = min(8, fmaps.shape[0])
    
    plt.figure(figsize=(16, 8))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        plt.imshow(fmaps[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Feature Map {i+1}')
    plt.suptitle(f'{model_name} - Layer {layer_idx+1} Feature Maps')
    plt.savefig(f'{PLOTS_PATH}/depth_analysis/{model_name}_feature_maps.png')
    plt.close()

# Анализ градиентов
def analyze_gradients(model, train_loader, device):
    model.train()
    gradients = []
    
    # Получаем один батч данных
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Прямой и обратный проход
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    
    # Собираем градиенты
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append((name, param.grad.abs().mean().item()))
    
    return gradients

# 2.1 Анализ размера ядра
def kernel_size_analysis():
    models = {
        '3x3': Kernel3x3CNN().to(device),
        '5x5': Kernel5x5CNN().to(device),
        '7x7': Kernel7x7CNN().to(device),
        'Mixed': MixedKernelCNN().to(device)
    }
    
    # Регулируем количество параметров
    for name, model in models.items():
        if name == 'Mixed':
            # Уменьшаем количество фильтров для Mixed модели
            model.conv1 = nn.Conv2d(3, 16, kernel_size=1).to(device)
            model.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1).to(device)
            model.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1).to(device)
            model.fc = nn.Linear(32 * 8 * 8, 10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for name, model in models.items():
        print(f"\nAnalyzing kernel: {name}")
        # Подбираем количество параметров
        target_params = count_parameters(models['3x3'])
        current_params = count_parameters(model)
        
        # Печатаем информацию о рецептивном поле
        rf = calculate_receptive_field(model)
        print(f"Receptive field: {rf}")
        
        # Обучение
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_loss, train_acc, train_time = train_model(
            model, train_loader, criterion, optimizer, device, EPOCHS
        )
        
        # Оценка
        test_loss, test_acc, activations = evaluate_model(
            model, test_loader, criterion, device
        )
        
        # Визуализация активаций
        visualize_activations(activations, name)
        
        # Сохранение результатов
        results[name] = {
            'params': count_parameters(model),
            'train_acc': train_acc[-1],
            'test_acc': test_acc,
            'train_time': train_time,
            'receptive_field': rf
        }
        
        print(f"{name} | Params: {results[name]['params']} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Train Time: {train_time:.2f}s | "
              f"RF: {rf}")
    
    return results

# 2.2 Анализ глубины сети
def depth_analysis():
    models = {
        'Shallow': ShallowCNN().to(device),
        'Medium': MediumCNN().to(device),
        'Deep': DeepCNN().to(device),
        'ResNet': ResNetCNN().to(device)
    }
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for name, model in models.items():
        print(f"\nAnalyzing depth: {name}")
        
        # Обучение
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_loss, train_acc, train_time = train_model(
            model, train_loader, criterion, optimizer, device, EPOCHS)
        
        # Оценка
        test_loss, test_acc, _ = evaluate_model(
            model, test_loader, criterion, device)
        
        # Анализ градиентов
        gradients = analyze_gradients(model, train_loader, device)
        
        # Визуализация карт признаков
        sample, _ = next(iter(test_loader))
        visualize_feature_maps(model, sample[:1].to(device), name)
        
        # Сохранение результатов
        results[name] = {
            'params': count_parameters(model),
            'train_acc': train_acc[-1],
            'test_acc': test_acc,
            'train_time': train_time,
            'gradients': gradients
        }
        
        # Анализ vanishing/exploding gradients
        grad_means = [g[1] for g in gradients]
        min_grad = min(grad_means)
        max_grad = max(grad_means)
        
        print(f"{name} | Params: {results[name]['params']} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Train Time: {train_time:.2f}s")
        print(f"Gradient range: {min_grad:.6f} to {max_grad:.6f}")
    
    return results

# Основной блок
if __name__ == "__main__":
    print("="*50)
    print("Starting Kernel Size Analysis")
    print("="*50)
    kernel_results = kernel_size_analysis()
    
    print("\n" + "="*50)
    print("Starting Depth Analysis")
    print("="*50)
    depth_results = depth_analysis()
    
    # Сохранение результатов
    with open(RESULTS_PATH + '/kernel_analysis.txt', 'w') as f:
        for model, res in kernel_results.items():
            f.write(f"{model}: Params={res['params']}, Test Acc={res['test_acc']:.2f}%, "
                    f"Train Time={res['train_time']:.2f}s, RF={res['receptive_field']}\n")
    
    with open(RESULTS_PATH + '/depth_analysis.txt', 'w') as f:
        for model, res in depth_results.items():
            f.write(f"{model}: Params={res['params']}, Test Acc={res['test_acc']:.2f}%, "
                    f"Train Time={res['train_time']:.2f}s\n")
            for layer, grad in res['gradients']:
                f.write(f"  {layer}: grad={grad:.6f}\n")