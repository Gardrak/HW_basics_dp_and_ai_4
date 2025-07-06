import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from Settings import *

# Установка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Гиперпараметры
RESULT_CUSTOM_LYAERS_PATH = RESULTS_PATH + '/custom_layers'
RESULT_RESIDUAL_BLOCKS_PATH = RESULTS_PATH + '/residual_blocks'

# Создаем директории для результатов
os.makedirs(RESULT_CUSTOM_LYAERS_PATH, exist_ok=True)
os.makedirs(RESULT_RESIDUAL_BLOCKS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# 3.1 Кастомные слои
class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с нормализацией весов и пороговой активацией"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, threshold=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        
        # Инициализация весов
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # Применяем свертку
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        
        # Применяем пороговую активацию
        x = torch.where(x > self.threshold, x, torch.zeros_like(x))
        return x

class AttentionBlock(nn.Module):
    """Простой механизм внимания для CNN"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Swish(nn.Module):
    """Кастомная функция активации Swish: x * sigmoid(x)"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)

class L2Pooling(nn.Module):
    """Кастомный pooling слой на основе L2-нормы"""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        
    def forward(self, x):
        # Возводим в квадрат
        x_sq = x ** 2
        
        # Применяем average pooling
        window_area = self.kernel_size * self.kernel_size
        x_pool = F.avg_pool2d(x_sq, self.kernel_size, self.stride, self.padding)
        
        # Берем квадратный корень
        return torch.sqrt(x_pool * window_area)

# 3.2 Residual блоки
class BasicResidualBlock(nn.Module):
    """Базовый Residual блок"""
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual блок"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResidualBlock(nn.Module):
    """Wide Residual блок с большим числом каналов"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Утилиты для тестирования кастомных слоев
def test_custom_layers():
    """Тестирование кастомных слоев на простых примерах"""
    results = {}
    
    # Тестирование CustomConv2d
    custom_conv = CustomConv2d(3, 16, kernel_size=3, padding=1, threshold=0.2)
    std_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    
    x = torch.randn(2, 3, 32, 32)
    custom_out = custom_conv(x)
    std_out = std_conv(x)
    
    results['CustomConv'] = {
        'output_shape': custom_out.shape,
        'min_val': custom_out.min().item(),
        'threshold_effect': (custom_out == 0).float().mean().item()
    }
    
    # Тестирование AttentionBlock
    attention = AttentionBlock(16)
    x = torch.randn(2, 16, 32, 32)
    attn_out = attention(x)
    
    results['Attention'] = {
        'output_shape': attn_out.shape,
        'min_val': attn_out.min().item(),
        'max_val': attn_out.max().item()
    }
    
    # Тестирование Swish
    swish = Swish()
    x = torch.randn(5)
    swish_out = swish(x)
    sigmoid_out = torch.sigmoid(x)
    
    results['Swish'] = {
        'input': x.tolist(),
        'output': swish_out.tolist(),
        'sigmoid': sigmoid_out.tolist()
    }
    
    # Тестирование L2Pooling
    l2pool = L2Pooling(kernel_size=2, stride=2)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    x = torch.randn(2, 16, 32, 32)
    l2_out = l2pool(x)
    max_out = maxpool(x)
    avg_out = avgpool(x)
    
    results['L2Pooling'] = {
        'output_shape': l2_out.shape,
        'l2_mean': l2_out.mean().item(),
        'max_mean': max_out.mean().item(),
        'avg_mean': avg_out.mean().item()
    }
    
    return results

# Модели для экспериментов с Residual блоками
class ResNetModel(nn.Module):
    """Базовая модель ResNet с настраиваемыми блоками"""
    def __init__(self, block_type, num_blocks, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Создаем слои
        self.layer1 = self._make_layer(block_type, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_type, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, block_type, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            if block_type == 'basic':
                blocks.append(BasicResidualBlock(self.in_channels, out_channels, stride))
            elif block_type == 'bottleneck':
                blocks.append(BottleneckResidualBlock(self.in_channels, out_channels, stride))
            elif block_type == 'wide':
                blocks.append(WideResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Утилиты для обучения
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.train()
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
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
        
        # Оценка на тестовом наборе
        model.eval()
        test_correct = 0
        test_total = 0
        test_running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_epoch_loss = test_running_loss / len(test_loader)
        test_epoch_acc = 100. * test_correct / test_total
        test_losses.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        
        model.train()
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% | '
              f'Test Loss: {test_epoch_loss:.4f}, Acc: {test_epoch_acc:.2f}%')
    
    training_time = time.time() - start_time
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'training_time': training_time
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    
    # Собираем градиенты
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append((name, param.grad.abs().mean().item()))
    
    return gradients

# Эксперименты с Residual блоками
def residual_block_experiment():
    """Сравнение различных Residual блоков"""
    # Загрузка CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    block_types = {
        'Basic': ('basic', [2, 2, 2]),
        'Bottleneck': ('bottleneck', [2, 2, 2]),
        'Wide': ('wide', [2, 2, 2])
    }
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for name, (block_type, num_blocks) in block_types.items():
        print(f"\nTraining model with {name} blocks")
        model = ResNetModel(block_type, num_blocks).to(device)
        
        # Анализ параметров
        params = count_parameters(model)
        
        # Обучение
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        history = train_model(
            model, train_loader, test_loader, criterion, optimizer, device, EPOCHS)
        
        # Анализ градиентов
        gradients = analyze_gradients(model, train_loader, device)
        grad_means = [g[1] for g in gradients]
        min_grad = min(grad_means)
        max_grad = max(grad_means)
        
        # Сохранение результатов
        results[name] = {
            'params': params,
            'train_acc': history['train_acc'][-1],
            'test_acc': history['test_acc'][-1],
            'train_time': history['training_time'],
            'min_grad': min_grad,
            'max_grad': max_grad,
            'grad_ratio': max_grad / min_grad if min_grad > 0 else float('inf')
        }
        
        print(f"{name} | Params: {params} | "
              f"Test Acc: {results[name]['test_acc']:.2f}% | "
              f"Train Time: {results[name]['train_time']:.2f}s | "
              f"Gradient Ratio: {results[name]['grad_ratio']:.2f}")
    
    return results

# Основной блок
if __name__ == "__main__":
    print("="*50)
    print("Testing Custom Layers")
    print("="*50)
    custom_layer_results = test_custom_layers()
    
    # Сохранение результатов тестирования кастомных слоев
    with open(RESULT_CUSTOM_LYAERS_PATH + '/test_results.txt', 'w') as f:
        for layer, res in custom_layer_results.items():
            f.write(f"{layer}:\n")
            for key, val in res.items():
                f.write(f"  {key}: {val}\n")
    
    print("\n" + "="*50)
    print("Starting Residual Block Experiments")
    print("="*50)
    residual_results = residual_block_experiment()
    
    # Сохранение результатов экспериментов с Residual блоками
    with open(RESULT_RESIDUAL_BLOCKS_PATH + '/comparison.txt', 'w') as f:
        for block, res in residual_results.items():
            f.write(f"{block}:\n")
            f.write(f"  Parameters: {res['params']}\n")
            f.write(f"  Test Accuracy: {res['test_acc']:.2f}%\n")
            f.write(f"  Training Time: {res['train_time']:.2f}s\n")
            f.write(f"  Min Gradient: {res['min_grad']:.6f}\n")
            f.write(f"  Max Gradient: {res['max_grad']:.6f}\n")
            f.write(f"  Gradient Ratio: {res['grad_ratio']:.2f}\n")
    
    print("\nExperiments completed!")