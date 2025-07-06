import time
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        
        test_loss, test_accuracy, _ , _ = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_acc.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Acc: {test_accuracy:.2f}%')
    
    training_time = time.time() - start_time
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'training_time': training_time
    }

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    inference_time = time.time() - start_time
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    return test_loss, accuracy, inference_time, cm

def get_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients.append(param.grad.cpu().numpy().flatten())
    return np.concatenate(gradients)