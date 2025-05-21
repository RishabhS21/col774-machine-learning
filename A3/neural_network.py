import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

# Define paths
TRAIN_DIR = './data2/train'
TEST_DIR = './data2/test'
TEST_LABELS_PATH = './data2/test_labels.csv'

def avg(arr):
    return sum(arr) / len(arr)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    return (a > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, n_features, hidden_layers, n_classes, activation='sigmoid'):
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.activation_type = activation
        self.weights = {}
        self.biases = {}
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # Initialize weights and biases
        layer_sizes = [self.n_features] + self.hidden_layers + [self.n_classes]
        
        for i in range(1, len(layer_sizes)):
            # He initialization for ReLU, Xavier for sigmoid
            if self.activation_type == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i-1])
            else:  # sigmoid
                scale = np.sqrt(1.0 / layer_sizes[i-1])
                
            self.weights[i] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * scale
            self.biases[i] = np.zeros((1, layer_sizes[i]))
    
    def activation(self, z):
        if self.activation_type == 'relu':
            return relu(z)
        else:  # sigmoid
            return sigmoid(z)
    
    def activation_derivative(self, a):
        if self.activation_type == 'relu':
            return relu_derivative(a)
        else:  # sigmoid
            return sigmoid_derivative(a)
    
    def forward_propagation(self, X):
        self.layer_inputs = {}
        self.layer_outputs = {}
        
        # Input layer
        self.layer_outputs[0] = X
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers) + 1):
            self.layer_inputs[i] = np.dot(self.layer_outputs[i-1], self.weights[i]) + self.biases[i]
            self.layer_outputs[i] = self.activation(self.layer_inputs[i])
        
        # Output layer
        i = len(self.hidden_layers) + 1
        self.layer_inputs[i] = np.dot(self.layer_outputs[i-1], self.weights[i]) + self.biases[i]
        self.layer_outputs[i] = softmax(self.layer_inputs[i])
        
        return self.layer_outputs[i]
    
    def compute_cost(self, y_pred, y_true):
        # Cross-entropy loss
        m = y_true.shape[0]
        cost = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
        return cost
    
    def backward_propagation(self, X, y):
        m = X.shape[0]
        n_layers = len(self.hidden_layers) + 1
        
        # Initialize gradients
        dW = {}
        db = {}
        
        # Output layer error
        dZ = self.layer_outputs[n_layers] - y
        
        # Compute gradients for output layer
        dW[n_layers] = np.dot(self.layer_outputs[n_layers-1].T, dZ) / m
        db[n_layers] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Backpropagate through hidden layers
        for i in range(n_layers-1, 0, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * self.activation_derivative(self.layer_outputs[i])
            dW[i] = np.dot(self.layer_outputs[i-1].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m
        
        return dW, db
    
    def update_parameters(self, dW, db, learning_rate):
        for i in range(1, len(self.hidden_layers) + 2):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]
    
    def train(self, X, y, X_val, y_val, batch_size=32, learning_rate=0.01, epochs=100, 
              adaptive_lr=False, verbose=True):
        m = X.shape[0]
        n_batches = m // batch_size
        
        # For tracking metrics
        train_costs = []
        val_costs = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Adjust learning rate if adaptive
            if adaptive_lr:
                current_lr = learning_rate / np.sqrt(epoch + 1)
            else:
                current_lr = learning_rate
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward and backward pass
                y_pred = self.forward_propagation(X_batch)
                dW, db = self.backward_propagation(X_batch, y_batch)
                self.update_parameters(dW, db, current_lr)
            
            # Compute metrics at the end of each epoch
            y_pred_train = self.forward_propagation(X)
            train_cost = self.compute_cost(y_pred_train, y)
            train_costs.append(train_cost)
            
            y_pred_val = self.forward_propagation(X_val)
            val_cost = self.compute_cost(y_pred_val, y_val)
            val_costs.append(val_cost)
            
            train_accuracy = self.accuracy(X, y)
            val_accuracy = self.accuracy(X_val, y_val)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Cost: {train_cost:.6f}, "
                      f"Val Cost: {val_cost:.6f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return {
            'train_costs': train_costs,
            'val_costs': val_costs,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = np.mean(y_pred == y_true)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

# Data loading and preprocessing
class GTSRBDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

def load_data(train_data_path, test_data_path):
    # Load training data
    train_images = []
    train_labels = []
    
    for class_dir in os.listdir(train_data_path):
        # if class_dir.startswith('000'):
        class_id = int(class_dir)
        class_dir_path = os.path.join(train_data_path, class_dir)
        
        for img_name in os.listdir(class_dir_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(class_dir_path, img_name)
                train_images.append(img_path)
                train_labels.append(class_id)
    
    # Load test data
    test_dir = os.path.dirname(test_data_path)
    test_labels_path = os.path.join(test_dir, '../test_labels.csv')
    test_labels_df = pd.read_csv(test_labels_path)
    test_images = []
    test_labels = []
    
    for idx, row in test_labels_df.iterrows():
        img_name = row['image']
        label = row['label']
        img_path = os.path.join(test_data_path, img_name)
        
        if os.path.exists(img_path):
            test_images.append(img_path)
            test_labels.append(label)
    
    # Create transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = GTSRBDataset(train_images, train_labels, transform)
    test_dataset = GTSRBDataset(test_images, test_labels, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Extract data for our custom NN implementation
    X_train = []
    y_train = []
    
    for images, labels in train_loader:
        batch_images = images.numpy()
        batch_images = batch_images.reshape(batch_images.shape[0], -1)  # Flatten
        X_train.append(batch_images)
        y_train.append(labels.numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    X_test = []
    y_test = []
    
    for images, labels in test_loader:
        batch_images = images.numpy()
        batch_images = batch_images.reshape(batch_images.shape[0], -1)  # Flatten
        X_test.append(batch_images)
        y_test.append(labels.numpy())
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    
    # Split training data into train and validation
    indices = np.random.permutation(len(X_train))
    split = int(0.8 * len(X_train))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train_final = X_train[train_idx]
    y_train_final = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    # One-hot encode labels
    num_classes = 43  # GTSRB has 43 classes
    y_train_onehot = np.zeros((len(y_train_final), num_classes))
    y_train_onehot[np.arange(len(y_train_final)), y_train_final] = 1
    
    y_val_onehot = np.zeros((len(y_val), num_classes))
    y_val_onehot[np.arange(len(y_val)), y_val] = 1
    
    y_test_onehot = np.zeros((len(y_test), num_classes))
    y_test_onehot[np.arange(len(y_test)), y_test] = 1
    
    return X_train_final, y_train_final, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot

def plot_metrics(results, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(10, 6))
    
    for label, values in results.items():
        plt.plot(values['x'], values['y'], marker='o', linestyle='-', label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def generate_classification_report(y_true, y_pred, target_names=None):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Extract precision, recall, and F1 score for each class
    classes_report = {}
    
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            classes_report[class_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score']
            }
    
    return classes_report, report['macro avg']['f1-score']

def save_predictions(y_pred, output_path, filename):
    output_folder = os.path.join(output_path, filename)
    pd.DataFrame({"prediction": y_pred}).to_csv(output_folder, index=False)

# Experiment functions
def experiment_b(X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, output_path):
    """
    Experiment (b): Varying number of hidden layer units in a single hidden layer
    """
    print("\n=== Experiment (b): Effect of Hidden Layer Size ===")
    
    n_features = X_train.shape[1]
    n_classes = y_train_onehot.shape[1]
    
    # Hidden layer sizes to test
    hidden_sizes = [1, 5, 10, 50, 100]
    
    results = {
        'train_accuracy': {'x': hidden_sizes, 'y': []},
        'test_accuracy': {'x': hidden_sizes, 'y': []},
        'train_f1': {'x': hidden_sizes, 'y': []},
        'test_f1': {'x': hidden_sizes, 'y': []}
    }
    y_test_pred_final = None
    for size in hidden_sizes:
        print(f"\nTraining with hidden layer size: {size}")
        
        # Create and train model
        model = NeuralNetwork(n_features, [size], n_classes, activation='sigmoid')
        history = model.train(X_train, y_train_onehot, X_val, y_val_onehot, 
                             batch_size=32, learning_rate=0.01, epochs=50)
        
        # Evaluate on train set
        y_train_pred = model.predict(X_train)
        _, train_f1 = generate_classification_report(y_train, y_train_pred)
        train_accuracy = model.accuracy(X_train, y_train_onehot)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        _, test_f1 = generate_classification_report(y_test, y_test_pred)
        test_accuracy = model.accuracy(X_test, y_test_onehot)
        
        # Store results
        y_test_pred_final = y_test_pred
        results['train_accuracy']['y'].append(train_accuracy)
        results['test_accuracy']['y'].append(test_accuracy)
        results['train_f1']['y'].append(train_f1)
        results['test_f1']['y'].append(test_f1)
        
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")
    # Save predictions
    save_predictions(y_test_pred_final, output_path, f"prediction_b.csv")
    
    # Plot results
    plot_metrics(
        {'Train Accuracy': results['train_accuracy'], 'Test Accuracy': results['test_accuracy']},
        'Accuracy vs Hidden Layer Size',
        'Number of Hidden Units',
        'Accuracy',
        'experiment_b_accuracy.png'
    )
    
    plot_metrics(
        {'Train F1 Score': results['train_f1'], 'Test F1 Score': results['test_f1']},
        'F1 Score vs Hidden Layer Size',
        'Number of Hidden Units',
        'F1 Score',
        'experiment_b_f1.png'
    )
    
    return results

def experiment_c(X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, output_path):
    """
    Experiment (c): Varying network depth (number of hidden layers)
    """
    print("\n=== Experiment (c): Effect of Network Depth ===")
    
    n_features = X_train.shape[1]
    n_classes = y_train_onehot.shape[1]
    
    # Network architectures to test
    architectures = [
        [512],                  # 1 hidden layer
        [512, 256],             # 2 hidden layers
        [512, 256, 128],        # 3 hidden layers
        [512, 256, 128, 64]     # 4 hidden layers
    ]
    
    depths = [len(arch) for arch in architectures]
    
    results = {
        'train_accuracy': {'x': depths, 'y': []},
        'test_accuracy': {'x': depths, 'y': []},
        'train_f1': {'x': depths, 'y': []},
        'test_f1': {'x': depths, 'y': []}
    }
    y_test_pred_final = None
    for i, arch in enumerate(architectures):
        print(f"\nTraining with architecture: {arch}")
        
        # Create and train model
        model = NeuralNetwork(n_features, arch, n_classes, activation='sigmoid')
        history = model.train(X_train, y_train_onehot, X_val, y_val_onehot, 
                             batch_size=32, learning_rate=0.01, epochs=50)
        
        # Evaluate on train set
        y_train_pred = model.predict(X_train)
        _, train_f1 = generate_classification_report(y_train, y_train_pred)
        train_accuracy = model.accuracy(X_train, y_train_onehot)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        _, test_f1 = generate_classification_report(y_test, y_test_pred)
        test_accuracy = model.accuracy(X_test, y_test_onehot)
        
        # Store results
        y_test_pred_final = y_test_pred
        results['train_accuracy']['y'].append(train_accuracy)
        results['test_accuracy']['y'].append(test_accuracy)
        results['train_f1']['y'].append(train_f1)
        results['test_f1']['y'].append(test_f1)
        
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")

    save_predictions(y_test_pred_final, output_path, f"prediction_c.csv")
    # Plot results
    plot_metrics(
        {'Train Accuracy': results['train_accuracy'], 'Test Accuracy': results['test_accuracy']},
        'Accuracy vs Network Depth',
        'Network Depth (Number of Hidden Layers)',
        'Accuracy',
        'experiment_c_accuracy.png'
    )
    
    plot_metrics(
        {'Train F1 Score': results['train_f1'], 'Test F1 Score': results['test_f1']},
        'F1 Score vs Network Depth',
        'Network Depth (Number of Hidden Layers)',
        'F1 Score',
        'experiment_c_f1.png'
    )
    
    return results

def experiment_d(X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, output_path):
    """
    Experiment (d): Adaptive learning rate with varying network depth
    """
    print("\n=== Experiment (d): Adaptive Learning Rate with Varying Network Depth ===")
    
    n_features = X_train.shape[1]
    n_classes = y_train_onehot.shape[1]
    
    # Network architectures to test
    architectures = [
        [512],                  # 1 hidden layer
        [512, 256],             # 2 hidden layers
        [512, 256, 128],        # 3 hidden layers
        [512, 256, 128, 64]     # 4 hidden layers
    ]
    
    depths = [len(arch) for arch in architectures]
    
    results = {
        'train_accuracy': {'x': depths, 'y': []},
        'test_accuracy': {'x': depths, 'y': []},
        'train_f1': {'x': depths, 'y': []},
        'test_f1': {'x': depths, 'y': []}
    }
    y_test_pred_final = None
    for i, arch in enumerate(architectures):
        print(f"\nTraining with architecture: {arch}")
        
        # Create and train model with adaptive learning rate
        model = NeuralNetwork(n_features, arch, n_classes, activation='sigmoid')
        history = model.train(X_train, y_train_onehot, X_val, y_val_onehot, 
                             batch_size=32, learning_rate=0.01, epochs=50, 
                             adaptive_lr=True)  # Use adaptive learning rate
        
        # Evaluate on train set
        y_train_pred = model.predict(X_train)
        _, train_f1 = generate_classification_report(y_train, y_train_pred)
        train_accuracy = model.accuracy(X_train, y_train_onehot)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        _, test_f1 = generate_classification_report(y_test, y_test_pred)
        test_accuracy = model.accuracy(X_test, y_test_onehot)
        
        # Store results
        y_test_pred_final = y_test_pred
        results['train_accuracy']['y'].append(train_accuracy)
        results['test_accuracy']['y'].append(test_accuracy)
        results['train_f1']['y'].append(train_f1)
        results['test_f1']['y'].append(test_f1)
        
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")

    save_predictions(y_test_pred_final, output_path, f"prediction_d.csv")
    
    # Plot results
    plot_metrics(
        {'Train Accuracy': results['train_accuracy'], 'Test Accuracy': results['test_accuracy']},
        'Accuracy vs Network Depth (Adaptive LR)',
        'Network Depth (Number of Hidden Layers)',
        'Accuracy',
        'experiment_d_accuracy.png'
    )
    
    plot_metrics(
        {'Train F1 Score': results['train_f1'], 'Test F1 Score': results['test_f1']},
        'F1 Score vs Network Depth (Adaptive LR)',
        'Network Depth (Number of Hidden Layers)',
        'F1 Score',
        'experiment_d_f1.png'
    )
    
    return results

def experiment_e(X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, output_path):
    """
    Experiment (e): ReLU activation with adaptive learning rate
    """
    print("\n=== Experiment (e): ReLU Activation with Adaptive Learning Rate ===")
    
    n_features = X_train.shape[1]
    n_classes = y_train_onehot.shape[1]
    
    # Network architectures to test
    architectures = [
        [512],                  # 1 hidden layer
        [512, 256],             # 2 hidden layers
        [512, 256, 128],        # 3 hidden layers
        [512, 256, 128, 64]     # 4 hidden layers
    ]
    
    depths = [len(arch) for arch in architectures]
    
    results = {
        'train_accuracy': {'x': depths, 'y': []},
        'test_accuracy': {'x': depths, 'y': []},
        'train_f1': {'x': depths, 'y': []},
        'test_f1': {'x': depths, 'y': []}
    }
    y_test_pred_final = None
    for i, arch in enumerate(architectures):
        print(f"\nTraining with architecture: {arch}")
        
        # Create and train model with ReLU activation and adaptive learning rate
        model = NeuralNetwork(n_features, arch, n_classes, activation='relu')
        history = model.train(X_train, y_train_onehot, X_val, y_val_onehot, 
                             batch_size=32, learning_rate=0.01, epochs=50, 
                             adaptive_lr=True)
        
        # Evaluate on train set
        y_train_pred = model.predict(X_train)
        _, train_f1 = generate_classification_report(y_train, y_train_pred)
        train_accuracy = model.accuracy(X_train, y_train_onehot)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        _, test_f1 = generate_classification_report(y_test, y_test_pred)
        test_accuracy = model.accuracy(X_test, y_test_onehot)
        
        # Store results
        y_test_pred_final = y_test_pred
        results['train_accuracy']['y'].append(train_accuracy)
        results['test_accuracy']['y'].append(test_accuracy)
        results['train_f1']['y'].append(train_f1)
        results['test_f1']['y'].append(test_f1)
        
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")

    save_predictions(y_test_pred_final, output_path, f"prediction_e.csv")
    # Plot results
    plot_metrics(
        {'Train Accuracy': results['train_accuracy'], 'Test Accuracy': results['test_accuracy']},
        'Accuracy vs Network Depth (ReLU + Adaptive LR)',
        'Network Depth (Number of Hidden Layers)',
        'Accuracy',
        'experiment_e_accuracy.png'
    )
    
    plot_metrics(
        {'Train F1 Score': results['train_f1'], 'Test F1 Score': results['test_f1']},
        'F1 Score vs Network Depth (ReLU + Adaptive LR)',
        'Network Depth (Number of Hidden Layers)',
        'F1 Score',
        'experiment_e_f1.png'
    )
    
    return results

def experiment_f(X_train, y_train, X_val, y_val, X_test, y_test, output_path):
    """
    Experiment (f): scikit-learn MLPClassifier implementation
    """
    print("\n=== Experiment (f): scikit-learn MLPClassifier ===")
    
    # Network architectures to test
    architectures = [
        (512,),                     # 1 hidden layer
        (512, 256),                 # 2 hidden layers
        (512, 256, 128),            # 3 hidden layers
        (512, 256, 128, 64)         # 4 hidden layers
    ]
    
    depths = [len(arch) for arch in architectures]
    
    results = {
        'train_accuracy': {'x': depths, 'y': []},
        'test_accuracy': {'x': depths, 'y': []},
        'train_f1': {'x': depths, 'y': []},
        'test_f1': {'x': depths, 'y': []}
    }
    y_test_pred_final = None
    for i, arch in enumerate(architectures):
        print(f"\nTraining with architecture: {arch}")
        
        # Create and train scikit-learn MLPClassifier
        mlp = MLPClassifier(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='sgd',
            alpha=0,
            batch_size=32,
            learning_rate='invscaling',
            learning_rate_init=0.01,
            max_iter=50,
            random_state=42
        )
        
        # Train the model
        mlp.fit(X_train, y_train)
        
        # Evaluate on train set
        y_train_pred = mlp.predict(X_train)
        _, train_f1 = generate_classification_report(y_train, y_train_pred)
        train_accuracy = mlp.score(X_train, y_train)
        
        # Evaluate on test set
        y_test_pred = mlp.predict(X_test)
        _, test_f1 = generate_classification_report(y_test, y_test_pred)
        test_accuracy = mlp.score(X_test, y_test)
        
        # Store results
        y_test_pred_final = y_test_pred
        results['train_accuracy']['y'].append(train_accuracy)
        results['test_accuracy']['y'].append(test_accuracy)
        results['train_f1']['y'].append(train_f1)
        results['test_f1']['y'].append(test_f1)
        
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")

    save_predictions(y_test_pred_final, output_path, f"prediction_f.csv")
    # Plot results
    plot_metrics(
        {'Train Accuracy': results['train_accuracy'], 'Test Accuracy': results['test_accuracy']},
        'Accuracy vs Network Depth (scikit-learn MLPClassifier)',
        'Network Depth (Number of Hidden Layers)',
        'Accuracy',
        'experiment_f_accuracy.png'
    )
    
    plot_metrics(
        {'Train F1 Score': results['train_f1'], 'Test F1 Score': results['test_f1']},
        'F1 Score vs Network Depth (scikit-learn MLPClassifier)',
        'Network Depth (Number of Hidden Layers)',
        'F1 Score',
        'experiment_f_f1.png'
    )
    
    return results

def compare_experiments(exp_c_results, exp_d_results, exp_e_results, exp_f_results):
    """
    Compare results from different experiments
    """
    print("\n=== Comparison of Experiments ===")
    
    # Network depths
    depths = [1, 2, 3, 4]
    
    # Compare test accuracies
    plt.figure(figsize=(12, 6))
    plt.plot(depths, exp_c_results['test_accuracy']['y'], 'o-', label='Sigmoid (Fixed LR)')
    plt.plot(depths, exp_d_results['test_accuracy']['y'], 's-', label='Sigmoid (Adaptive LR)')
    plt.plot(depths, exp_e_results['test_accuracy']['y'], '^-', label='ReLU (Adaptive LR)')
    plt.plot(depths, exp_f_results['test_accuracy']['y'], 'D-', label='scikit-learn MLPClassifier')
    plt.xlabel('Network Depth (Number of Hidden Layers)')
    plt.ylabel('Test Accuracy')
    plt.title('Comparison of Test Accuracies Across Experiments')
    plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    plt.savefig('comparison_test_accuracy.png')
    plt.show()
    
    # Compare test F1 scores
    plt.figure(figsize=(12, 6))
    plt.plot(depths, exp_c_results['test_f1']['y'], 'o-', label='Sigmoid (Fixed LR)')
    plt.plot(depths, exp_d_results['test_f1']['y'], 's-', label='Sigmoid (Adaptive LR)')
    plt.plot(depths, exp_e_results['test_f1']['y'], '^-', label='ReLU (Adaptive LR)')
    plt.plot(depths, exp_f_results['test_f1']['y'], 'D-', label='scikit-learn MLPClassifier')
    plt.xlabel('Network Depth (Number of Hidden Layers)')
    plt.ylabel('Test F1 Score')
    plt.title('Comparison of Test F1 Scores Across Experiments')
    plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    plt.savefig('comparison_test_f1.png')
    plt.show()

def main():
    # Parse command line arguments
    if len(sys.argv) != 5:
        print("Usage: python neural_network.py <train_data_path> <test_data_path> <output_folder_path> <question_part>")
        sys.exit(1)
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_path = sys.argv[3]
    question_part = sys.argv[4].lower()
    # make output_path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot = load_data(train_data_path, test_data_path)    
    
    # Run experiments
    if question_part == 'a':
        print("\n=== Experiment (a): Data Loading and Preprocessing ===")
        print("Data loaded successfully!")
    elif question_part == 'b':
        exp_b_results = experiment_b(X_train, y_train, X_val, y_val, X_test, y_test, 
                                y_train_onehot, y_val_onehot, y_test_onehot, output_path)
    
    elif question_part == 'c':
        exp_c_results = experiment_c(X_train, y_train, X_val, y_val, X_test, y_test, 
                                y_train_onehot, y_val_onehot, y_test_onehot, output_path)
    elif question_part == 'd':
        exp_d_results = experiment_d(X_train, y_train, X_val, y_val, X_test, y_test, 
                                y_train_onehot, y_val_onehot, y_test_onehot, output_path)
    
    elif question_part == 'e':
        exp_e_results = experiment_e(X_train, y_train, X_val, y_val, X_test, y_test, 
                                y_train_onehot, y_val_onehot, y_test_onehot, output_path)
    elif question_part == 'f':
        exp_f_results = experiment_f(X_train, y_train, X_val, y_val, X_test, y_test, output_path)
    else:
        print("Invalid question part. Please specify 'a', 'b', 'c', 'd', 'e', or 'f'.")
    return
    
if __name__ == "__main__":
    import sys
    main()
