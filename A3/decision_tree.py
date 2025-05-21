import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import os
import sys

# Define the types of attributes (categorical or continuous)
def identify_attribute_types(data):
    types = []
    for col in data.columns[:-1]:  # Exclude the target column
        if data[col].dtype == 'object':
            types.append("cat")
        else:
            types.append("cont")
    return types

# Calculate entropy
def entropy(y):
    if len(y) == 0:
        return 0
    p1 = np.sum(y == '>50K') / len(y)
    p0 = 1 - p1
    if p1 == 0 or p0 == 0:
        return 0
    return -p1 * np.log2(p1) - p0 * np.log2(p0)

# Calculate overall entropy after splitting
def calc_overall_entropy(splits):
    total = 0
    size = 0
    for split in splits:
        if len(split) > 0:
            total += entropy(split) * len(split)
            size += len(split)
    if size == 0:
        return 0
    return total / size

# Check if a node is pure (all examples have the same class)
def check_pure(y):
    return len(np.unique(y)) == 1

# Find the best attribute to split on
def find_best_split(X, y, types):
    best_gain = 0
    best_split = -1
    
    if check_pure(y):
        return -1
    
    for column in range(X.shape[1]):
        if types[column] == "cont":
            # Split on median for continuous attributes
            median = np.median(X[:, column])
            left_indices = X[:, column] <= median
            right_indices = X[:, column] > median
            
            splits = [y[left_indices], y[right_indices]]
            
            if len(splits[0]) == 0 or len(splits[1]) == 0:
                continue
                
            gain = entropy(y) - calc_overall_entropy(splits)
            
            if gain > best_gain:
                best_gain = gain
                best_split = column
                best_threshold = median
        else:
            # Split on categories for categorical attributes
            unique_values = np.unique(X[:, column])
            splits = []
            
            for value in unique_values:
                splits.append(y[X[:, column] == value])
                
            gain = entropy(y) - calc_overall_entropy(splits)
            
            if gain > best_gain:
                best_gain = gain
                best_split = column
    
    return best_split

# Decision Tree Node class
class DTNode:
    def __init__(self, depth, is_leaf=False, value=None, column=None):
        self.depth = depth
        self.children = []
        self.is_leaf = is_leaf
        self.value = value
        self.column = column
        self.threshold = None
        self.vals = None
        
    def get_children(self, X, types):  # Added types parameter
        if self.is_leaf:
            return None
            
        column = self.column
        
        if types[column] == "cont":
            # For continuous attributes
            if X[column] <= self.threshold:
                return self.children[0]
            else:
                return self.children[1]
        else:
            # For categorical attributes
            if X[column] in self.vals:
                return self.children[self.vals.index(X[column])]
            else:
                # If value not seen during training, treat as leaf
                return None

# Decision Tree class
class DTTree:
    def __init__(self):
        self.root = None
        self.types = None  # Store types as instance variable
        
    def fit(self, X, y, types, max_depth=5):
        self.types = types  # Save types for prediction
        self.root = self._build_tree(X, y, types, max_depth)
        return self
        
    def _build_tree(self, X, y, types, max_depth, depth=0):
        root = DTNode(depth=depth)
        
        # If max depth reached or node is pure, make it a leaf
        if depth == max_depth or check_pure(y):
            root.is_leaf = True
            root.value = '>50K' if np.sum(y == '>50K') > np.sum(y == '<=50K') else '<=50K'
            return root
            
        # Find the best attribute to split on
        split = find_best_split(X, y, types)
        
        if split == -1:
            # If no good split found, make it a leaf
            root.is_leaf = True
            root.value = '>50K' if np.sum(y == '>50K') > np.sum(y == '<=50K') else '<=50K'
            return root
            
        root.column = split
        
        if types[split] == "cont":
            # For continuous attributes
            root.threshold = np.median(X[:, split])
            left_indices = X[:, split] <= root.threshold
            right_indices = X[:, split] > root.threshold
            
            # Create child nodes
            root.children = [
                self._build_tree(X[left_indices], y[left_indices], types, max_depth, depth+1),
                self._build_tree(X[right_indices], y[right_indices], types, max_depth, depth+1)
            ]
        else:
            # For categorical attributes
            root.vals = np.unique(X[:, split]).tolist()
            
            # Create a child for each unique value
            for val in root.vals:
                indices = X[:, split] == val
                root.children.append(self._build_tree(X[indices], y[indices], types, max_depth, depth+1))
                
        # Set majority class as value (used when a test example can't reach a leaf)
        root.value = '>50K' if np.sum(y == '>50K') > np.sum(y == '<=50K') else '<=50K'
        
        return root
        
    def predict(self, X):
        y_pred = np.empty(X.shape[0], dtype=object)
        
        for i in range(X.shape[0]):
            node = self.root
            
            while not node.is_leaf:
                child = node.get_children(X[i], self.types)  # Pass types parameter
                if child is None:
                    break
                node = child
                
            y_pred[i] = node.value
            
        return y_pred
        
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)
        
    def count_nodes(self):
        """Count the total number of nodes in the tree"""
        if self.root is None:
            return 0
            
        count = 0
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            count += 1
            
            if not node.is_leaf:
                queue.extend(node.children)
                
        return count
        
    def prune(self, X_val, y_val, X_train, y_train, X_test, y_test):
        """Prune the tree to improve validation accuracy"""
        # Store the initial tree state
        nodes = [self.count_nodes()]
        training_acc = [self.accuracy(X_train, y_train)]
        test_acc = [self.accuracy(X_test, y_test)]
        val_acc = [self.accuracy(X_val, y_val)]
        
        best_tree = self
        current_tree = self
        
        # Continue pruning until no improvement is possible
        while True:
            current_tree, improved = current_tree._prune_one_node(X_val, y_val)
            
            if not improved:
                break
                
            nodes.append(current_tree.count_nodes())
            training_acc.append(current_tree.accuracy(X_train, y_train))
            test_acc.append(current_tree.accuracy(X_test, y_test))
            val_acc.append(current_tree.accuracy(X_val, y_val))
            
            best_tree = current_tree
        
        # Reverse the lists to show progression from fewer to more nodes
        nodes.reverse()
        training_acc.reverse()
        test_acc.reverse()
        val_acc.reverse()
        
        return best_tree, nodes, training_acc, test_acc, val_acc

        
    def _prune_one_node(self, X_val, y_val):
        """Prune one node that gives the maximum increase in validation accuracy"""
        best_accuracy = self.accuracy(X_val, y_val)
        best_node = None
        
        # Collect all non-leaf nodes
        nodes = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            
            if not node.is_leaf:
                nodes.append(node)
                queue.extend(node.children)
                
        # Try pruning each non-leaf node
        for node in nodes:
            # Temporarily make it a leaf
            was_leaf = node.is_leaf
            children = node.children
            
            node.is_leaf = True
            node.children = []
            
            # Check if accuracy improves
            accuracy = self.accuracy(X_val, y_val)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_node = node
            
            # Restore the node
            if not best_node == node:
                node.is_leaf = was_leaf
                node.children = children
                
        # If a node was found that improves accuracy, prune it
        if best_node:
            best_node.is_leaf = True
            best_node.children = []
            return self, True
        else:
            return self, False
        
# Load and preprocess data

def load_data(file_path):
    """Load data from CSV file and strip whitespaces from string values"""
    data = pd.read_csv(file_path)
    # Strip whitespace from string columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    return data

def preprocess_data(data):
    """Preprocess the data for decision tree learning"""
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    return X, y

def one_hot_encode(X_train, X_val, X_test):
    """Perform one-hot encoding for categorical features"""
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    # Apply one-hot encoding
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols)
    X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols)
    
    # Ensure all datasets have the same columns
    all_columns = set(X_train_encoded.columns).union(set(X_val_encoded.columns)).union(set(X_test_encoded.columns))
    
    for col in all_columns:
        if col not in X_train_encoded.columns:
            X_train_encoded[col] = 0
        if col not in X_val_encoded.columns:
            X_val_encoded[col] = 0
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0
    
    # Ensure columns are in the same order
    X_train_encoded = X_train_encoded[sorted(X_train_encoded.columns)]
    X_val_encoded = X_val_encoded[sorted(X_val_encoded.columns)]
    X_test_encoded = X_test_encoded[sorted(X_test_encoded.columns)]
    
    return X_train_encoded, X_val_encoded, X_test_encoded

# Train and evaluate decision tree
def experiment_1(X_train, y_train, X_test, y_test, types, output_path):
    """Experiment with different maximum depths for decision tree"""
    max_depths = [5, 10, 15, 20]
    train_accuracies = []
    test_accuracies = []
    
    for max_depth in tqdm(max_depths, desc="Training trees with different depths"):
        # Train decision tree
        tree = DTTree()
        tree.fit(X_train.values, y_train.values, types, max_depth=max_depth)
        
        # Calculate accuracies
        train_acc = tree.accuracy(X_train.values, y_train.values)
        test_acc = tree.accuracy(X_test.values, y_test.values)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Max Depth: {max_depth}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Calculate class-specific accuracies
        y_pred_test = tree.predict(X_test.values)
        
        # Accuracy for >50K predictions - with zero-division protection
        high_income_indices = y_test.values == '>50K'
        high_income_count = np.sum(high_income_indices)
        if high_income_count > 0:
            high_income_acc = np.sum(y_pred_test[high_income_indices] == '>50K') / high_income_count
        else:
            high_income_acc = 0.0
        
        # Accuracy for <=50K predictions - with zero-division protection
        low_income_indices = y_test.values == '<=50K'
        low_income_count = np.sum(low_income_indices)
        if low_income_count > 0:
            low_income_acc = np.sum(y_pred_test[low_income_indices] == '<=50K') / low_income_count
        else:
            low_income_acc = 0.0
        
        print(f"Accuracy for >50K: {high_income_acc:.4f}, Accuracy for <=50K: {low_income_acc:.4f}")
    
    # Save predictions to CSV named 'prediction_a.csv' in output_path dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pd.DataFrame({'prediction': y_pred_test}).to_csv(os.path.join(output_path, 'prediction_a.csv'), index=False)
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(max_depths, test_accuracies, marker='x', label='Testing Accuracy')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy vs. Maximum Depth')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_1_accuracies.png')
    plt.show()
    
    return train_accuracies, test_accuracies

def experiment_2(X_train_encoded, y_train, X_test_encoded, y_test, output_path):
    """Experiment with one-hot encoded data"""
    # Identify types for encoded data (all are continuous after one-hot encoding)
    types_encoded = ["cont"] * X_train_encoded.shape[1]
    
    max_depths = [25, 35, 45, 55]
    train_accuracies = []
    test_accuracies = []
    
    for max_depth in tqdm(max_depths, desc="Training trees with one-hot encoding"):
        # Train decision tree
        tree = DTTree()
        tree.fit(X_train_encoded.values, y_train.values, types_encoded, max_depth=max_depth)
        
        # Calculate accuracies
        train_acc = tree.accuracy(X_train_encoded.values, y_train.values)
        test_acc = tree.accuracy(X_test_encoded.values, y_test.values)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Max Depth: {max_depth}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # save predictions
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pd.DataFrame({'prediction': tree.predict(X_test_encoded.values)}).to_csv(os.path.join(output_path, 'prediction_b.csv'), index=False)
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(max_depths, test_accuracies, marker='x', label='Testing Accuracy')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy vs. Maximum Depth (One-Hot Encoded)')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_2_accuracies.png')
    plt.show()
    
    return train_accuracies, test_accuracies, max_depths

def experiment_3(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test, max_depths, output_path):
    """Experiment with post-pruning"""
    types_encoded = ["cont"] * X_train_encoded.shape[1]
    
    for max_depth in tqdm(max_depths, desc="Post-pruning trees"):
        # Train decision tree
        tree = DTTree()
        tree.fit(X_train_encoded.values, y_train.values, types_encoded, max_depth=max_depth)
        # Calculate accuracies BEFORE pruning
        train_acc_before = tree.accuracy(X_train_encoded.values, y_train.values)
        test_acc_before = tree.accuracy(X_test_encoded.values, y_test.values)
        val_acc_before = tree.accuracy(X_val_encoded.values, y_val.values)
        
        # Prune the tree
        pruned_tree, nodes, train_acc, test_acc, val_acc = tree.prune(
            X_val_encoded.values, y_val.values, 
            X_train_encoded.values, y_train.values, 
            X_test_encoded.values, y_test.values
        )
        # save the predictions in csv named 'prediction_c.csv'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pd.DataFrame({'prediction': pruned_tree.predict(X_test_encoded.values)}).to_csv(os.path.join(output_path, 'prediction_c.csv'), index=False)
        # Plot accuracies vs number of nodes
        plt.figure(figsize=(10, 6))
        plt.plot(nodes, train_acc, marker='o', label='Training Accuracy')
        plt.plot(nodes, test_acc, marker='x', label='Testing Accuracy')
        plt.plot(nodes, val_acc, marker='s', label='Validation Accuracy')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs. Number of Nodes (Max Depth: {max_depth})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/experiment_3_pruning_depth_{max_depth}.png')
        plt.show()
        
        print(f"Max Depth: {max_depth}")
        print(f"Before pruning - Train: {train_acc_before:.4f}, "
              f"Test: {test_acc_before:.4f}, "
              f"Val: {val_acc_before:.4f}")
        print(f"After pruning - Train: {train_acc[0]:.4f}, Test: {test_acc[0]:.4f}, Val: {val_acc[0]:.4f}")


def experiment_4(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test, output_path):
    """Experiment with scikit-learn's decision tree"""
    # Part (i): Vary max_depth
    max_depths = [25, 35, 45, 55]
    train_accuracies = []
    test_accuracies = []
    val_accuracies = []
    
    for max_depth in max_depths:
        # Train decision tree
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
        clf.fit(X_train_encoded, y_train)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, clf.predict(X_train_encoded))
        test_acc = accuracy_score(y_test, clf.predict(X_test_encoded))
        val_acc = accuracy_score(y_val, clf.predict(X_val_encoded))
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        val_accuracies.append(val_acc)
        
        print(f"Max Depth: {max_depth}, Train: {train_acc:.4f}, Test: {test_acc:.4f}, Val: {val_acc:.4f}")
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(max_depths, test_accuracies, marker='x', label='Testing Accuracy')
    plt.plot(max_depths, val_accuracies, marker='s', label='Validation Accuracy')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Maximum Depth (Scikit-learn)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/experiment_4i_accuracies.png')
    plt.show()
    
    # Find best max_depth based on validation accuracy
    best_max_depth = max_depths[np.argmax(val_accuracies)]
    print(f"Best max_depth: {best_max_depth}")
    
    # Part (ii): Vary ccp_alpha
    ccp_alphas = [0.001, 0.01, 0.1, 0.2]
    train_accuracies_ccp = []
    test_accuracies_ccp = []
    val_accuracies_ccp = []
    
    for ccp_alpha in ccp_alphas:
        # Train decision tree
        clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=ccp_alpha, random_state=42)
        clf.fit(X_train_encoded, y_train)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, clf.predict(X_train_encoded))
        test_acc = accuracy_score(y_test, clf.predict(X_test_encoded))
        val_acc = accuracy_score(y_val, clf.predict(X_val_encoded))
        
        train_accuracies_ccp.append(train_acc)
        test_accuracies_ccp.append(test_acc)
        val_accuracies_ccp.append(val_acc)
        
        print(f"CCP Alpha: {ccp_alpha}, Train: {train_acc:.4f}, Test: {test_acc:.4f}, Val: {val_acc:.4f}")
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, train_accuracies_ccp, marker='o', label='Training Accuracy')
    plt.plot(ccp_alphas, test_accuracies_ccp, marker='x', label='Testing Accuracy')
    plt.plot(ccp_alphas, val_accuracies_ccp, marker='s', label='Validation Accuracy')
    plt.xlabel('CCP Alpha')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. CCP Alpha (Scikit-learn)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/experiment_4ii_accuracies.png')
    plt.show()
    
    # Find best ccp_alpha based on validation accuracy
    best_ccp_alpha = ccp_alphas[np.argmax(val_accuracies_ccp)]
    print(f"Best ccp_alpha: {best_ccp_alpha}")
    
    # Train final model with best parameters
    best_model_depth = DecisionTreeClassifier(criterion='entropy', max_depth=best_max_depth, random_state=42)
    best_model_depth.fit(X_train_encoded, y_train)
    
    best_model_ccp = DecisionTreeClassifier(criterion='entropy', ccp_alpha=best_ccp_alpha, random_state=42)
    best_model_ccp.fit(X_train_encoded, y_train)
    
    # Calculate final accuracies
    depth_test_acc = accuracy_score(y_test, best_model_depth.predict(X_test_encoded))
    ccp_test_acc = accuracy_score(y_test, best_model_ccp.predict(X_test_encoded))
    
    print(f"Final Test Accuracy (best max_depth={best_max_depth}): {depth_test_acc:.4f}")
    print(f"Final Test Accuracy (best ccp_alpha={best_ccp_alpha}): {ccp_test_acc:.4f}")
    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if depth_test_acc > ccp_test_acc:
        pd.DataFrame({'prediction': best_model_depth.predict(X_test_encoded)}).to_csv(os.path.join(output_path, 'prediction_d.csv'), index=False)
    else:
        pd.DataFrame({'prediction': best_model_ccp.predict(X_test_encoded)}).to_csv(os.path.join(output_path, 'prediction_d.csv'), index=False)
    return best_max_depth, best_ccp_alpha, depth_test_acc, ccp_test_acc

def experiment_5(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test, output_path):
    """Experiment with Random Forests"""
    print("Starting Random Forest experiment...")
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': range(50, 350, 100),  # [50, 150, 250]
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'min_samples_split': range(2, 10, 2)  # [2, 4, 6, 8]
    }
    
    print("Performing grid search for Random Forest parameters...")
    # Create Random Forest classifier with out-of-bag score
    rf = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_encoded, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Train Random Forest with best parameters
    best_rf = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=42, **best_params)
    best_rf.fit(X_train_encoded, y_train)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, best_rf.predict(X_train_encoded))
    val_acc = accuracy_score(y_val, best_rf.predict(X_val_encoded))
    test_acc = accuracy_score(y_test, best_rf.predict(X_test_encoded))
    oob_acc = best_rf.oob_score_
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Out-of-Bag Accuracy: {oob_acc:.4f}")
    
    # Calculate class-specific accuracies for test set
    y_pred_test = best_rf.predict(X_test_encoded)
    
    # Accuracy for >50K predictions
    high_income_indices = y_test == '>50K'
    high_income_acc = np.sum(y_pred_test[high_income_indices] == '>50K') / np.sum(high_income_indices)
    
    # Accuracy for <=50K predictions
    low_income_indices = y_test == '<=50K'
    low_income_acc = np.sum(y_pred_test[low_income_indices] == '<=50K') / np.sum(low_income_indices)
    
    print(f"Accuracy for >50K: {high_income_acc:.4f}, Accuracy for <=50K: {low_income_acc:.4f}")
    
    # Save predictions to CSV
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pd.DataFrame({'prediction': y_pred_test}).to_csv(os.path.join(output_path, 'prediction_e.csv'), index=False)
    
    return {
        'best_params': best_params,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'oob_acc': oob_acc,
        'high_income_acc': high_income_acc,
        'low_income_acc': low_income_acc
    }



if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        train_path = sys.argv[1]
        val_path = sys.argv[2]
        test_path = sys.argv[3]
        output_path = sys.argv[4]
        question_part = sys.argv[5]
        
        if question_part == 'a':
            # Experiment 1
            train_data = load_data(train_path)
            test_data = load_data(test_path)
            X_train, y_train = preprocess_data(train_data)
            X_test, y_test = preprocess_data(test_data)
            types = identify_attribute_types(train_data)
            experiment_1(X_train, y_train, X_test, y_test, types, output_path)
        elif question_part == 'b':
            # Experiment 2
            train_data = load_data(train_path)
            test_data = load_data(test_path)
            X_train, y_train = preprocess_data(train_data)
            X_test, y_test = preprocess_data(test_data)
            X_train_encoded, _, X_test_encoded = one_hot_encode(X_train, X_train, X_test)  # Using train as val placeholder
            experiment_2(X_train_encoded, y_train, X_test_encoded, y_test, output_path)
        elif question_part == 'c':
            # Experiment 3
            train_data = load_data(train_path)
            val_data = load_data(val_path)
            test_data = load_data(test_path)
            X_train, y_train = preprocess_data(train_data)
            X_val, y_val = preprocess_data(val_data)
            X_test, y_test = preprocess_data(test_data)
            X_train_encoded, X_val_encoded, X_test_encoded = one_hot_encode(X_train, X_val, X_test)
            max_depths = [25, 35, 45, 55]
            experiment_3(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test, max_depths, output_path)
        elif question_part == 'd':
            # Experiment 4
            train_data = load_data(train_path)
            val_data = load_data(val_path)
            test_data = load_data(test_path)
            X_train, y_train = preprocess_data(train_data)
            X_val, y_val = preprocess_data(val_data)
            X_test, y_test = preprocess_data(test_data)
            X_train_encoded, X_val_encoded, X_test_encoded = one_hot_encode(X_train, X_val, X_test)
            experiment_4(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test, output_path)
        elif question_part == 'e':
            # Experiment 5
            train_data = load_data(train_path)
            val_data = load_data(val_path)
            test_data = load_data(test_path)
            X_train, y_train = preprocess_data(train_data)
            X_val, y_val = preprocess_data(val_data)
            X_test, y_test = preprocess_data(test_data)
            X_train_encoded, X_val_encoded, X_test_encoded = one_hot_encode(X_train, X_val, X_test)
            experiment_5(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test, output_path)
    # else:
    #     # Run all experiments
    #     main()

# def main():
#     """Main function to run all experiments"""
#     # Create output directory if it doesn't exist
#     if not os.path.exists('output'):
#         os.makedirs('output')
    
#     print("Loading data...")
#     # Load the data
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     train_data = load_data(os.path.join(script_dir, "data/train.csv"))
#     val_data = load_data(os.path.join(script_dir, "data/valid.csv"))
#     test_data = load_data(os.path.join(script_dir, "data/test.csv"))
    
#     # Preprocess the data
#     X_train, y_train = preprocess_data(train_data)
#     X_val, y_val = preprocess_data(val_data)
#     X_test, y_test = preprocess_data(test_data)
    
#     # Identify attribute types
#     types = identify_attribute_types(train_data)
#     print(f"Identified {types.count('cat')} categorical and {types.count('cont')} continuous features")
    
#     # Experiment 1: Decision Tree with different max depths
#     print("\n=== Experiment 1: Decision Tree with Different Max Depths ===")
#     exp1_results = experiment_1(X_train, y_train, X_test, y_test, types)
    
#     # Experiment 2: Decision Tree with One-Hot Encoding
#     print("\n=== Experiment 2: Decision Tree with One-Hot Encoding ===")
#     # Perform one-hot encoding
#     X_train_encoded, X_val_encoded, X_test_encoded = one_hot_encode(X_train, X_val, X_test)
#     exp2_results = experiment_2(X_train_encoded, y_train, X_test_encoded, y_test)
    
#     # Experiment 3: Post-Pruning
#     print("\n=== Experiment 3: Post-Pruning ===")
#     experiment_3(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test, exp2_results[2])
    
#     # Experiment 4: Scikit-learn Decision Tree
#     print("\n=== Experiment 4: Scikit-learn Decision Tree ===")
#     exp4_results = experiment_4(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test)
    
#     # Experiment 5: Random Forests
#     print("\n=== Experiment 5: Random Forests ===")
#     exp5_results = experiment_5(X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test)
    
#     # Compare results from all experiments
#     print("\n=== Comparison of Results ===")
#     print("Experiment 1 (Custom DT): Test Accuracy = {:.4f}".format(exp1_results[1][-1]))
#     print("Experiment 2 (Custom DT with One-Hot): Test Accuracy = {:.4f}".format(exp2_results[1][-1]))
#     print("Experiment 4 (Scikit-learn DT): Test Accuracy = {:.4f}".format(max(exp4_results[2], exp4_results[3])))
#     print("Experiment 5 (Random Forest): Test Accuracy = {:.4f}".format(exp5_results['test_acc']))
    
#     print("\nAll experiments completed successfully!")
