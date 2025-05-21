import os
import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix, solvers
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import cross_val_score

class SupportVectorMachine:
    '''
    Binary Classifier solved via CVXOPT.
    Supports two kernels: linear and gaussian.
    For linear kernel, computes weight vector w.
    For gaussian kernel (K(x,z)=exp(-gamma*||x-z||^2)), w is not explicitly formed.
    '''
    def __init__(self):
        self.w = None
        self.b = 0
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.sv_idx = None
        self.kernel = None
        self.gamma = None  # used for gaussian kernel
        
    def fit(self, X, y, kernel='linear', C=1.0, gamma=0.001):
        '''
        Train the SVM using the dual formulation.
        For labels, converts 0 -> -1 and 1 -> +1.
        If kernel=='linear', uses K = X X^T.
        If kernel=='gaussian', computes K[i,j] = exp(-gamma * ||x_i - x_j||^2).
        '''
        n_samples = X.shape[0]
        self.kernel = kernel
        self.gamma = gamma
        
        # Convert labels: 0-> -1, 1 -> +1
        y = y.astype(np.float64)
        y = np.where(y == 0, -1, 1)
        
        # Compute kernel matrix
        if kernel == 'linear':
            K = np.dot(X, X.T)
        elif kernel == 'gaussian':
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    diff = X[i] - X[j]
                    K[i, j] = np.exp(-gamma * np.dot(diff, diff))
        else:
            raise ValueError("Unknown kernel")
            
        # Set up quadratic program: P, q, G, h, A, b as required by CVXOPT.
        P = matrix(np.outer(y, y) * K, tc='d')
        q = matrix(-np.ones(n_samples), tc='d')
        # A must be 1 x n_samples double matrix.
        A = matrix(y.reshape(1, -1), tc='d')
        b_mat = matrix(0.0, (1, 1), tc='d')
        
        # Inequality constraints: 0 <= alpha_i <= C.
        G_std = -np.eye(n_samples)
        h_std = np.zeros(n_samples)
        G_slack = np.eye(n_samples)
        h_slack = np.ones(n_samples) * C
        G = matrix(np.vstack((G_std, G_slack)), tc='d')
        h = matrix(np.hstack((h_std, h_slack)), tc='d')
        
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b_mat)
        alphas_all = np.ravel(solution['x'])
        
        # Select support vectors (alpha > 1e-5)
        sv = alphas_all > 1e-5
        self.sv_idx = np.where(sv)[0]
        self.alphas = alphas_all[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        
        if kernel == 'linear':
            # Compute weight vector: w = sum(alpha_i * y_i * x_i)
            self.w = np.sum((self.alphas * self.support_vector_labels)[:, np.newaxis] * self.support_vectors, axis=0)
            # Compute bias as average: b = y_i - w^T x_i over support vectors.
            self.b = np.mean(self.support_vector_labels - np.dot(self.support_vectors, self.w))
        elif kernel == 'gaussian':
            # Cannot form w explicitly.
            # Compute bias b = average( y_i - sum_j(alpha_j*y_j*K(x_j, x_i)) ) for support vectors.
            b_sum = 0
            for i in range(len(self.alphas)):
                s = 0
                for alpha, label, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                    diff = self.support_vectors[i] - sv
                    s += alpha * label * np.exp(-gamma * np.dot(diff, diff))
                b_sum += (self.support_vector_labels[i] - s)
            self.b = b_sum / len(self.alphas)
        
        # Reporting support vector information.
        n_sv = len(self.alphas)
        print(f"[{kernel.upper()}] Number of support vectors: {n_sv}")
        print(f"[{kernel.upper()}] Percentage of training samples as support vectors: {n_sv / n_samples * 100:.2f}%")
        
    def project(self, X):
        '''
        Compute decision function f(x)= sum(alpha_i*y_i*K(x_i,x)) + b.
        For linear kernel K(x,z)=x^T z.
        For gaussian kernel, K(x,z)= exp(-gamma*||x-z||^2).
        '''
        if self.kernel == 'linear':
            return np.dot(X, self.w) + self.b
        elif self.kernel == 'gaussian':
            y_predict = []
            for i in range(X.shape[0]):
                s = 0
                for alpha, label, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                    diff = X[i] - sv
                    s += alpha * label * np.exp(-self.gamma * np.dot(diff, diff))
                y_predict.append(s)
            return np.array(y_predict) + self.b
        else:
            raise ValueError("Unknown kernel")
            
    def predict(self, X):
        '''
        Predict class labels: 1 if decision function >= 0, else 0.
        '''
        return np.where(self.project(X) >= 0, 1, 0)

# ------- Image Preprocessing and Data Loading -------
def preprocess_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            width, height = img.size
            # If dimensions are at least 100, center crop; otherwise, resize.
            if width >= 100 and height >= 100:
                left = (width - 100) // 2
                top = (height - 100) // 2
                right = left + 100
                bottom = top + 100
                img = img.crop((left, top, right, bottom))
            else:
                img = img.resize((100, 100))
            img_array = np.asarray(img, dtype=np.float32) / 255.0
            return img_array.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_data(data_dir, selected_classes):
    X, y = [], []
    for label, class_name in enumerate(selected_classes):
        class_dir = os.path.join(data_dir, class_name)
        image_paths = glob.glob(os.path.join(class_dir, '*.jpg'))
        for path in image_paths:
            x = preprocess_image(path)
            if x is not None:
                X.append(x)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_binary_svm_for_multiclass(X, y, class1, class2, C=1.0, gamma=0.001):
    idx = np.where((y == class1) | (y == class2))[0]
    X_pair = X[idx]
    new_y = (y[idx] == class2).astype(np.int32)  # class1 -> 0, class2 -> 1
    model = SupportVectorMachine()
    model.fit(X_pair, new_y, kernel='gaussian', C=C, gamma=gamma)
    return model

def multi_class_svm_predict(X, classifiers, num_classes):
    n = X.shape[0]
    votes = np.zeros((n, num_classes), dtype=int)
    scores = np.zeros((n, num_classes), dtype=float)
    for (i, j), model in classifiers.items():
        f = model.project(X)  # decision function values for X on classifier (i,j)
        for idx in range(n):
            if f[idx] >= 0:
                votes[idx, j] += 1
                scores[idx, j] += f[idx]
            else:
                votes[idx, i] += 1
                scores[idx, i] += -f[idx]
    pred = np.zeros(n, dtype=int)
    for idx in range(n):
        max_votes = np.max(votes[idx])
        candidates = np.where(votes[idx] == max_votes)[0]
        if len(candidates) == 1:
            pred[idx] = candidates[0]
        else:
            pred[idx] = candidates[np.argmax(scores[idx, candidates])]
    return pred

def multi_class_svm_experiment(X_train, y_train, X_test, y_test, C=1.0, gamma=0.001):
    num_classes = len(np.unique(y_train))
    classifiers = {}
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            model = train_binary_svm_for_multiclass(X_train, y_train, i, j, C, gamma)
            classifiers[(i, j)] = model
            print(f"Trained classifier for classes {i} vs {j}")
    y_pred = multi_class_svm_predict(X_test, classifiers, num_classes)
    acc = np.mean(y_pred == y_test) * 100
    print(f"\nMulti-class SVM Test set accuracy: {acc:.2f}%")


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# ----- Visualize Misclassified Examples -----
def visualize_misclassifications(X, y_true, y_pred, classes, title_prefix, num_examples=10):
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        print(f"No misclassifications for {title_prefix}")
        return
    np.random.shuffle(mis_idx)
    mis_idx = mis_idx[:num_examples]
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(mis_idx):
        img = X[idx].reshape((100, 100, 3))
        plt.subplot(2, int(np.ceil(num_examples/2)), i+1)
        plt.imshow(img)
        plt.title(f"True: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}")
        plt.axis('off')
    plt.suptitle(f"{title_prefix} - Misclassified Examples")
    plt.show()

if __name__ == '__main__':
    train_dir = "../data/Q2/train"
    test_dir = "../data/Q2/test"

    # ---------------- Binary Classification Experiments ----------------
    # For binary classification I used two classes based on entry number "03" (i.e. indices 3 and 4)
    all_train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if len(all_train_classes) == 0:
        print("No class subdirectories found in train directory.")
        exit(1)
    d = 3  # corresponds to entry "03"
    num_classes_binary = 11
    if len(all_train_classes) < num_classes_binary:
        print(f"Expected at least {num_classes_binary} classes, but found {len(all_train_classes)}")
        exit(1)
    selected_binary_classes = [all_train_classes[d], all_train_classes[(d+1)%num_classes_binary]]
    print(f"Selected classes for binary classification: {selected_binary_classes}")
    
    print("Loading binary training data...")
    X_train, y_train = load_data(train_dir, selected_binary_classes)
    print(f"Binary Training samples: {X_train.shape[0]}")
    print("Loading binary test data...")
    X_test, y_test = load_data(test_dir, selected_binary_classes)
    print(f"Binary Test samples: {X_test.shape[0]}")
    
    # CVXOPT SVM with Linear Kernel (binary)
    print("\n--- CVXOPT SVM with Linear Kernel (Binary) ---")
    svm_linear = SupportVectorMachine()
    svm_linear.fit(X_train, y_train, kernel='linear', C=1.0)
    y_pred_linear = svm_linear.predict(X_test)
    accuracy_linear = np.mean(y_pred_linear == y_test) * 100
    print(f"Linear SVM Test set accuracy: {accuracy_linear:.2f}%")
    print(f"\nLinear SVM Weight vector (w) norm: {np.linalg.norm(svm_linear.w):.4f}")
    print(f"Linear SVM Bias term (b): {svm_linear.b:.4f}")
    # Visualization for top-5 support vectors (linear)
    top5_idx_linear = np.argsort(-svm_linear.alphas)[:5]
    top5_sv_linear = svm_linear.support_vectors[top5_idx_linear]
    plt.figure(figsize=(12, 3))
    for i, sv in enumerate(top5_sv_linear):
        img = sv.reshape((100, 100, 3))
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"LSV {i+1}")
        plt.axis('off')
    plt.suptitle("Top-5 Linear Support Vectors")
    plt.show()
    # plotting weight vector as image
    w_normalized = (svm_linear.w - np.min(svm_linear.w)) / (np.max(svm_linear.w) - np.min(svm_linear.w))
    w_img = w_normalized.reshape((100, 100, 3))

    plt.figure()
    plt.imshow(w_img)
    plt.title("Weight Vector as Image")
    plt.axis('off')
    plt.show()

    # CVXOPT SVM with Gaussian Kernel (binary)
    print("\n--- CVXOPT SVM with Gaussian Kernel (Binary) ---")
    svm_gauss = SupportVectorMachine()
    svm_gauss.fit(X_train, y_train, kernel='gaussian', C=1.0, gamma=0.001)
    y_pred_gauss = svm_gauss.predict(X_test)
    accuracy_gauss = np.mean(y_pred_gauss == y_test) * 100
    print(f"Gaussian SVM Test set accuracy: {accuracy_gauss:.2f}%")
    common_sv = np.intersect1d(svm_linear.sv_idx, svm_gauss.sv_idx)
    print(f"Number of common support vectors between linear and gaussian: {len(common_sv)}")
    top5_idx_gauss = np.argsort(-svm_gauss.alphas)[:5]
    top5_sv_gauss = svm_gauss.support_vectors[top5_idx_gauss]
    plt.figure(figsize=(12, 3))
    for i, sv in enumerate(top5_sv_gauss):
        img = sv.reshape((100, 100, 3))
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"GSV {i+1}")
        plt.axis('off')
    plt.suptitle("Top-5 Gaussian Support Vectors")
    plt.show()
    
    # scikit-learn SVM experiments (binary)
    print("\n--- scikit-learn SVM with Linear Kernel (Binary) ---")
    t_start = time.time()
    sklearn_linear = SVC(kernel='linear', C=1.0)
    sklearn_linear.fit(X_train, y_train)
    t_linear = time.time() - t_start
    y_pred_sklearn_linear = sklearn_linear.predict(X_test)
    accuracy_sklearn_linear = np.mean(y_pred_sklearn_linear == y_test) * 100
    n_sv_sklearn_linear = len(sklearn_linear.support_)
    print(f"scikit-learn Linear SVM Test set accuracy: {accuracy_sklearn_linear:.2f}%")
    print(f"scikit-learn Linear SVM number of support vectors: {n_sv_sklearn_linear}")
    print(f"scikit-learn Linear SVM training time: {t_linear:.4f} sec")
    print("\nComparison (Linear Kernel):")
    print("CVXOPT w norm:", np.linalg.norm(svm_linear.w))
    print("scikit-learn w norm:", np.linalg.norm(sklearn_linear.coef_[0]))
    print("CVXOPT bias:", svm_linear.b)
    print("scikit-learn bias:", sklearn_linear.intercept_[0])
    common_linear_count = 0
    for sv in sklearn_linear.support_vectors_:
        diff = np.linalg.norm(svm_linear.support_vectors - sv, axis=1)
        if np.any(diff < 1e-5):
            common_linear_count += 1
    print(f"Common support vectors between CVXOPT linear and scikit-learn linear: {common_linear_count}")
    
    print("\n--- scikit-learn SVM with Gaussian Kernel (Binary) ---")
    t_start = time.time()
    sklearn_gauss = SVC(kernel='rbf', C=1.0, gamma=0.001)
    sklearn_gauss.fit(X_train, y_train)
    t_gauss = time.time() - t_start
    y_pred_sklearn_gauss = sklearn_gauss.predict(X_test)
    accuracy_sklearn_gauss = np.mean(y_pred_sklearn_gauss == y_test) * 100
    n_sv_sklearn_gauss = len(sklearn_gauss.support_)
    print(f"scikit-learn Gaussian SVM Test set accuracy: {accuracy_sklearn_gauss:.2f}%")
    print(f"scikit-learn Gaussian SVM number of support vectors: {n_sv_sklearn_gauss}")
    print(f"scikit-learn Gaussian SVM training time: {t_gauss:.4f} sec")
    common_gauss_count = 0
    for sv in sklearn_gauss.support_vectors_:
        diff = np.linalg.norm(svm_gauss.support_vectors - sv, axis=1)
        if np.any(diff < 1e-5):
            common_gauss_count += 1
    print(f"Common support vectors between CVXOPT gaussian and scikit-learn gaussian: {common_gauss_count}")
    
    print("\n--- Computational Cost Comparison ---")
    print(f"scikit-learn Linear SVM training time: {t_linear:.4f} sec")
    print(f"scikit-learn Gaussian SVM training time: {t_gauss:.4f} sec")
    
    print("\n--- scikit-learn SGD SVM (Hinge Loss) ---")
    t_start = time.time()
    sgd_svm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
    sgd_svm.fit(X_train, y_train)
    t_sgd = time.time() - t_start
    y_pred_sgd = sgd_svm.predict(X_test)
    accuracy_sgd = np.mean(y_pred_sgd == y_test) * 100
    print(f"scikit-learn SGD SVM Test set accuracy: {accuracy_sgd:.2f}%")
    print(f"scikit-learn SGD SVM training time: {t_sgd:.4f} sec")
    print("\nComparison with LIBLINEAR (scikit-learn SVC with Linear Kernel):")
    print(f"LIBLINEAR Linear SVM Test set accuracy: {accuracy_sklearn_linear:.2f}%")
    print(f"LIBLINEAR training time: {t_linear:.4f} sec")
    
    # ---------------- Multi-Class Classification Experiment ----------------
    print("\n--- Multi-Class SVM Experiment using One-vs-One (CVXOPT Gaussian) ---")
    # For multi-class, use all available classes
    multi_train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    X_train_multi, y_train_multi = load_data(train_dir, multi_train_classes)
    X_test_multi, y_test_multi = load_data(test_dir, multi_train_classes)
    print(f"Multi-class Training samples: {X_train_multi.shape[0]} (for {len(multi_train_classes)} classes)")
    print(f"Multi-class Test samples: {X_test_multi.shape[0]}")
    multi_class_svm_experiment(X_train_multi, y_train_multi, X_test_multi, y_test_multi, C=1.0, gamma=0.001)


    # ------------------ scikit-learn Multi-Class SVM Experiment ------------------
    print("\n--- scikit-learn Multi-Class SVM with Gaussian Kernel ---")
    # Use all available classes
    multi_train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    X_train_multi, y_train_multi = load_data(train_dir, multi_train_classes)
    X_test_multi, y_test_multi = load_data(test_dir, multi_train_classes)
    print(f"Multi-class Training samples: {X_train_multi.shape[0]} (for {len(multi_train_classes)} classes)")
    print(f"Multi-class Test samples: {X_test_multi.shape[0]}")

    t_start = time.time()
    sklearn_multi = SVC(kernel='rbf', C=1.0, gamma=0.001)
    sklearn_multi.fit(X_train_multi, y_train_multi)
    t_multi = time.time() - t_start

    y_pred_sklearn_multi = sklearn_multi.predict(X_test_multi)
    accuracy_sklearn_multi = np.mean(y_pred_sklearn_multi == y_test_multi) * 100
    print(f"scikit-learn Multi-class SVM Test set accuracy: {accuracy_sklearn_multi:.2f}%")
    print(f"scikit-learn Multi-class SVM training time: {t_multi:.4f} sec")

    # ----- Run CVXOPT One-vs-One Multi-Class SVM Experiment and get predictions -----
    print("\n--- CVXOPT Multi-class SVM (One-vs-One, Gaussian) for Confusion Matrix -----")
    # Use all available classes (assumed 11)
    multi_train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    X_train_multi, y_train_multi = load_data(train_dir, multi_train_classes)
    X_test_multi, y_test_multi = load_data(test_dir, multi_train_classes)
    num_classes = len(np.unique(y_train_multi))
    cvxopt_classifiers = {}
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            model = train_binary_svm_for_multiclass(X_train_multi, y_train_multi, i, j, C=1.0, gamma=0.001)
            cvxopt_classifiers[(i, j)] = model
            print(f"Trained classifier for classes {i} vs {j}")
    y_pred_cvxopt = multi_class_svm_predict(X_test_multi, cvxopt_classifiers, num_classes)
    cm_cvxopt = confusion_matrix(y_test_multi, y_pred_cvxopt)
    plot_confusion_matrix(cm_cvxopt, classes=multi_train_classes, title='CVXOPT Multi-Class Confusion Matrix')
    plt.show()

    # ----- scikit-learn Multi-Class SVM (Gaussian) Confusion Matrix -----
    print("\n--- scikit-learn Multi-class SVM with Gaussian Kernel for Confusion Matrix -----")
    t_start = time.time()
    sklearn_multi = SVC(kernel='rbf', C=1.0, gamma=0.001)
    sklearn_multi.fit(X_train_multi, y_train_multi)
    t_multi = time.time() - t_start
    y_pred_sklearn_multi = sklearn_multi.predict(X_test_multi)
    cm_sklearn = confusion_matrix(y_test_multi, y_pred_sklearn_multi)
    plot_confusion_matrix(cm_sklearn, classes=multi_train_classes, title='scikit-learn Multi-Class Confusion Matrix')
    plt.show()

    print("\n--- Misclassified Examples (CVXOPT Multi-Class) ---")
    visualize_misclassifications(X_test_multi, y_test_multi, y_pred_cvxopt, multi_train_classes, "CVXOPT")

    print("\n--- Misclassified Examples (scikit-learn Multi-Class) ---")
    visualize_misclassifications(X_test_multi, y_test_multi, y_pred_sklearn_multi, multi_train_classes, "scikit-learn")

    # exoeriment 8

    # ---- Hyperparameter tuning: 5-Fold CV for SVM with Gaussian kernel ----
    print("\n--- 5-Fold Cross-Validation for Hyperparameter Tuning (Gaussian SVM) ---")
    # Use the multi-class training and test data (all available classes)
    multi_train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    X_train_multi, y_train_multi = load_data(train_dir, multi_train_classes)
    X_test_multi, y_test_multi = load_data(test_dir, multi_train_classes)
    print(f"Multi-class Training samples: {X_train_multi.shape[0]} (for {len(multi_train_classes)} classes)")
    print(f"Multi-class Test samples: {X_test_multi.shape[0]}")

    param_C = [1e-5, 1e-3, 1, 5, 10]
    cv_scores = []
    test_scores = []

    for C_val in param_C:
        clf = SVC(kernel='rbf', C=C_val, gamma=0.001)
        # Perform 5-fold cross-validation on training set:
        scores = cross_val_score(clf, X_train_multi, y_train_multi, cv=5)
        cv_acc = np.mean(scores) * 100
        cv_scores.append(cv_acc)
        # Train on the full training set and evaluate test accuracy:
        clf.fit(X_train_multi, y_train_multi)
        test_acc = clf.score(X_test_multi, y_test_multi) * 100
        test_scores.append(test_acc)
        print(f"C: {C_val}, 5-fold CV Accuracy: {cv_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    # Plotting the 5-fold CV accuracy and test accuracy vs C
    plt.figure(figsize=(8,6))
    plt.plot(param_C, cv_scores, marker='o', label='5-Fold CV Accuracy')
    plt.plot(param_C, test_scores, marker='s', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel("C (log scale)")
    plt.ylabel("Accuracy (%)")
    plt.title("5-Fold CV and Test Accuracy vs C (γ = 0.001)")
    plt.legend()
    plt.grid(True)
    plt.show()

    best_index = np.argmax(cv_scores)
    best_C = param_C[best_index]
    print(f"Best C value based on 5-fold CV: {best_C} with CV accuracy: {cv_scores[best_index]:.2f}%")

    # Train an SVM using the best hyper-parameter C on the entire training set
    best_clf = SVC(kernel='rbf', C=best_C, gamma=0.001)
    best_clf.fit(X_train_multi, y_train_multi)
    final_test_acc = best_clf.score(X_test_multi, y_test_multi) * 100
    print(f"Final Test Accuracy with SVM (C = {best_C}): {final_test_acc:.2f}%")
    print("--End of Experiment for SVM--")