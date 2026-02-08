import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessing import load_and_preprocess_data
from hybrid_model import HybridEnsemble
import torch

# Set style for professional paper quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Normal', 'Medium', 'Overload'],
                yticklabels=['Normal', 'Medium', 'Overload'])
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

def plot_roc_curve(y_true, y_probs, model_name, ax):
    n_classes = 3
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    colors = ['blue', 'green', 'red']
    labels = ['Normal', 'Medium', 'Overload']
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} ROC Curve')
    ax.legend(loc="lower right")

def generate_paper_graphs():
    print("Generating Graphs for Research Paper...")
    
    # 1. Load Data
    X_seq, y, scaler = load_and_preprocess_data()
    
    # Flatten for RF
    nsamples, nx, ny = X_seq.shape
    X_flat = X_seq.reshape((nsamples, nx*ny))
    
    # Split
    X_train_seq, X_test_seq, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    X_train_flat, X_test_flat, _, _ = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # 2. Train Models
    # ---------------------------------------------------------
    
    # A. Random Forest (Baseline)
    print("Training Random Forest (Baseline)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train)
    rf_preds = rf.predict(X_test_flat)
    rf_probs = rf.predict_proba(X_test_flat)
    
    # B. Hybrid Model (Proposed)
    print("Training Hybrid Ensemble (Proposed)...")
    hybrid = HybridEnsemble()
    
    # Train LSTM
    X_train_seq_float = X_train_seq.astype(np.float32)
    hybrid.fit_lstm(X_train_seq_float, y_train, epochs=50) # Reduced epochs for plotting speed
    
    # Train RF component
    hybrid.fit_rf(X_train_seq_float, y_train)
    
    # Predict
    X_test_seq_float = X_test_seq.astype(np.float32)
    
    # We need Probabilities for ROC, so we manually get them from Hybrid
    # (Replicating predict logic from hybrid_model.py but returning probs)
    # 1. LSTM Probs
    X_tensor = torch.FloatTensor(X_test_seq_float)
    with torch.no_grad():
        lstm_probs = torch.softmax(hybrid.lstm_model(X_tensor), dim=1).numpy()
    
    # 2. RF Probs
    nsamples, nx, ny = X_test_seq.shape
    X_flat_test = X_test_seq.reshape((nsamples, nx*ny))
    _rf_probs = hybrid.rf_model.predict_proba(X_flat_test)
    
    # 3. Soft Voting
    hybrid_probs = (0.5 * lstm_probs) + (0.5 * _rf_probs)
    hybrid_preds = np.argmax(hybrid_probs, axis=1)

    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    
    # Figure 1: Confusion Matrices (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_confusion_matrix(y_test, rf_preds, "Random Forest", axes[0])
    plot_confusion_matrix(y_test, hybrid_preds, "Hybrid Ensemble", axes[1])
    plt.tight_layout()
    plt.savefig('graph_confusion_matrices.png', dpi=300)
    print("Saved 'graph_confusion_matrices.png'")
    
    # Figure 2: ROC Curves (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_roc_curve(y_test, rf_probs, "Random Forest", axes[0])
    plot_roc_curve(y_test, hybrid_probs, "Hybrid Ensemble", axes[1])
    plt.tight_layout()
    plt.savefig('graph_roc_curves.png', dpi=300)
    print("Saved 'graph_roc_curves.png'")
    
    # Figure 3: Time Series Snapshot (The "Money Shot")
    # Show last 100 points
    window = 100
    start_idx = 0 
    end_idx = start_idx + window
    
    plt.figure(figsize=(15, 6))
    plt.plot(range(window), y_test[start_idx:end_idx], 'k-', linewidth=2, label='True Load', alpha=0.6)
    plt.plot(range(window), rf_preds[start_idx:end_idx], 'r--', label='RF Prediction', alpha=0.8)
    plt.plot(range(window), hybrid_preds[start_idx:end_idx], 'g-', linewidth=2, label='Hybrid Prediction', alpha=0.9)
    
    # Highlight disagreements?
    
    plt.yticks([0, 1, 2], ['Normal', 'Medium', 'Overload'])
    plt.xlabel('Time Steps')
    plt.ylabel('Load State')
    plt.title('Real-time Prediction Comparison (Snapshot)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('graph_time_series.png', dpi=300)
    print("Saved 'graph_time_series.png'")

    print("All graphs generated successfully.")

if __name__ == "__main__":
    generate_paper_graphs()
