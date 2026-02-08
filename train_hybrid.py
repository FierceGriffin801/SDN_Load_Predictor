from hybrid_model import HybridEnsemble
from preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def train_and_evaluate_hybrid():
    print("--- Starting Hybrid Model Training ---")
    
    # Load Data
    X, y, scaler = load_and_preprocess_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Model
    model = HybridEnsemble()
    
    # Train Components
    print("\n1. Training LSTM Feature Extractor...")
    # Reshape for LSTM: (Samples, Time, Feat) -> PyTorch handles this
    # But need to make sure data is float32
    X_train = x_train_float = X_train.astype(np.float32)
    y_train = y_train_long = y_train.astype(np.int64)
    model.fit_lstm(X_train, y_train, epochs=100) # Deep training
    
    print("\n2. Training Base Random Forest...")
    model.fit_rf(X_train, y_train)
    
    # print("\n3. Training Meta-Learner (XGBoost)...")
    # model.fit_meta(X_train, y_train)
    
    # Evaluate
    print("\n--- Evaluating Hybrid Model ---")
    X_test = X_test.astype(np.float32)
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error, r2_score
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    
    return {"Model": "Hybrid Ensemble (Proposed)", "Accuracy": acc, "MAE": mae, "R2": r2}

if __name__ == "__main__":
    train_and_evaluate_hybrid()
