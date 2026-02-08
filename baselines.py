from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from preprocessing import load_and_preprocess_data
import numpy as np

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    return {"Model": name, "Accuracy": acc, "MAE": mae, "R2": r2}

def train_baselines():
    print("Training Baseline Models...")
    
    # Load Data
    X_seq, y, scaler = load_and_preprocess_data()
    
    # Flatten: (Samples, Time, Feat) -> (Samples, Time*Feat)
    nsamples, nx, ny = X_seq.shape
    X_flat = X_seq.reshape((nsamples, nx*ny))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    results = []

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    results.append(evaluate("Logistic Regression", y_test, y_pred))
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    results.append(evaluate("Decision Tree", y_test, y_pred))

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, y_pred))

    # 4. SVM
    svm = SVC(kernel='rbf', probability=True) 
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results.append(evaluate("SVM", y_test, y_pred))

    # 5. MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    results.append(evaluate("MLP (Neural Net)", y_test, y_pred))

    # 6. RL (DQN)
    from rl_model import DQNClassifier
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    dqn = DQNClassifier(input_dim, output_dim, lr=0.001)
    dqn.fit(X_train, y_train, epochs=10) # 10 Epochs (optimized)
    y_pred = dqn.predict(X_test)
    results.append(evaluate("DQN (RL Baseline)", y_test, y_pred))

    return results

if __name__ == "__main__":
    train_baselines()
