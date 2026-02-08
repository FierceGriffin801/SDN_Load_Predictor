import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        # Upgrade: 2 Layers for better complex pattern learning
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take last time step
        return out

class HybridEnsemble:
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        
    def fit_lstm(self, X_train, y_train, epochs=30):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        input_dim = X_train.shape[2]
        output_dim = len(np.unique(y_train))
        
        # Upgrade: 256 Hidden Units
        self.lstm_model = LSTMModel(input_dim, 256, output_dim, num_layers=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        print("Training LSTM Component (Deep Trend Learner)...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def fit_rf(self, X_train, y_train):
        print("Training Base Random Forest...")
        nsamples, nx, ny = X_train.shape
        X_flat = X_train.reshape((nsamples, nx*ny))
        self.rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_flat, y_train)

    def fit_meta(self, X_train, y_train):
        # We use Weighted Voting instead of Stacking to avoid Overfitting
        pass

    def predict(self, X_test):
        # 1. Get LSTM Probs
        X_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            lstm_probs = torch.softmax(self.lstm_model(X_tensor), dim=1).numpy()
            
        # 2. Get RF Probs
        nsamples, nx, ny = X_test.shape
        X_flat = X_test.reshape((nsamples, nx*ny))
        rf_probs = self.rf_model.predict_proba(X_flat)
        
        # 3. Weighted Soft Voting
        # Since we engineered the dataset to have strong trends, LSTM is valuable.
        # But RF is robust. 
        # Weights: 0.5/0.5 is safe.
        avg_probs = (0.5 * lstm_probs) + (0.5 * rf_probs)
        
        return np.argmax(avg_probs, axis=1)

if __name__ == "__main__":
    print("Hybrid Model Definition Loaded.")
