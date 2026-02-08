import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_data
from hybrid_model import HybridEnsemble
from sklearn.model_selection import train_test_split
import torch

# Constants for Simulation
LATENCY_LOW = 3.0       # ms (Balanced state)
LATENCY_MIGRATION = 10.0 # ms (Cost of moving flows)
LATENCY_PENALTY = 100.0  # ms (Cost of missing an overload = Packet Loss/Queueing)

def simulate_latency_reduction():
    print("--- Simulating Latency Reduction (Proactive vs Reactive) ---")
    
    # 1. Load Data (with new Latency column)
    # We need the RAW dataframe for latency values, not just the processed X, y
    df = pd.read_csv('sdn_traffic_data.csv')
    
    # Get the processed X for the model
    X_processed, y_processed, _ = load_and_preprocess_data()
    
    # Align Dataframes (Preprocessing cuts off the first 'sequence_length' rows)
    seq_len = 10
    df_test = df.iloc[seq_len:].reset_index(drop=True)
    
    # Split exactly as training did (random_state=42)
    # indices: train_idx, test_idx
    # We need to subset df_test to get the test set rows
    _, X_test, _, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
    _, df_test_split = train_test_split(df_test, test_size=0.2, random_state=42)
    
    # Ensure they match
    assert len(df_test_split) == len(X_test)
    
    # 2. Get Model Predictions (Proactive)
    print("Loading Hybrid Model...")
    model = HybridEnsemble()
    
    # Train quickly to get a working model (or load saved weights if we had them)
    # ideally we recount training, but strict separation is fine for simulation demo
    X_train, _, y_train, _ = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
    model.fit_lstm(X_train.astype(np.float32), y_train, epochs=20)
    model.fit_rf(X_train.astype(np.float32), y_train)
    
    print("Generating Predictions...")
    proactive_predictions = model.predict(X_test.astype(np.float32))
    
    # 3. Simulation Loop
    # We compare 3 strategies on the Test Set
    
    # Buffers to store latency over time
    lat_no_action = []
    lat_reactive = []
    lat_proactive = []
    
    ground_truth_labels = df_test_split['load_label'].values
    # ground_truth_latency = df_test_split['latency'].values # We don't use this directly, we SIMULATE the effect
    
    # State tracking
    reactive_cooldown = 0
    proactive_cooldown = 0
    
    for t in range(len(df_test_split)):
        true_state = ground_truth_labels[t] # 0, 1, 2
        pred_state = proactive_predictions[t]
        
        # --- A. No Action Strategy ---
        # If Overload (2), we suffer LATENCY_PENALTY
        # If Normal/Med, we get LATENCY_LOW (simplified)
        if true_state == 2:
            l_base = LATENCY_PENALTY
        elif true_state == 1:
             l_base = LATENCY_LOW * 2 # Medium load
        else:
            l_base = LATENCY_LOW
        lat_no_action.append(l_base)
        
        # --- B. Reactive Strategy (Threshold based) ---
        # Action only if we SEE overload (True State = 2)
        # If we see overload, we migrate. Migration takes 1 step.
        # Step t: We see overload. We start migration. We suffer Penalty (too late).
        # Step t+1: Migration finishes. We enjoy Low Latency.
        
        curr_lat_reactive = l_base # Default to base behavior
        
        if reactive_cooldown > 0:
            # We are migrating or recently migrated -> Low Latency
            curr_lat_reactive = LATENCY_LOW 
            reactive_cooldown -= 1
        elif true_state == 2:
            # Detection! Trigger migration for NEXT steps.
            # BUT for THIS step, we are already overloaded.
            curr_lat_reactive = LATENCY_PENALTY 
            reactive_cooldown = 2 # Protect for next 2 steps
        
        lat_reactive.append(curr_lat_reactive)
            
        # --- C. Proactive Strategy (Hybrid Model) ---
        # Action if PREDICTION == 2
        # Step t: Model says "Overload coming". We migrate NOW.
        # If True State was 2: We successfully avoided penalty! Latency = Migration Cost.
        # If True State was 0: We migrated for nothing. Latency = Migration Cost (Small penalty).
        
        curr_lat_proactive = l_base
        
        if proactive_cooldown > 0:
            curr_lat_proactive = LATENCY_LOW
            proactive_cooldown -= 1
        elif pred_state == 2:
            # PROACTIVE TRIGGER
            # We migrate immediately. 
            # Cost = Migration Overhead (Small) vs Penalty (Huge)
            curr_lat_proactive = LATENCY_MIGRATION
            proactive_cooldown = 2 # Protected
            
        # Corner case: Proactive missed it (False Negative)
        # If pred != 2 but true == 2. We suffer penalty.
        if pred_state != 2 and true_state == 2 and proactive_cooldown == 0:
            curr_lat_proactive = LATENCY_PENALTY
            
        lat_proactive.append(curr_lat_proactive)

    # 4. Results & Plotting
    avg_no = np.mean(lat_no_action)
    avg_react = np.mean(lat_reactive)
    avg_pro = np.mean(lat_proactive)
    
    print(f"\n--- Simulation Results (Average Latency per Time Step) ---")
    print(f"1. No Load Balancing: {avg_no:.2f} ms")
    print(f"2. Reactive (Threshold): {avg_react:.2f} ms")
    print(f"3. Proactive (Hybrid):   {avg_pro:.2f} ms")
    
    reduction = ((avg_react - avg_pro) / avg_react) * 100
    print(f"Latency Reduction (Proactive vs Reactive): {reduction:.2f}%")
    
    # Plot Cumulative Latency (Total Network Delay experienced)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(lat_no_action), label='No Load Balancing', linestyle=':', color='gray')
    plt.plot(np.cumsum(lat_reactive), label='Reactive (Standard)', linestyle='--', color='orange')
    plt.plot(np.cumsum(lat_proactive), label='Proactive (Hybrid)', linewidth=2, color='green')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Network Latency (ms)')
    plt.title('Latency Accumulation: Proactive vs Reactive')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('latency_reduction_simulation.png', dpi=300)
    print("Saved 'latency_reduction_simulation.png'")

if __name__ == "__main__":
    simulate_latency_reduction()
