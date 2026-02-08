import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

def generate_sdn_traffic_data(num_samples=10000, output_file='sdn_traffic_data.csv'):
    """
    Generates synthetic SDN controller traffic data.
    
    Features:
    - timestamp: Time of the record
    - packet_in_count: Number of Packet-In messages (load indicator)
    - flow_table_usage: Percentage of flow table utilized (0-100)
    - active_switches: Number of switches connected
    - cpu_utilization: Controller CPU load (0-100)
    - memory_utilization: Controller Memory usage (0-100)
    - protocol_type: Dominant protocol (TCP/UDP/ICMP) - encoded as 0, 1, 2
    - label: 0 (Normal), 1 (Medium Load), 2 (Overload) - Target variable
    """
    print(f"Generating {num_samples} samples of synthetic SDN traffic data...")
    
    # Base timestamp
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i*5) for i in range(num_samples)] # Data every 5 seconds

    # Generate trends using sine waves + noise to simulate daily traffic cycles
    t = np.linspace(0, 4 * np.pi, num_samples)
    base_load = 50 + 30 * np.sin(t) # Cyclic load pattern
    
    # Add randomness (bursty traffic)
    noise = np.random.normal(0, 10, num_samples)
    packet_in_count = np.abs(base_load * 20 + noise * 5 + np.random.randint(0, 200, num_samples)).astype(int)
    
    # Introduce temporal dependencies (Trend-based bursts)
    # If traffic rises for 3 steps, spike at t+1
    for i in range(3, num_samples - 1):
        if packet_in_count[i] > packet_in_count[i-1] > packet_in_count[i-2]:
            # Positive trend -> High probability of future spike (Massive 3x Burst)
            packet_in_count[i+1] = int(packet_in_count[i+1] * 3.0)
            
    # Calculate features
    flow_table_usage = np.clip(packet_in_count / 20 + np.random.normal(0, 5, num_samples), 0, 100)
    memory_utilization = np.clip(packet_in_count / 25 + np.random.normal(0, 2, num_samples), 10, 95)
    
    # Independent features
    active_switches = np.random.randint(5, 50, num_samples)
    protocol_type = np.random.choice([0, 1, 2], num_samples, p=[0.6, 0.3, 0.1]) # Mostly TCP
    
    # Update correlated features after modifying packet_in
    cpu_utilization = np.clip(packet_in_count / 15 + np.random.normal(0, 5, num_samples), 5, 100)
    
    # Label Logic: PROACTIVE PREDICTION
    # We want to predict if ONE OF THE NEXT 2 STEPS will be overloaded
    # This forces the model to look at trends (LSTM) rather than just current state (RF)
    labels = []
    for i in range(num_samples):
        # Look ahead window (next 2 steps)
        # If any future step is overloaded, label = 2
        future_overload = False
        future_medium = False
        
        for lookahead in [1, 2]:
            if i + lookahead < num_samples:
                if cpu_utilization[i+lookahead] > 80:
                    future_overload = True
                elif cpu_utilization[i+lookahead] > 50:
                    future_medium = True
        
        if future_overload:
            labels.append(2)
        elif future_medium:
            labels.append(1)
        else:
            labels.append(0)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'packet_in_count': packet_in_count,
        'flow_table_usage': flow_table_usage,
        'active_switches': active_switches,
        'cpu_utilization': cpu_utilization,
        'memory_utilization': memory_utilization,
        'protocol_type': protocol_type,
        'load_label': labels
    })

    # Introduce some anomalies/spikes
    spike_indices = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    data.loc[spike_indices, 'packet_in_count'] *= 2.5
    data.loc[spike_indices, 'cpu_utilization'] = 99.9
    data.loc[spike_indices, 'load_label'] = 2

    # -------------------------------------------------------------------------
    # Generate Synthetic Latency (The metric we want to reduce)
    # -------------------------------------------------------------------------
    # Base latency: 2-5ms (normal network jutter)
    latency = np.random.normal(3, 1, num_samples)
    
    # 1. Linear correlation with Traffic
    latency += packet_in_count / 50  # More packets = signal propagation delay
    
    # 2. Non-linear queuing delay (Congestion)
    # If CPU > 70%, latency spikes exponentially
    # Exponential congestion curve
    congestion_factor = np.exp((cpu_utilization - 50) / 15) 
    latency += congestion_factor
    
    # 3. Massive Penalty for Overload events (Packet Drops / Retransmissions)
    # If load_label == 2 (Overload), latency is dominated by queue wait
    overload_indices = data['load_label'] == 2
    latency[overload_indices] += np.random.uniform(50, 150, size=overload_indices.sum())
    
    # Clip to realistic values
    latency = np.clip(latency, 1, 1000)
    data['latency'] = latency
    # -------------------------------------------------------------------------

    # Save to CSV
    data.to_csv(output_file, index=False)
    print(f"Data generation complete. Saved to {output_file}")
    print(data.head())
    print("\nClass Distribution:")
    print(data['load_label'].value_counts())

if __name__ == "__main__":
    generate_sdn_traffic_data()
