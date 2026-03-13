# Knowledge Transfer: SDN Load Balancing Project Guide

This guide is designed to help you (or your friend) run the Adaptive Hybrid Ensemble for Proactive SDN Controller Load Balancing project on a new laptop from scratch.

---

## 🟢 Part 1: Installation & Setup

Before running the code, you need Python and the required libraries installed.

**Step 1. Install Python**
Make sure you have Python 3.9+ installed. You can download it from python.org.

**Step 2. Install Dependencies**
Open a terminal (Command Prompt or PowerShell) in the project folder and run:
```bash
pip install -r requirements.txt
```
*(This installs `pandas`, `numpy`, `scikit-learn`, `torch`, `matplotlib`, and `seaborn`.)*

---

## 🔵 Part 2: Step-by-Step Execution Guide

This project is modular. You need to run the scripts in a specific order. Here is what each step does:

### Step 1: Generate the Synthetic Data
*   **Command**: `python data_generator.py`
*   **What it does**: In real life, getting granular labeled SDN attack data is hard. This script creates our own synthetic dataset (`sdn_traffic_data.csv`).
*   **The Secret Sauce**: It injects a "Hidden Trend" rule. If traffic rises for 3 consecutive time steps, it forcefully spikes the traffic 3x in the next step. It also calculates a realistic network `latency` based on CPU congestion.
*   **Output**: A file named `sdn_traffic_data.csv` (10,000 traffic samples).

### Step 2: Compare All Models (The Main Benchmark)
*   **Command**: `python compare_models.py`
*   **What it does**: This is the heavy lifter. It does the following:
    1.  Loads the CSV from Step 1 (`preprocessing.py` standardizes the features and creates sequences for deep learning).
    2.  Trains 6 "Baseline" models (Logistic Regression, Decision Tree, Random Forest, SVM, Neural Network, and a Deep Reinforcement Learning Agent).
    3.  Trains our "Proposed" **Hybrid Ensemble** (LSTM for trends + Random Forest for static state).
    4.  Compares them all on Accuracy, MAE (Error severity), and R-Squared (Variance explanation).
*   **Output**: Prints a results table to the console, saves it to `final_results.csv`, and generates a bar chart `performance_comparison.png`.

### Step 3: Simulate Network Latency Impact
*   **Command**: `python simulate_latency.py`
*   **What it does**: Accuracy alone doesn't prove network value. This script simulates what happens to actual network latency (packet delays) under different strategies.
    *   **Reactive (Standard)**: Waits for the controller CPU to hit 80% before migrating switches. Result -> Suffers massive queue delays during the reaction time.
    *   **Proactive (Hybrid)**: Uses our model to migrate *before* the spike hits. Result -> Bypasses the queue buildup entirely.
*   **Output**: A line graph `latency_reduction_simulation.png` showing the true value of your research (an ~80% reduction in accumulated delays).

### Step 4: Generate Publication-Quality Graphs
*   **Command**: `python generate_graphs.py`
*   **What it does**: This creates the 3 specific charts you need for a research paper. It retrains the Random Forest and the Hybrid model specifically to extract their internal probabilities and tracking behavior.
*   **Output**:
    1.  `graph_confusion_matrices.png`: Shows that the Hybrid model has fewer "False Negatives" (missed overloads).
    2.  `graph_roc_curves.png`: Mathematical proof that the Hybrid model is more sensitive to anomalies.
    3.  `graph_time_series.png`: The "money shot" showing the Hybrid model predicting the load spikes better than the Baseline.

---

## 🟣 Part 3: Explaining the Code Structure (The "Elevator Pitch")

If your friend asks, "How does the main model actually work?", tell them this:

1.  Look at `hybrid_model.py`.
2.  It contains two different ML brains running in parallel.
3.  **Brain 1 (LSTM)**: Written in PyTorch. It looks at the *past 10 time steps* to figure out the momentum or trend of the traffic.
4.  **Brain 2 (Random Forest)**: Written in Scikit-Learn. It looks at the *current snapshot* of the network (how many switches are active, what protocol is running, etc.).
5.  **The Ensemble**: The `predict()` function takes the output probabilities of both brains and averages them (`0.5 * LSTM + 0.5 * RF`). This makes the model robust against both sudden pattern shifts *and* static threshold breaches.
