from baselines import train_baselines
from train_hybrid import train_and_evaluate_hybrid
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("========================================")
    print("    SDN LOAD BALANCING MODEL EVALUATION    ")
    print("========================================")
    
    # 1. Run Baselines
    print("\n[1/2] Running Baseline Models...")
    baseline_results = train_baselines()
    
    # 2. Run Hybrid Model
    print("\n[2/2] Running Proposed Hybrid Ensemble...")
    hybrid_result = train_and_evaluate_hybrid()
    
    # 3. Aggregate Results
    all_results = baseline_results + [hybrid_result]
    
    print("\n========================================")
    print("           FINAL RESULTS                ")
    print("========================================")
    
    results_df = pd.DataFrame(all_results)
    
    # Calculate Improvement vs RF (Baseline) for Accuracy
    rf_acc = results_df.loc[results_df['Model'] == 'Random Forest', 'Accuracy'].values[0]
    results_df['Improvement (Acc)'] = ((results_df['Accuracy'] - rf_acc) / rf_acc) * 100
    
    print(results_df)
    
    # Save results
    results_df.to_csv('final_results.csv', index=False)
    print("\nResults saved to 'final_results.csv'.")
    
    # Generate Comparison Plot (Accuracy Only for clarity)
    plt.figure(figsize=(12, 6))
    colors = ['#bdc3c7'] * (len(results_df) - 1) + ['#2ecc71'] # Last one is Green
    bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors)
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title('SDN Load Prediction Model Comparison')
    plt.xticks(rotation=45, ha='right')
    
    # Add text labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Comparison plot saved to 'performance_comparison.png'.")

if __name__ == "__main__":
    main()
