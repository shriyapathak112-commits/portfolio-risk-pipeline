import os
import mlflow
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Initialize Tracking Systems
# -------------------------------------------------------------
# Set up MLflow local tracking directory
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("Quantum_Optimization_Experiment")

# Initialize Weights & Biases
wandb.init(
    project="quantum-research-intern",
    name="mlflow-wandb-integration",
    config={
        "learning_rate": 0.01,
        "epochs": 100,
        "optimizer": "Adam",
        "quantum_layers": 4
    }
)

# Start MLflow run context
with mlflow.start_run():
    
    # ---------------------------------------------------------
    # 2. Log Configuration Parameters
    # ---------------------------------------------------------
    params = wandb.config # retrieve from config dict
    
    # Log to MLflow
    mlflow.log_params(params)

    print("✅ Configuration parameters logged successfully.")

    # ---------------------------------------------------------
    # 3. Generate & Log Mock Artifacts (Circuit & Heatmap)
    # ---------------------------------------------------------
    os.makedirs("artifacts", exist_ok=True)

    # A. Circuit Screenshot Placeholder
    fig_circuit, ax = plt.subplots(figsize=(5, 2))
    ax.text(0.5, 0.5, "--- [Qubit 0] --- [H] --- (X) ---\n--- [Qubit 1] ----------- | ---", 
            fontsize=12, ha='center', va='center', family='monospace')
    ax.axis('off')
    circuit_path = "artifacts/circuit_screenshot.png"
    plt.savefig(circuit_path, bbox_inches='tight')
    plt.close()

    # B. Optimization Heatmap
    fig_heatmap, ax = plt.subplots(figsize=(6, 4))
    data = np.random.rand(10, 10)
    c = ax.imshow(data, cmap='viridis', interpolation='nearest')
    fig_heatmap.colorbar(c)
    ax.set_title("Quantum Parameter Optimization Landscape")
    heatmap_path = "artifacts/optimization_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()

    # Log Artifacts to MLflow
    mlflow.log_artifact(circuit_path)
    mlflow.log_artifact(heatmap_path)
    print("✅ Images and artifacts logged successfully.")

    # ---------------------------------------------------------
    # 4. Log Optimization Metrics (Simulated Run loop)
    # ---------------------------------------------------------
    print("Logging training metrics...")
    for step in range(params["epochs"]):
        # Simulated cost function decay with noise
        loss = 0.5 * np.exp(-step / 20) + 0.02 * np.random.randn()
        fidelity = 1.0 - loss
        
        # Log to MLflow
        mlflow.log_metric("loss", loss, step=step)
        mlflow.log_metric("fidelity", fidelity, step=step)
        
        # Log to W&B
        wandb.log({"loss": loss, "fidelity": fidelity}, step=step)
    # Log Media to W&B
    wandb.log({
        "Circuit Screenshot": wandb.Image(circuit_path),
        "Optimization Heatmap": wandb.Image(heatmap_path)
    })

    print("✅ Optimization metrics tracking finished successfully.")

# Close the W&B run cleanly
wandb.finish()
