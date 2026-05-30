import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def generate_universe_data(n_assets: int, n_periods: int = 252) -> pd.DataFrame:
    """Generates clean synthetic prices for an asset universe size N."""
    np.random.seed(42 + n_assets)
    dates = pd.date_range(start="2026-01-01", periods=n_periods, freq='B')
    asset_data = {}
    for i in range(n_assets):
        asset_name = f"Asset_{i+1:02d}"
        drift = np.random.uniform(0.0001, 0.001)
        volatility = np.random.uniform(0.005, 0.02)
        shocks = np.random.normal(drift, volatility, n_periods)
        asset_data[asset_name] = 100 * np.exp(shocks.cumsum())
    return pd.DataFrame(asset_data, index=dates)
def validate_dataset(returns_df: pd.DataFrame, cov_matrix: pd.DataFrame) -> dict:
    """Checks for missing data and confirms the matrix is PSD."""
    validation_results = {}
    
    # Check for missing data
    missing_count = returns_df.isnull().sum().sum()
    validation_results['no_missing_values'] = (missing_count == 0)
    
    # Check shape alignment
    validation_results['data_consistency'] = (cov_matrix.shape[0] == cov_matrix.shape[1] == returns_df.shape[1])
    
    # PSD Verification via Eigenvalues
    eigenvalues = np.linalg.eigvals(cov_matrix)
    is_psd = np.all(eigenvalues >= -1e-8)
    validation_results['positive_semi_definite'] = bool(is_psd)
    
    return validation_results
if __name__ == "__main__":
    universe_sizes = [5, 10, 20, 30, 50]
    summary_report = []
    
    for N in universe_sizes:
        prices = generate_universe_data(n_assets=N)
        returns = prices.pct_change().dropna()
        
        # Ledoit-Wolf Shrinkage
        lw = LedoitWolf()
        lw.fit(returns)
        cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        
        # Run scripts
        checks = validate_dataset(returns, cov_matrix)
        
        # Save benchmarks
        returns.to_csv(f"data/benchmarks/returns_universe_{N}.csv")
        cov_matrix.to_csv(f"data/benchmarks/covariance_universe_{N}.csv")
        
        summary_report.append({
            'Universe Size (N)': N,
            'No Missing Values': 'PASSED' if checks['no_missing_values'] else 'FAILED',
            'Data Consistency': 'PASSED' if checks['data_consistency'] else 'FAILED',
            'PSD Matrix Verification': 'PASSED' if checks['positive_semi_definite'] else 'FAILED'
        })
        
    print("\n" + "="*50 + "\nSCALING DOCUMENTATION SUMMARY\n" + "="*50)
    print(pd.DataFrame(summary_report).to_string(index=False))
