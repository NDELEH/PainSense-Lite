import pandas as pd
import numpy as np

def generate_biometric_data(num_samples=1000):
    """Generate synthetic biometric data."""
    time = np.arange(num_samples)
    heart_rate = 60 + 10 * np.sin(2 * np.pi * 0.01 * time)  # Simulate heart rate
    skin_conductance = 5 + 2 * np.random.randn(num_samples)  # Simulate sweat levels
    muscle_tension = 10 + 3 * np.sin(2 * np.pi * 0.05 * time)  # Simulate muscle tension
    
    data = pd.DataFrame({
        "time": time,
        "heart_rate": heart_rate,
        "skin_conductance": skin_conductance,
        "muscle_tension": muscle_tension
    })
    return data

if __name__ == "__main__":
    data = generate_biometric_data()
    data.to_csv("data/biometric_data.csv", index=False)
    print("Synthetic biometric data saved to data/biometric_data.csv")
