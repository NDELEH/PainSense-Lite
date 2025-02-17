import time
import random
import joblib

# Load the model
model = joblib.load("models/pain_model.pkl")

def simulate_real_time_monitoring():
    """Simulate real-time pain monitoring."""
    while True:
        # Generate random biometric data
        heart_rate = random.uniform(60, 100)
        skin_conductance = random.uniform(3, 7)
        muscle_tension = random.uniform(8, 15)
        
        # Predict pain
        prediction = model.predict([[heart_rate, skin_conductance, muscle_tension]])
        if prediction == 1:
            print("⚠️ Pain detected! Try deep breathing or stretching.")
        else:
            print("✅ No pain detected.")
        time.sleep(1)  # Simulate 1-second intervals

if __name__ == "__main__":
    simulate_real_time_monitoring()
