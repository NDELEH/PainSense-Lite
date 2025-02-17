from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load synthetic data
data = pd.read_csv("data/biometric_data.csv")

# Simulate pain labels (0 = no pain, 1 = pain)
data["pain"] = (data["muscle_tension"] > 12).astype(int)

# Train a model
X = data[["heart_rate", "skin_conductance", "muscle_tension"]]
y = data["pain"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "models/pain_model.pkl")
print("Model saved to models/pain_model.pkl")
