import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

# ── 1. Generate synthetic student behavioral data ──────────────────────────
np.random.seed(42)
n_normal = 500

normal_data = pd.DataFrame({
    "sleep_hours":      np.random.normal(7.0, 0.8, n_normal).clip(4, 10),
    "study_hours":      np.random.normal(4.0, 1.0, n_normal).clip(0, 10),
    "social_score":     np.random.normal(6.0, 1.2, n_normal).clip(0, 10),
    "mood_score":       np.random.normal(6.5, 1.0, n_normal).clip(0, 10),
    "exercise_minutes": np.random.normal(30,  10,  n_normal).clip(0, 90),
    "screen_hours":     np.random.normal(4.0, 1.0, n_normal).clip(1, 12),
})

# Burnout/anomaly patterns — low sleep, low mood, high screen, low social
n_anomaly = 60
anomaly_data = pd.DataFrame({
    "sleep_hours":      np.random.normal(4.5, 0.7, n_anomaly).clip(2, 6),
    "study_hours":      np.random.normal(1.5, 1.0, n_anomaly).clip(0, 4),
    "social_score":     np.random.normal(2.5, 1.0, n_anomaly).clip(0, 5),
    "mood_score":       np.random.normal(2.5, 1.0, n_anomaly).clip(0, 4),
    "exercise_minutes": np.random.normal(5,   5,   n_anomaly).clip(0, 20),
    "screen_hours":     np.random.normal(8.0, 1.5, n_anomaly).clip(5, 12),
})

df = pd.concat([normal_data, anomaly_data], ignore_index=True)

# ── 2. Train model ──────────────────────────────────────────────────────────
features = ["sleep_hours", "study_hours", "social_score",
            "mood_score", "exercise_minutes", "screen_hours"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)
model.fit(X_scaled)

# ── 3. Save model and scaler ────────────────────────────────────────────────
with open("depth_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("depth_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model trained and saved successfully!")

# ── 4. Quick self-test ──────────────────────────────────────────────────────
def predict_burnout(sleep, study, social, mood, exercise, screen):
    """
    Returns a dict with:
      - label: 'At Risk' or 'Healthy'
      - depth_score: 0-100 (higher = more at risk)
      - raw_score: Isolation Forest anomaly score
    """
    X = scaler.transform([[sleep, study, social, mood, exercise, screen]])
    raw = model.decision_function(X)[0]   # negative = more anomalous
    # Normalise to 0–100 risk score (100 = highest risk)
    depth_score = int(np.clip(((-raw + 0.1) / 0.5) * 100, 0, 100))
    label = "At Risk" if model.predict(X)[0] == -1 else "Healthy"
    return {"label": label, "depth_score": depth_score, "raw": raw}

# Test with a healthy profile
healthy = predict_burnout(7.5, 4, 7, 7, 40, 3)
print(f"Healthy profile  → {healthy['label']} | Depth Score: {healthy['depth_score']}")

# Test with a burnout profile
burnout = predict_burnout(4.0, 1, 2, 2, 5, 9)
print(f"Burnout profile  → {burnout['label']} | Depth Score: {burnout['depth_score']}")

if __name__ == "__main__":
    pass
