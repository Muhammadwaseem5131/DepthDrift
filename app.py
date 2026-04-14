import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DepthDrift", page_icon="🧠", layout="centered")

@st.cache_resource
def train_model():
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
    features = ["sleep_hours","study_hours","social_score","mood_score","exercise_minutes","screen_hours"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    model.fit(X_scaled)
    return model, scaler

model, scaler = train_model()

def predict(sleep, study, social, mood, exercise, screen):
    cols = ["sleep_hours","study_hours","social_score","mood_score","exercise_minutes","screen_hours"]
    X = pd.DataFrame([[sleep, study, social, mood, exercise, screen]], columns=cols)
    X_scaled = scaler.transform(X)
    raw = model.decision_function(X_scaled)[0]
    depth_score = int(np.clip(((-raw + 0.1) / 0.3) * 100, 0, 100))
    label = model.predict(X_scaled)[0]
    return depth_score, label

st.title("🧠 DepthDrift")
st.caption("AI-powered early warning system for student burnout")
st.markdown("---")
st.subheader("📋 How are you doing today?")
st.markdown("Move the sliders to reflect your day:")

col1, col2 = st.columns(2)
with col1:
    sleep    = st.slider("😴 Sleep (hours)",       0.0, 10.0, 7.0, 0.5)
    study    = st.slider("📚 Study (hours)",        0.0, 10.0, 4.0, 0.5)
    social   = st.slider("👥 Social energy (0–10)", 0.0, 10.0, 6.0, 0.5)
with col2:
    mood     = st.slider("😊 Mood (0–10)",          0.0, 10.0, 6.5, 0.5)
    exercise = st.slider("🏃 Exercise (minutes)",   0.0, 90.0, 30.0, 5.0)
    screen   = st.slider("📱 Screen time (hours)",  0.0, 12.0, 4.0, 0.5)

st.markdown("---")
if st.button("🔍 Analyse My Day", use_container_width=True):
    depth_score, label = predict(sleep, study, social, mood, exercise, screen)
    st.markdown("### 📊 Your Results")
    st.metric("Burnout Risk Score", f"{depth_score} / 100")
    st.progress(depth_score / 100)
    if label == -1:
        st.error("⚠️ At Risk — Your patterns suggest you may be descending into burnout.")
        st.markdown("**Recommendations:**\n- 🛌 Get at least 7–8 hours of sleep\n- 📵 Reduce screen time\n- 🚶 Take a short walk\n- 💬 Talk to a friend or counselor")
    else:
        if depth_score > 40:
            st.warning("🟡 Mild Concern — Some early signs detected. Keep an eye on your habits.")
        else:
            st.success("✅ Healthy — Your behavioral patterns look balanced. Keep it up!")
    st.markdown("---")
    st.caption("DepthDrift uses an Isolation Forest ML model. This is a research tool — not a medical diagnosis.")
