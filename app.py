import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DepthDrift",
    page_icon="🧠",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("depth_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("depth_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Predict function ───────────────────────────────────────────────────────
def predict(sleep, study, social, mood, exercise, screen):
    cols = ["sleep_hours", "study_hours", "social_score",
            "mood_score", "exercise_minutes", "screen_hours"]
    X = pd.DataFrame([[sleep, study, social, mood, exercise, screen]], columns=cols)
    X_scaled = scaler.transform(X)
    raw = model.decision_function(X_scaled)[0]
    depth_score = int(np.clip(((-raw + 0.1) / 0.3) * 100, 0, 100))
    label = model.predict(X_scaled)[0]
    return depth_score, label

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🧠 DepthDrift")
st.caption("AI-powered early warning system for student burnout")
st.markdown("---")

st.subheader("📋 How are you doing today?")
st.markdown("Move the sliders to reflect your day:")

col1, col2 = st.columns(2)

with col1:
    sleep    = st.slider("😴 Sleep (hours)",        0.0, 10.0, 7.0, 0.5)
    study    = st.slider("📚 Study (hours)",         0.0, 10.0, 4.0, 0.5)
    social   = st.slider("👥 Social energy (0–10)",  0.0, 10.0, 6.0, 0.5)

with col2:
    mood     = st.slider("😊 Mood (0–10)",           0.0, 10.0, 6.5, 0.5)
    exercise = st.slider("🏃 Exercise (minutes)",    0.0, 90.0, 30.0, 5.0)
    screen   = st.slider("📱 Screen time (hours)",   0.0, 12.0, 4.0, 0.5)

st.markdown("---")

if st.button("🔍 Analyse My Day", use_container_width=True):
    depth_score, label = predict(sleep, study, social, mood, exercise, screen)

    st.markdown("### 📊 Your Results")

    # Depth score bar
    st.metric("Burnout Risk Score", f"{depth_score} / 100")
    st.progress(depth_score / 100)

    # Status
    if label == -1:
        st.error("⚠️ At Risk — Your patterns suggest you may be descending into burnout.")
        st.markdown("""
**Recommendations:**
- 🛌 Try to get at least 7–8 hours of sleep tonight
- 📵 Reduce screen time by 1–2 hours
- 🚶 Take a short walk or stretch break
- 💬 Talk to a friend or counselor
        """)
    else:
        if depth_score > 40:
            st.warning("🟡 Mild Concern — Some early signs detected. Keep an eye on your habits.")
        else:
            st.success("✅ Healthy — Your behavioral patterns look balanced. Keep it up!")

    st.markdown("---")
    st.caption("DepthDrift uses an Isolation Forest ML model trained on student behavioral patterns. "
               "This is a research tool — not a medical diagnosis.")
