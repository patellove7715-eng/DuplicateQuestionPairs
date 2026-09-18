import streamlit as st
import helper
import pickle
import gzip
import os

st.set_page_config(
    page_title="Quora Duplicate Question Pairs",
    page_icon="❓",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_GZ_PATH = os.path.join(BASE_DIR, 'model.pkl.gz')
MODEL_PKL_PATH = os.path.join(BASE_DIR, 'model.pkl')

@st.cache_resource
def load_model():
    # 1. First try loading the bundled compressed model (24MB, included in repo)
    if os.path.exists(MODEL_GZ_PATH) and os.path.getsize(MODEL_GZ_PATH) > 0:
        try:
            with gzip.open(MODEL_GZ_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Note loading compressed model: {e}")

    # 2. Fallback to uncompressed model.pkl if available locally
    if os.path.exists(MODEL_PKL_PATH) and os.path.getsize(MODEL_PKL_PATH) > 0:
        try:
            with open(MODEL_PKL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Note loading uncompressed model: {e}")

    return None

st.title("❓ Quora Duplicate Question Pairs")
st.markdown(
    "Detect whether two questions have the same meaning or intent using Machine Learning & NLP."
)

with st.spinner("Loading machine learning model..."):
    model = load_model()

if model is None:
    st.error("⚠️ **Model file could not be loaded.** Please check repository files.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    q1 = st.text_area("Question 1", placeholder="e.g. How can I learn Python effectively?", height=120)
with col2:
    q2 = st.text_area("Question 2", placeholder="e.g. What is the best way to study Python?", height=120)

if st.button("Check Duplicate", type="primary", use_container_width=True):
    if not q1.strip() or not q2.strip():
        st.warning("⚠️ Please enter both questions before analyzing.")
    else:
        with st.spinner("Analyzing semantic similarity..."):
            query = helper.query_point_creator(q1, q2)
            result = model.predict(query)[0]
            
            # Predict probabilities if supported by the model
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(query)[0]
                dup_prob = proba[1] * 100
                non_dup_prob = proba[0] * 100
            else:
                dup_prob = 100.0 if result == 1 else 0.0
                non_dup_prob = 100.0 - dup_prob

            st.divider()
            if result == 1:
                st.success(f"✅ **Duplicate Question Pair** (Confidence: {dup_prob:.1f}%)")
            else:
                st.info(f"ℹ️ **Not Duplicate Questions** (Confidence: {non_dup_prob:.1f}%)")
