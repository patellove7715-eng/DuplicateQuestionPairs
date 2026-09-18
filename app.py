import streamlit as st
import helper
import pickle
import os
import urllib.request

st.set_page_config(
    page_title="Quora Duplicate Question Pairs",
    page_icon="❓",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Optional custom download URL via Streamlit Secrets or Environment Variable
DEFAULT_MODEL_URL = "https://github.com/patellove7715-eng/DuplicateQuestionPairs/releases/download/v1.0.0/model.pkl"

def get_model_url():
    try:
        if "MODEL_URL" in st.secrets:
            return st.secrets["MODEL_URL"]
    except Exception:
        pass
    return os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)

def download_model(url, target_path):
    try:
        urllib.request.urlretrieve(url, target_path)
        return True
    except Exception:
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        return False

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        url = get_model_url()
        with st.spinner("📦 Downloading trained model file (approx. 156MB) for the first time... Please wait a moment."):
            success = download_model(url, MODEL_PATH)
            if not success or not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
                return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

st.title("❓ Quora Duplicate Question Pairs")
st.markdown(
    "Detect whether two questions have the same meaning or intent using Machine Learning & NLP."
)

model = load_model()

if model is None:
    st.error("⚠️ **Model file (`model.pkl`) is missing on Streamlit Cloud.**")
    st.markdown(
        """
        Because `model.pkl` exceeds GitHub's **100 MB** file limit, it is excluded by `.gitignore`.

        ### 🔧 How to Fix in 2 Minutes:
        1. **Create a GitHub Release:**
           - Go to: [GitHub Releases New](https://github.com/patellove7715-eng/DuplicateQuestionPairs/releases/new)
           - Enter tag name: `v1.0.0` and title: `Initial Model Release`
           - Drag & drop your local `model.pkl` (156MB) into the **"Attach binaries by dropping them here"** box.
           - Click **Publish release**.
        2. Refresh this Streamlit app! It will automatically download and cache the model.
        """
    )
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
