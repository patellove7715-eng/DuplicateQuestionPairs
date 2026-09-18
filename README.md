# ❓ Quora Duplicate Question Pairs Detector

An end-to-end Machine Learning and Natural Language Processing web application built with **Streamlit** to identify whether two questions posted on Quora have the same semantic intent (duplicate pairs).

---

## 📌 Overview

Duplicate questions are a common challenge on community question-answering platforms like Quora. Identifying duplicates helps:
- Provide instant answers to users by linking to existing high-quality discussions.
- Prevent fragmentation of knowledge across multiple redundant threads.
- Improve search and recommendation relevance.

---

## 🛠️ Architecture & Feature Engineering

The pipeline extracts **22 engineered NLP features** along with **Bag-of-Words (CountVectorizer)** representation:

### 1. Basic Features
- Character length of Question 1 and Question 2
- Word count of Question 1 and Question 2
- Total number of unique words across both questions
- Number of common words between both questions
- Word share ratio (`common_words / total_words`)

### 2. Advanced Token & Length Features
- **cwc_min / cwc_max**: Common non-stopwords ratio over min/max word counts
- **csc_min / csc_max**: Common stopwords ratio over min/max stopword counts
- **ctc_min / ctc_max**: Common token ratio over min/max token counts
- **first_word_eq / last_word_eq**: Binary flags for whether first/last words match
- **abs_len_diff**: Absolute word count difference
- **mean_len**: Average token length across both questions
- **longest_substr_ratio**: Longest common substring ratio

### 3. Fuzzy Matching Features
- `fuzz.QRatio`
- `fuzz.partial_ratio`
- `fuzz.token_sort_ratio`
- `fuzz.token_set_ratio`

### 4. Vectorization & Classifier
- **CountVectorizer**: 3,000 top n-gram features for Question 1 and Question 2.
- **Classifier**: Random Forest Classifier trained on 6,022 total features.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Duplicate_Question_Pairs.git
cd Duplicate_Question_Pairs
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📦 Project Structure

```text
├── app.py              # Streamlit web UI application
├── helper.py           # NLP text preprocessing and feature extraction pipeline
├── model.pkl           # Trained Random Forest classifier
├── cv.pkl              # Fitted CountVectorizer vocabulary
├── requirements.txt    # Python dependencies
├── Procfile            # Cloud deployment configuration
├── setup.sh            # Streamlit server startup script
├── .gitignore          # Git exclusion rules
└── README.md           # Documentation
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
