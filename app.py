import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer

@st.cache_resource
def load_models():
    data = joblib.load("models/classifier.joblib")
    clf = data["clf"]
    labels = data["labels"]

    try:
        embedder = SentenceTransformer("models/sbert_model")
    except:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return clf, labels, embedder

clf, labels, embedder = load_models()

st.title("Fake News Detection & Credibility Analysis")
st.write("Enter a headline or short article to analyze.")

text = st.text_area("Input Text", height=200)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        emb = embedder.encode([text], convert_to_numpy=True)
        probs = clf.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        label = labels[idx]
        confidence = float(probs[idx])
        credibility = int(confidence * 100) if label == "real" else int((1-confidence)*100)

        st.subheader("Prediction")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**Credibility Score:** {credibility}/100")

        st.subheader("Word-level Explanation (LIME)")
        explainer = LimeTextExplainer(class_names=labels)

        def predict_wrapper(texts):
            emb = embedder.encode(texts, convert_to_numpy=True)
            return clf.predict_proba(emb)

        exp = explainer.explain_instance(text, predict_wrapper, num_features=8)
        html = exp.as_html()
        st.components.v1.html(html, height=350, scrolling=True)
