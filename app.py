import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["USE_TF"] = "0" 
os.environ["USE_TORCH"] = "1" 


# Fix for PyTorch + Streamlit + asyncio environment issue
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from utils.file_utils import extract_text_from_file
from utils.similarity_utils import calculate_bert_similarity, calculate_tfidf_similarity
from utils.classification import classify_document
from PIL import Image

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')


# --- Configuration ---
UKM_RED = "#E60000"
UKM_BLUE = "#0066B3"

st.set_page_config(page_title="UKM Transfer Credit Checker", layout="centered")

# --- UI Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://raw.githubusercontent.com/khaliesahazmin/DataExtraction/main/assets/logo_UKM.png", width=80)
with col2:
    st.markdown(f"<h1 style='color:{UKM_RED};'>Transfer Credit Checker System</h1>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color:{UKM_BLUE};'>Universiti Kebangsaan Malaysia</h5>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload Section ---
st.title("Syllabus Comparison via OCR")
st.markdown(f"<h3 style='color:{UKM_RED};'>📄 Upload Syllabus Documents</h3>", unsafe_allow_html=True)

uploaded_ukm = st.file_uploader("Upload UKM Syllabus (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], key="ukm_file")
uploaded_ipts = st.file_uploader("Upload IPT Syllabus Documents (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True, key="ipt_files")

# --- Submit Button ---
if uploaded_ukm and uploaded_ipts and st.button("🚀 Submit for Analysis"):
    st.session_state.similarity_results = []

    with st.spinner("🔍 Processing documents..."):
        ukm_text = extract_text_from_file(uploaded_ukm)
        if not ukm_text:
            st.error("Unable to extract text from the UKM syllabus.")
        else:
            ukm_class = classify_document(ukm_text)
            st.markdown("### 📘 UKM Syllabus Document")
            st.info(ukm_class)
            st.text_area("Extracted Text (UKM)", ukm_text, height=200)

            for ipt_file in uploaded_ipts:
                ipt_text = extract_text_from_file(ipt_file)
                if not ipt_text:
                    st.warning(f"Unable to extract text from IPT file: {ipt_file.name}")
                    continue

                ipt_class = classify_document(ipt_text)
                bert_score = calculate_bert_similarity(ukm_text, ipt_text)
                tfidf_score = calculate_tfidf_similarity(ukm_text, ipt_text)

                st.markdown(f"### 🏫 IPT Document: {ipt_file.name}")
                st.info(ipt_class)
                st.text_area("Extracted Text (IPT)", ipt_text, height=200)
                st.write(f"**BERT Similarity:** {bert_score:.2f}%")
                st.write(f"**TF-IDF Similarity:** {tfidf_score:.2f}%")

                st.session_state.similarity_results.append({
                    "filename": ipt_file.name,
                    "bert": bert_score,
                    "tfidf": tfidf_score
                })

# --- Reset Button ---
st.markdown("---")
if st.button("🔁 Next Course / Reset"):
    st.session_state.similarity_results = []
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown(f"<p style='text-align:center;color:{UKM_BLUE};'>© 2025 Universiti Kebangsaan Malaysia | Transfer Credit Checker</p>", unsafe_allow_html=True)
