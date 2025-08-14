import os
import re
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
    st.image("https://raw.githubusercontent.com/04divya/Result/main/assets/logo_UKM.png", width=80)
with col2:
    st.markdown(f"<h1 style='color:{UKM_RED};'>Transfer Credit Checker System</h1>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color:{UKM_BLUE};'>Universiti Kebangsaan Malaysia</h5>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload Section ---
st.title("Syllabus / Transcript Comparison via OCR")
st.markdown(f"<h3 style='color:{UKM_RED};'>📄 Upload Transcript & Structure File</h3>", unsafe_allow_html=True)

uploaded_ukm = st.file_uploader("Upload Student Transcript (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], key="ukm_file")
uploaded_ipts = st.file_uploader("Upload Structure Courses File (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True, key="ipt_files")

# --- Credit Extraction Function ---
def extract_credits(text):
    """Extract total required credits and credits passed from transcript text."""
    total_required = None
    credits_passed = None
    
    # Look for patterns like JUMLAH CREDIT: 122, LULUS: 110
    total_match = re.search(r"(?:jumlah|total)\s*(?:kredit|credits)?\s*[:\-]?\s*(\d{1,3})", text, re.IGNORECASE)
    passed_match = re.search(r"(?:lulus|passed)\s*(?:kredit|credits)?\s*[:\-]?\s*(\d{1,3})", text, re.IGNORECASE)

    if total_match:
        total_required = int(total_match.group(1))
    if passed_match:
        credits_passed = int(passed_match.group(1))
    
    return total_required, credits_passed

# --- Submit Button ---
if uploaded_ukm and uploaded_ipts and st.button("🚀 Submit for Analysis"):
    st.session_state.similarity_results = []

    with st.spinner("🔍 Processing documents..."):
        # Extract student transcript text
        ukm_text = extract_text_from_file(uploaded_ukm)
        if not ukm_text:
            st.error("Unable to extract text from the student transcript.")
        else:
            # --- NEW: Extract and display credits ---
            total_required, credits_passed = extract_credits(ukm_text)
            if total_required is not None and credits_passed is not None:
                remaining_credits = total_required - credits_passed
                st.markdown(f"**Total Required Credits:** {total_required}")
                st.markdown(f"**Credits Passed:** {credits_passed}")
                st.markdown(f"**Remaining Credits:** {remaining_credits}")
            else:
                st.warning("Could not find credit information in the transcript.")

            # Show transcript classification & text
            ukm_class = classify_document(ukm_text)
            st.markdown("### 📘 Student Transcript")
            st.info(ukm_class)
            st.text_area("Extracted Text (Transcript)", ukm_text, height=200)

            # Process structure file(s)
            for ipt_file in uploaded_ipts:
                ipt_text = extract_text_from_file(ipt_file)
                if not ipt_text:
                    st.warning(f"Unable to extract text from structure file: {ipt_file.name}")
                    continue

                ipt_class = classify_document(ipt_text)
                bert_score = calculate_bert_similarity(ukm_text, ipt_text)
                tfidf_score = calculate_tfidf_similarity(ukm_text, ipt_text)

                st.markdown(f"### 🏫 Structure File: {ipt_file.name}")
                st.info(ipt_class)
                st.text_area("Extracted Text (Structure File)", ipt_text, height=200)
                st.write(f"**BERT Similarity:** {bert_score:.2f}%")
                st.write(f"**TF-IDF Similarity:** {tfidf_score:.2f}%")

                st.session_state.similarity_results.append({
                    "filename": ipt_file.name,
                    "bert": bert_score,
                    "tfidf": tfidf_score
                })

# --- Reset Button ---
st.markdown("---")
if st.button("🔁 Next / Reset"):
    st.session_state.similarity_results = []
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown(f"<p style='text-align:center;color:{UKM_BLUE};'>© 2025 Universiti Kebangsaan Malaysia | Transfer Credit Checker</p>", unsafe_allow_html=True)
