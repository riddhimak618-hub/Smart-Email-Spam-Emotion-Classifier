import streamlit as st

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Email Spam & Emotion Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Hide Streamlit default chrome (header, footer, menu) ──────
st.markdown("""
<style>
    #MainMenu  { visibility: hidden; }
    header     { visibility: hidden; }
    footer     { visibility: hidden; }
    .stApp     { background: #03050f; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── Load HTML file ────────────────────────────────────────────
import os

HTML_FILE = "spam_emotion_classifier_v2.html"

if os.path.exists(HTML_FILE):
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()
else:
    st.error(f"HTML file '{HTML_FILE}' not found! Make sure it's in the same folder.")
    st.stop()

# ── Embed HTML ────────────────────────────────────────────────
st.components.v1.html(html_content, height=2200, scrolling=False)
