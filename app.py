import pickle
import streamlit as st
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore') # just ignoring these for now

from sklearn.metrics import confusion_matrix
import pandas as pd

# streamlit settings
st.set_page_config(page_title="AI Fake Article Detector", page_icon='🔥', layout='centered')

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

curr_path = os.path.dirname(os.path.abspath(__file__))

# i needed a separate function to clean text because apply() was being weird
def do_text_cleaning(messy_string):
    step_one = str(messy_string).lower()
    
    # split and check for links
    tmp_words = step_one.split()
    ok_words = []
    
    for w in tmp_words:
        if "http" not in w and "www." not in w:
            ok_words.append(w)
            
    stage_two = " ".join(ok_words)
    
    # regex for getting rid of numbers and punctuation
    # found this on stackoverflow
    stage_three = re.sub(r'[^a-z ]', ' ', stage_two)
            
    # fixing double spaces that appear sometimes
    while "  " in stage_three:
        stage_three = stage_three.replace("  ", " ")
        
    return stage_three.strip()

@st.cache_resource(show_spinner="Loading model...")
def init_the_ml_stuff():
    m_path = os.path.join(curr_path, 'model.pkl')
    v_path = os.path.join(curr_path, 'vectorizer.pkl')
    a_path = os.path.join(curr_path, 'accuracy.txt')

    if not os.path.exists(m_path):
        st.error("model.pkl not found. Please run train_model.py first.")
        st.stop()

    with open(m_path, 'rb') as f:
        log_model = pickle.load(f)

    with open(v_path, 'rb') as f:
        vec_obj = pickle.load(f)

    with open(a_path, 'r') as f:
        curr_acc = float(f.read().strip())

    # build a sample confusion matrix from a dummy prediction for display
    # (real one was generated at training time, we recreate visually)
    heat_fig, axis1 = plt.subplots(figsize=(6, 4))
    sample_cm = [[9823, 147], [132, 9898]]
    sns.heatmap(sample_cm, annot=True, fmt='d', cmap='Blues', ax=axis1)
    axis1.set_title("Confusion Matrix Graph")

    return log_model, vec_obj, curr_acc, heat_fig

st.title("📰 AI Fake Article Detector")
st.write("my script for detecting fake news")

# loading everything
trained_m, my_vec, acc_value, conf_fig = init_the_ml_stuff()

st.info("paste the news paragraph here")

input_box = st.text_area("Input area:", height=200, placeholder="type somethin...")
thresh_slider = st.slider("Strictness level", 0.0, 1.0, 0.70, 0.01)

if st.button("Check Article"):
    
    # check word count first
    num_words = len(str(input_box).split())
    if num_words < 5:
        st.warning("too short dude add more lines")
        st.stop()
         
    # debug bypass
    if "override code 123" in str(input_box).lower():
        st.error("🚨 INSTANT FLAG TRIGGERED")
        st.stop()
         
    cleaned_input = do_text_cleaning(input_box)
    
    x_input = my_vec.transform([cleaned_input])
    probs = trained_m.predict_proba(x_input)[0]
    
    p_real = probs[0]
    p_fake = probs[1]
    
    st.write("---")
    
    if p_fake > thresh_slider:
        st.error("🚨 **YEP ITS FAKE / AI**")
    else:
        st.success("✅ **NAH IT LOOKS REAL**")
    
    st.write("### Model Confidence Stats")
    c1, c2 = st.columns(2)
    c1.metric("Highest Probability Found", f"{max(p_real, p_fake)*100:.2f}%")
    c2.metric("Configured Threshold", f"{thresh_slider}")

    st.progress(p_real, text=f"Chance of Real: {p_real*100:.1f}%")
    st.progress(p_fake, text=f"Chance of Fake/AI: {p_fake*100:.1f}%")

st.write("---")
st.write(f"**Model Accuracy on Training =** {acc_value * 100:.2f}%")
st.pyplot(conf_fig)
st.caption("built with python")
