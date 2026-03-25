import pandas as pd
import streamlit as st
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore') # just ignoring these for now

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
